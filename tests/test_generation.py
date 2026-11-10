from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import httpx
import pytest

from eval.generation import score_item, summarize
from eval.harness import DEFAULT_QUESTION_FILE, act_source_horizon, load_eval_items
from eval.tune_abstention import choose
from generation.answer import Answerer
from generation.config import TUNED_ABSTAIN_THRESHOLD, GenerationConfig
from generation.context import Chunk, Context
from generation.horizon import (
    AMENDMENTS_FILE,
    RAW,
    knowledge_horizons,
    rate_amendments,
    rate_horizon_details,
    rate_notification_dates,
    rates_horizon,
)
from generation.llm import LLMClient, LLMError
from generation.prompts import chunk_label, generator_prompt
from generation.render import DISCLAIMER, render_abstention, render_answer
from generation.schemas import Claim, GenerationOutput
from generation.verify import verify
from gst_copilot.models import EvalItem
from retrieval.corpus import RATE_NOTIFICATIONS
from retrieval.store import OPEN_END


def chunk(chunk_id: str, path: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id, chunk_id.split("#")[0], path, "rule", "2021-05-18", "9999-12-31", text, 0.9, False
    )


CONTEXT = [
    chunk("CGST/r36/ss4@v5", "CGST/r36/ss4"),
    chunk("CTR/9-2025/schI@v1#3", "CTR/9-2025/schI"),
]


# -- LLM client -------------------------------------------------------------------------


def gemini_reply(text: str, model: str = "m") -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 10},
        "modelVersion": model,
    }


def client(handler, tmp_path, **kwargs) -> LLMClient:
    return LLMClient(
        "gemini",
        model="primary",
        fallback="backup",
        api_key="k",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        **kwargs,
    )


def test_llm_sends_temperature_zero_json_mode_and_caches(tmp_path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=gemini_reply('{"ok": true}'))

    llm = client(handler, tmp_path)
    first = llm.complete("sys", "prompt", schema={"type": "OBJECT"})
    second = llm.complete("sys", "prompt", schema={"type": "OBJECT"})
    assert first.json() == {"ok": True} and not first.cached
    assert second.cached and second.text == first.text
    assert len(calls) == 1
    config = calls[0]["generationConfig"]
    assert config["temperature"] == 0
    assert config["responseMimeType"] == "application/json"


def test_llm_retries_transient_errors_with_backoff(tmp_path) -> None:
    replies = iter([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(503)])
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return next(replies, httpx.Response(200, json=gemini_reply("{}")))

    llm = LLMClient(
        "gemini",
        model="primary",
        api_key="k",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=slept.append,
    )
    assert llm.complete("s", "p").model == "primary"
    assert len(slept) == 2 and slept[0] >= 7  # Retry-After honoured


def test_llm_falls_back_when_primary_is_missing(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/primary:" in request.url.path:
            return httpx.Response(404, text="model not found")
        return httpx.Response(200, json=gemini_reply("{}"))

    response = client(handler, tmp_path).complete("s", "p")
    assert response.model == "backup"


def test_llm_stops_on_daily_quota_without_fallback(tmp_path) -> None:
    from generation.llm import QuotaExhausted

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        error = {
            "code": 429,
            "message": "Quota exceeded",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13367s"}
            ],
        }
        return httpx.Response(429, json={"error": error})

    with pytest.raises(QuotaExhausted, match="Quota exceeded.*13367s"):
        client(handler, tmp_path).complete("s", "p")
    assert len(calls) == 1  # no retries, no fallback model


def test_llm_gives_up_after_retries(tmp_path) -> None:
    llm = client(lambda request: httpx.Response(500), tmp_path, max_retries=1)
    with pytest.raises(LLMError, match="primary.*backup|backup"):
        llm.complete("s", "p")


def test_llm_without_key_serves_only_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("generation.llm.load_env", lambda: None)
    llm = LLMClient(
        "gemini",
        model="primary",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
    )
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        llm.complete("s", "p")
    assert llm.count_tokens("s" * 40, "p" * 40).method == "estimate"


def test_groq_uses_openai_chat_format(tmp_path) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        reply = {"choices": [{"message": {"content": '{"verdict": "correct"}'}}], "usage": {}}
        return httpx.Response(200, json=reply)

    llm = LLMClient(
        "groq",
        model="judge",
        api_key="g",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
    )
    assert llm.complete("s", "p").json() == {"verdict": "correct"}
    assert seen["temperature"] == 0 and seen["response_format"] == {"type": "json_object"}
    assert seen["auth"] == "Bearer g"


# -- verify and render ------------------------------------------------------------------


def test_verify_drops_unknown_citations_then_unsupported_claims() -> None:
    output = GenerationOutput(
        claims=[
            Claim(text="kept", citation_ids=["CGST/r36/ss4@v5", "CGST/r36/ss4@v4"]),
            Claim(text="dropped", citation_ids=["CGST/s17/ss5@v1"]),
            Claim(text="no citation", citation_ids=[]),
        ],
        ref_date_used="2025-01-01",
    )
    checked = verify(output, CONTEXT, "2025-01-01")
    assert [c.text for c in checked.claims] == ["kept"]
    assert checked.claims[0].citation_ids == ["CGST/r36/ss4@v5"]
    assert [c.text for c in checked.dropped_claims] == ["dropped", "no citation"]
    assert checked.dropped_citations == ["CGST/r36/ss4@v4", "CGST/s17/ss5@v1"]
    assert checked.ref_date_matches
    assert not verify(output, CONTEXT, "2024-01-01").ref_date_matches


def test_render_fixed_template_sections_and_labels() -> None:
    claims = [
        Claim(text="Rate is 2.5 per cent.", citation_ids=["CTR/9-2025/schI@v1#3"]),
        Claim(text="Only if reported.", citation_ids=["CGST/r36/ss4@v5"], kind="condition"),
    ]
    text = render_answer(claims, [], date(2025, 10, 1), CONTEXT)
    order = [
        "**Conclusion**",
        "**Conditions**",
        "**Exceptions**",
        "**Assumptions made**",
        "**Reference date used:** 2025-10-01",
        DISCLAIMER,
    ]
    positions = [text.index(heading) for heading in order]
    assert positions == sorted(positions)
    assert "[Notification No. 9/2025-Central Tax (Rate), Schedule I]" in text
    assert "[CGST Rules, rule 36(4)]" in text
    abstained = render_abstention("Out of scope.", date(2025, 10, 1), CONTEXT)
    assert abstained.startswith("**Not enough information**") and "rule 36(4)" in abstained


def test_chunk_labels_and_prompt_carry_ids() -> None:
    assert chunk_label("CGST/s17/ss5/cl(a)") == "CGST Act, section 17(5)(a)"
    assert chunk_label("CTR/2-2017") == "Notification No. 2/2017-Central Tax (Rate)"
    prompt = generator_prompt("Q?", date(2025, 10, 1), CONTEXT)
    assert "Reference date: 2025-10-01" in prompt
    assert "[ID: CTR/9-2025/schI@v1#3]" in prompt and "in force 2021-05-18 to date" in prompt


# -- metrics ----------------------------------------------------------------------------


def item(answerable: bool, gold: list[str]) -> EvalItem:
    return EvalItem(
        question="q",
        ref_date=date(2025, 1, 1),
        gold_paths=gold,
        gold_answer="a",
        category="direct_lookup" if answerable else "abstain",
        answerable=answerable,
        source="hand",
        group_id="g",
    )


def record(it: EvalItem, abstained: bool, cited: list[str], verdict: str | None) -> dict:
    rec = {
        "source": "hand",
        "category": it.category,
        "answerable": it.answerable,
        "abstained": abstained,
        "abstain_reason": "model" if abstained else None,
        "cited_paths": cited,
        "judge": {"verdict": verdict} if verdict else None,
    }
    rec["score"] = score_item(it, rec)
    return rec


def test_metrics_counts() -> None:
    good, bad, refused = item(True, ["A", "B"]), item(True, ["C"]), item(False, [])
    records = [
        record(good, False, ["A", "X"], "correct"),  # 1 of 2 cited correct; 1 of 2 gold found
        record(bad, True, [], None),  # false abstention: no citations, gold missed
        record(refused, True, [], None),  # true abstention
        record(item(False, []), False, ["Y"], None),  # should have abstained
    ]
    (overall,) = [
        r for r in summarize(records, {}) if r["source"] == "all" and r["category"] == "overall"
    ]
    assert (overall["cited_paths"], overall["cited_correct"]) == (3, 1)
    assert (overall["gold_paths"], overall["gold_cited"]) == (3, 1)
    assert (overall["abstain_tp"], overall["abstain_fp"], overall["abstain_fn"]) == (1, 1, 1)
    assert overall["abstain_f1"] == 0.5
    assert (overall["correct"], overall["n"]) == (2, 4)  # judged-correct + true abstention


def test_threshold_choice_maximises_f1_then_fewest_false_abstentions() -> None:
    confidences = [(0.05, False), (0.10, True), (0.30, False), (0.80, True), (0.90, True)]
    threshold, candidates = choose(confidences)
    best = next(c for c in candidates if c["threshold"] == threshold)
    assert threshold == 0.55 and (best["tp"], best["fp"], best["fn"]) == (2, 1, 0)


def test_abstention_f1_is_na_without_unanswerable_items() -> None:
    answerable = item(True, ["A"])
    records = [record(answerable, True, [], None), record(answerable, False, ["A"], "correct")]
    (overall,) = [r for r in summarize(records, {}) if r["category"] == "overall"][:1]
    assert overall["abstain_fp"] == 1
    assert overall["abstain_f1"] == "n/a" and overall["abstain_recall"] == "n/a"


def test_bootstrap_intervals_are_seeded_and_bracket_the_estimate() -> None:
    good, refused = item(True, ["A"]), item(False, [])
    records = [record(good, False, ["A"], "correct")] * 6 + [record(good, True, [], None)] * 4
    records += [record(refused, True, [], None)] * 2
    first = summarize(records, {})[0]
    again = summarize(records, {})[0]
    assert first == again
    assert first["correct_rate_ci_low"] <= first["correct_rate"] <= first["correct_rate_ci_high"]
    low, high = first["citation_recall_ci_low"], first["citation_recall_ci_high"]
    assert low <= first["citation_recall"] == 0.6 <= high


# -- date gate --------------------------------------------------------------------------


def future_rate_item() -> EvalItem:
    """The hand-written "Portland cement in 2028" abstain item (eval/questions.jsonl)."""
    (found,) = [
        entry
        for entry in load_eval_items(DEFAULT_QUESTION_FILE)
        if entry.group_id == "abs-future-rate"
    ]
    return found


def rate_context() -> Context:
    # The item's v2 context: rows of Schedule II of 9/2025, best score above the threshold.
    schedule = next(n for n in RATE_NOTIFICATIONS if n.number == "9/2025")
    record_id = "CTR/9-2025/schII@v1"
    chunks = [
        Chunk(
            f"{record_id}#{s_no}",
            record_id,
            "CTR/9-2025/schII",
            "notification",
            schedule.effective_from.isoformat(),
            OPEN_END,
            "text",
            score,
            False,
        )
        for s_no, score in ((9, 0.8965), (226, 0.5))
    ]
    return Context(chunks, confidence=0.8965, candidates=len(chunks))


def model_abstains(tmp_path, calls: list) -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        reply = {"abstain": True, "ref_date_used": "", "claims": [], "assumptions": []}
        return httpx.Response(200, json=gemini_reply(json.dumps(reply)))

    return client(handler, tmp_path)


def test_date_gate_abstains_on_future_rate_question_without_calling_the_llm(
    tmp_path, monkeypatch
) -> None:
    question = future_rate_item()
    assert "Portland cement" in question.question and question.ref_date == date(2028, 1, 1)
    context = rate_context()
    assert context.confidence >= TUNED_ABSTAIN_THRESHOLD  # the confidence gate lets it through
    monkeypatch.setattr("generation.answer.build_context", lambda *args, **kwargs: context)
    calls: list[str] = []
    answerer = Answerer(None, model_abstains(tmp_path, calls))

    result = answerer.answer(question.question, question.ref_date)

    horizon = knowledge_horizons()["notification"]
    assert question.ref_date > horizon
    assert result.abstained and result.abstain_reason == "horizon" and not calls
    assert result.rendered.startswith("**Not enough information**")
    assert (
        f"The reference date 2028-01-01 is after {horizon.isoformat()}, the last date the "
        "corpus covers for rate notifications"
    ) in result.rendered
    # On the horizon date itself the question goes to the generator.
    on_horizon = answerer.answer(question.question, horizon)
    assert on_horizon.abstain_reason == "model" and len(calls) == 1


def test_date_gate_is_a_flag_and_uses_the_leading_chunk_source_type(tmp_path, monkeypatch) -> None:
    question = future_rate_item()
    context = rate_context()
    monkeypatch.setattr("generation.answer.build_context", lambda *args, **kwargs: context)
    calls: list[str] = []
    off = Answerer(None, model_abstains(tmp_path, calls), GenerationConfig(date_gate=False))
    assert off.answer(question.question, question.ref_date).abstain_reason == "model"
    assert len(calls) == 1
    # Each source type has its own horizon: a date between two horizons is gated for one only.
    horizons = {"notification": date(2026, 1, 1), "rule": date(2027, 1, 1)}
    gated = Answerer(None, model_abstains(tmp_path, calls), horizons=horizons)
    assert gated.answer("q", date(2026, 6, 1)).abstain_reason == "horizon"
    rule_first = Context([CONTEXT[0], *context.chunks], context.confidence, 3)
    monkeypatch.setattr("generation.answer.build_context", lambda *args, **kwargs: rule_first)
    assert gated.answer("q", date(2026, 6, 1)).abstain_reason == "model"


def test_horizons_come_from_the_sources() -> None:
    # ctr-01-2026_2026-04-30.pdf: "New Delhi, the 30th April, 2026" and "This notification
    # shall come into force from 1st May, 2026."
    issued, commencement = rate_notification_dates(RAW / "ctr-01-2026_2026-04-30.pdf")
    assert (issued, commencement) == (date(2026, 4, 30), date(2026, 5, 1))
    horizons = knowledge_horizons()
    assert horizons["notification"] == rate_horizon_details()[0]
    assert horizons["act"] == act_source_horizon()
    with AMENDMENTS_FILE.open(encoding="utf-8") as handle:
        issue_dates = [json.loads(line)["issue_date"] for line in handle if line.strip()]
    assert horizons["rule"].isoformat() == max(issue_dates)


def test_rates_horizon_ends_before_the_first_unapplied_amendment(caplog) -> None:
    # ctr-19-2025_2025-12-31.pdf amends No. 9/2025- Central Tax (Rate) and "shall come into
    # force on the 1st day of February, 2026"; ctr-01-2026_2026-04-30.pdf amends it "from 1st
    # May, 2026". Neither is applied to the index rows (RATE_NOTIFICATIONS).
    amendments, unreadable = rate_amendments()
    by_number = {a.number: a for a in amendments}
    # ctr-18-2025 has no text layer; its OCR (data/raw/manifest.json) reads "makes the following
    # further amendments in the notification ..., No. 26/2018-Central Tax (Rate)", which is not
    # an indexed schedule, so it does not move the horizon.
    assert not unreadable
    assert by_number["18/2025"].amends == (26, 2018)
    assert (by_number["19/2025"].amends, by_number["19/2025"].effective) == (
        (9, 2025),
        date(2026, 2, 1),
    )
    assert (by_number["01/2026"].amends, by_number["01/2026"].effective) == (
        (9, 2025),
        date(2026, 5, 1),
    )
    with caplog.at_level("WARNING", logger="generation.horizon"):
        horizon = rates_horizon()
    assert horizon == date(2026, 1, 31)
    assert "amendments 19/2025, 01/2026 not applied to index" in caplog.text
    # Amendments to a superseded schedule (01/2025 amends 1/2017) do not move the horizon.
    assert by_number["01/2025"].amends == (1, 2017)
    # Once an amendment is applied to the rows, the horizon moves to the next one.
    applied = [
        replace(n, applied_amendments=("19/2025",)) if n.number == "9/2025" else n
        for n in RATE_NOTIFICATIONS
    ]
    assert rate_horizon_details(notifications=applied) == (
        date(2026, 4, 30),
        "amendments 01/2026 not applied to index",
    )


def test_judge_v2_sees_the_gold_provision_text_in_force_on_the_reference_date() -> None:
    from eval.generation import gold_provision_text, judge
    from generation.prompts import JUDGE_SYSTEM_V2, PROVISION_SAME_AS_REFERENCE

    # Synthetic rows (mechanics only, not legal text).
    def row(row_id: str, path: str, start: str, end: str, text: str) -> dict:
        return {
            "id": row_id,
            "path": path,
            "effective_from": start,
            "effective_to": end,
            "text": text,
        }

    rows = {
        "CGST/r1": [
            row("CGST/r1@v1", "CGST/r1", "2017-07-01", "2020-12-31", "old version"),
            row("CGST/r1@v2", "CGST/r1", "2021-01-01", OPEN_END, "new version"),
        ],
        "CTR/0-0000/schI": [
            row(
                "CTR/0-0000/schI@v1#1",
                "CTR/0-0000/schI",
                "2017-07-01",
                OPEN_END,
                "S. No. 1. alpha goods",
            ),
            row(
                "CTR/0-0000/schI@v1#2",
                "CTR/0-0000/schI",
                "2017-07-01",
                OPEN_END,
                "S. No. 2. beta goods",
            ),
        ],
    }

    def item(paths: list[str], gold: str, ref: date) -> EvalItem:
        return EvalItem(
            group_id="g",
            question="q",
            ref_date=ref,
            category="direct_lookup",
            gold_paths=paths,
            gold_answer=gold,
            answerable=True,
            source="hand",
        )

    assert gold_provision_text(item(["CGST/r1"], "x", date(2020, 6, 1)), rows) == "old version"
    assert gold_provision_text(item(["CGST/r1"], "x", date(2021, 6, 1)), rows) == "new version"
    # A rate schedule contributes only the row the gold answer quotes.
    rate = item(["CTR/0-0000/schI"], "6 per cent: beta goods", date(2021, 6, 1))
    assert gold_provision_text(rate, rows) == "S. No. 2. beta goods"

    seen = {}

    class Client:
        def complete(self, system, prompt):
            seen["system"], seen["prompt"] = system, prompt
            raise LLMError("no call")

    judge(Client(), item(["CGST/r1"], "a reference", date(2021, 6, 1)), "answer", "v2", rows)
    assert seen["system"] == JUDGE_SYSTEM_V2
    assert "Extra statements supported by the provision text are not errors" in seen["system"]
    assert "Gold provision text (in force on the reference date):\nnew version" in seen["prompt"]
    # When the gold answer is the provision text, it is not repeated.
    judge(Client(), item(["CGST/r1"], "new version", date(2021, 6, 1)), "answer", "v2", rows)
    assert PROVISION_SAME_AS_REFERENCE in seen["prompt"]
    # v1 is unchanged: no provision text.
    judge(Client(), item(["CGST/r1"], "a reference", date(2021, 6, 1)), "answer", "v1", rows)
    assert "Gold provision text" not in seen["prompt"]


def test_judge_versions_keep_separate_items_files_and_kappa() -> None:
    from eval.generation import items_path
    from eval.judge_audit import cohen_kappa

    assert items_path("dev", "v3").name == "generation_dev_v3_items.jsonl"
    assert items_path("dev", "v3", "v2").name == "generation_dev_v3_judge-v2_items.jsonl"
    assert items_path("test", "", "v2").name == "generation_test_judge-v2_items.jsonl"
    assert cohen_kappa(["correct", "incorrect"], ["correct", "incorrect"]) == 1.0
    # 3/4 agreement; marginals a = (2 correct, 2 partial), b = (1 correct, 3 partial).
    a = ["correct", "correct", "partially_correct", "partially_correct"]
    b = ["correct", "partially_correct", "partially_correct", "partially_correct"]
    assert abs(cohen_kappa(a, b) - 0.5) < 1e-9


def test_audit_sheet_with_human_verdicts_is_never_overwritten(tmp_path) -> None:
    from eval.judge_audit import FIELDS, write

    sheet = tmp_path / "audit.csv"
    row = {field: "x" for field in FIELDS}
    write([{**row, "human_verdict": ""}], sheet)
    write([{**row, "human_verdict": "correct"}], sheet)  # still unlabelled: allowed
    with pytest.raises(SystemExit):
        write([{**row, "human_verdict": ""}], sheet)


def test_prompt_variants() -> None:
    import re

    from generation.prompts import _FEW_SHOT, GENERATOR_SYSTEM, PROMPT_VARIANTS, generator_system
    from generation.schemas import RESPONSE_SCHEMA, response_schema

    # The baseline is the prompt every earlier run used, so its cached answers stay valid.
    assert generator_system("baseline") == GENERATOR_SYSTEM
    assert response_schema("baseline") is RESPONSE_SCHEMA
    systems = {variant: generator_system(variant) for variant in PROMPT_VARIANTS}
    assert len(set(systems.values())) == len(PROMPT_VARIANTS)
    assert all(text.startswith(GENERATOR_SYSTEM) for text in systems.values())
    assert response_schema("provision_first")["propertyOrdering"][0] == "governing"
    output = GenerationOutput.model_validate_json(
        '{"governing": {"citation_id": "X@v1", "in_force": "2021", "covers_ref_date": true},'
        ' "abstain": false, "ref_date_used": "2021-01-01", "claims": [], "assumptions": []}'
    )
    assert output.claims == [] and not output.abstain
    with pytest.raises(ValueError):
        generator_system("unknown")

    # Few-shot examples overlap no eval item: no gold path, and no goods or provision words.
    items = load_eval_items(DEFAULT_QUESTION_FILE)
    gold = {path for item in items for path in item.gold_paths}
    for question, _, chunks, output in _FEW_SHOT:
        cited = {cid.split("@")[0] for claim in output["claims"] for cid in claim["citation_ids"]}
        assert cited <= {chunk.path for chunk in chunks}
        for chunk in chunks:
            if chunk.path.startswith("CTR/"):
                goods = chunk.text.split(":", 1)[1].split(". ", 1)[1]
                words = set(re.findall(r"[a-z]{4,}", goods.lower()))
                text = " ".join(f"{i.question} {i.gold_answer}" for i in items).lower()
                assert not any(word in text for word in words), goods
            else:
                assert not any(g == chunk.path or g.startswith(chunk.path + "/") for g in gold), (
                    chunk.path
                )


def test_prompt_variant_choice_follows_the_adoption_rule() -> None:
    from eval.prompt_variants import choose

    def row(variant, rate, low, high, recall=1.0, tokens=1000.0):
        return {
            "variant": variant,
            "correct_rate": rate,
            "correct_rate_ci_low": low,
            "correct_rate_ci_high": high,
            "citation_recall": recall,
            "mean_tokens": tokens,
        }

    base = row("baseline", 0.70, 0.60, 0.80)
    # Gain inside the baseline interval and dearer: keep the baseline.
    assert choose([base, row("few_shot", 0.75, 0.65, 0.85, tokens=2000.0)])[:2] == (
        "few_shot",
        "baseline",
    )
    # Gain inside the interval but cheaper: adopt.
    assert choose([base, row("strict", 0.75, 0.65, 0.85, tokens=900.0)])[1] == "strict"
    # Gain above the interval: adopt.
    assert choose([base, row("first", 0.85, 0.75, 0.95, tokens=2000.0)])[1] == "first"
    # Tie on correct rate: citation recall, then tokens.
    tie = [base, row("a", 0.70, 0.6, 0.8, recall=0.9), row("b", 0.70, 0.6, 0.8, tokens=500.0)]
    assert choose(tie)[:2] == ("b", "b")


def test_prompt_variant_prescreen_drops_and_picks_one_challenger() -> None:
    from eval.prompt_variants import prescreen

    def row(variant, recall, fp=1, tokens=1000.0, precision=1.0, failures=0):
        return {
            "variant": variant,
            "citation_recall": recall,
            "citation_precision": precision,
            "abstain_fp": fp,
            "format_failures": failures,
            "mean_tokens": tokens,
        }

    rows, challenger = prescreen(
        [
            row("baseline", 0.97),
            row("lower_recall", 0.95),
            row("more_fp", 0.98, fp=2),
            row("dear_no_gain", 0.97, tokens=1400.0),
            row("dear_with_gain", 0.99, tokens=1400.0),
            row("equal", 0.97, tokens=900.0),
        ]
    )
    verdict = {r["variant"]: r["prescreen"] for r in rows}
    assert verdict["lower_recall"].startswith("dropped: citation recall")
    assert verdict["more_fp"].startswith("dropped: more abstention FP")
    assert verdict["dear_no_gain"].startswith("dropped: >30% more tokens")
    assert verdict["equal"] == "survives"
    assert challenger == "dear_with_gain"

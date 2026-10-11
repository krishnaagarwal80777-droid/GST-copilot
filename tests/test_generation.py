from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from eval.generation import score_item, summarize
from eval.tune_abstention import choose
from generation.context import Chunk
from generation.llm import LLMClient, LLMError
from generation.prompts import chunk_label, generator_prompt
from generation.render import DISCLAIMER, render_abstention, render_answer
from generation.schemas import Claim, GenerationOutput
from generation.verify import verify
from gst_copilot.models import EvalItem


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

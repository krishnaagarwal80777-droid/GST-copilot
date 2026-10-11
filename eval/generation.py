"""Generation evaluation: citation precision/recall, abstention F1, answer correctness (judge).

Runs the full answer pipeline (generation.answer) on every scored item of a split. Metrics are
reported as counts and rates, per source (all / hand / templated) and category:

- Citations (paths of the chunks cited by verified claims, compared with gold_paths):
  precision = correct cited paths / cited paths, over every answered item (micro);
  recall = gold paths cited / gold paths, over every answerable item (an abstention cites none).
- Abstention (positive class = abstain; gold = not answerable): TP, FP, FN, TN, P, R, F1.
- Correctness: answerable items that were answered are graded by the Groq judge against the
  gold answer (correct / partially_correct / incorrect). "correct" overall counts judge-correct
  answerable items plus unanswerable items that abstained; answerable items that abstained and
  unanswerable items that were answered count as wrong.

Generator and judge calls are cached on disk (generation.llm), so re-runs are free. Per-item
results go to eval/results/generation_<split>_items.jsonl and summary rows are appended to
eval/results/generation.csv. The test split is run once: a second test run needs --rerun-test.

``--dry-run`` makes no LLM calls: it builds every context and prompt and counts input tokens.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from eval.retrieval import git_hash
from eval.split import item_key, items_in, source_kind
from generation.config import GenerationConfig
from generation.prompts import JUDGE_SYSTEM, judge_prompt
from gst_copilot.models import EvalItem

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"
SUMMARY_FILE = RESULTS_DIR / "generation.csv"
JUDGE_REFERENCE_CAP = 6000  # characters of gold text shown to the judge
VERDICTS = ("correct", "partially_correct", "incorrect")
FIELDS = (
    "timestamp",
    "git_hash",
    "split",
    "source",
    "category",
    "config",
    "generator_model",
    "judge_model",
    "abstain_threshold",
    "n",
    "answerable",
    "unanswerable",
    "answered",
    "abstained",
    "abstained_gate",
    "cited_paths",
    "cited_correct",
    "citation_precision",
    "gold_paths",
    "gold_cited",
    "citation_recall",
    "abstain_tp",
    "abstain_fp",
    "abstain_fn",
    "abstain_tn",
    "abstain_precision",
    "abstain_recall",
    "abstain_f1",
    "judge_correct",
    "judge_partially_correct",
    "judge_incorrect",
    "judge_error",
    "correct",
    "correct_rate",
)


def judge_reference(item: EvalItem) -> tuple[str, dict | None]:
    """Gold text for the judge and, for templated version pairs, what changed at the boundary."""
    reference = item.gold_answer
    if len(reference) > JUDGE_REFERENCE_CAP:
        reference = reference[:JUDGE_REFERENCE_CAP] + " [... reference text truncated]"
    template = item.template or {}
    if template.get("kind") == "version_change":
        return reference, {
            "only_in_this": template.get("only_in_this", ""),
            "only_in_other": template.get("only_in_other", ""),
        }
    return reference, None


def judge(client, item: EvalItem, rendered: str) -> dict[str, Any]:
    from generation.llm import LLMError

    reference, distinguishing = judge_reference(item)
    prompt = judge_prompt(item.question, item.ref_date, reference, rendered, distinguishing)
    try:
        response = client.complete(JUDGE_SYSTEM, prompt)
        verdict = response.json()
    except (LLMError, ValueError) as error:
        return {"verdict": "judge_error", "rationale": str(error)[:500]}
    label = str(verdict.get("verdict", "")).strip().lower()
    return {
        "verdict": label if label in VERDICTS else "judge_error",
        "rationale": verdict.get("rationale", ""),
        "model": response.model,
        "cached": response.cached,
    }


def score_item(item: EvalItem, result: dict[str, Any]) -> dict[str, Any]:
    cited = result["cited_paths"] if not result["abstained"] else []
    gold = set(item.gold_paths)
    verdict = (result.get("judge") or {}).get("verdict")
    if item.answerable:
        correct = (not result["abstained"]) and verdict == "correct"
    else:
        correct = result["abstained"]
    return {
        "cited": len(cited),
        "cited_correct": sum(1 for path in cited if path in gold),
        "gold": len(gold),
        "gold_cited": len(gold & set(cited)),
        "correct": correct,
    }


def _rate(numerator: int, denominator: int) -> float | str:
    return round(numerator / denominator, 4) if denominator else ""


def summarize(records: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        for source in ("all", record["source"]):
            groups[(source, "overall")].append(record)
            groups[(source, record["category"])].append(record)
    sources = ("all", "hand", "templated")
    rows = []
    for key in sorted(groups, key=lambda k: (sources.index(k[0]), k[1] != "overall", k[1])):
        members = groups[key]
        answered = [r for r in members if not r["abstained"]]
        tp = sum(1 for r in members if r["abstained"] and not r["answerable"])
        fp = sum(1 for r in members if r["abstained"] and r["answerable"])
        fn = sum(1 for r in members if not r["abstained"] and not r["answerable"])
        tn = sum(1 for r in members if not r["abstained"] and r["answerable"])
        precision, recall = _rate(tp, tp + fp), _rate(tp, tp + fn)
        f1 = _rate(2 * tp, 2 * tp + fp + fn)  # F1 = 2TP / (2TP + FP + FN)
        verdicts = [(r.get("judge") or {}).get("verdict") for r in answered if r["answerable"]]
        cited = sum(r["score"]["cited"] for r in answered)
        cited_correct = sum(r["score"]["cited_correct"] for r in answered)
        answerable = [r for r in members if r["answerable"]]
        gold = sum(r["score"]["gold"] for r in answerable)
        gold_cited = sum(r["score"]["gold_cited"] for r in answerable)
        correct = sum(1 for r in members if r["score"]["correct"])
        rows.append(
            {
                **meta,
                "source": key[0],
                "category": key[1],
                "n": len(members),
                "answerable": len(answerable),
                "unanswerable": len(members) - len(answerable),
                "answered": len(answered),
                "abstained": len(members) - len(answered),
                "abstained_gate": sum(1 for r in members if r["abstain_reason"] == "gate"),
                "cited_paths": cited,
                "cited_correct": cited_correct,
                "citation_precision": _rate(cited_correct, cited),
                "gold_paths": gold,
                "gold_cited": gold_cited,
                "citation_recall": _rate(gold_cited, gold),
                "abstain_tp": tp,
                "abstain_fp": fp,
                "abstain_fn": fn,
                "abstain_tn": tn,
                "abstain_precision": precision,
                "abstain_recall": recall,
                "abstain_f1": f1,
                "judge_correct": verdicts.count("correct"),
                "judge_partially_correct": verdicts.count("partially_correct"),
                "judge_incorrect": verdicts.count("incorrect"),
                "judge_error": verdicts.count("judge_error"),
                "correct": correct,
                "correct_rate": _rate(correct, len(members)),
            }
        )
    return rows


def write_summary(rows: list[dict[str, Any]], path: Path = SUMMARY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    head = (
        f"{'source':9} {'category':15} {'n':>3} {'ans':>4} {'abs':>4} "
        f"{'cit P (k/n)':>15} {'cit R (k/n)':>15} {'abst TP/FP/FN':>13} {'F1':>6} "
        f"{'judge C/P/I/E':>13} {'correct':>11}"
    )
    print(head)
    for r in rows:
        cp = f"{r['citation_precision']} ({r['cited_correct']}/{r['cited_paths']})"
        cr = f"{r['citation_recall']} ({r['gold_cited']}/{r['gold_paths']})"
        ab = f"{r['abstain_tp']}/{r['abstain_fp']}/{r['abstain_fn']}"
        jd = (
            f"{r['judge_correct']}/{r['judge_partially_correct']}/"
            f"{r['judge_incorrect']}/{r['judge_error']}"
        )
        print(
            f"{r['source']:9} {r['category']:15} {r['n']:>3} {r['answered']:>4} "
            f"{r['abstained']:>4} {cp:>15} {cr:>15} {ab:>13} {str(r['abstain_f1']):>6} "
            f"{jd:>13} {r['correct']:>4}/{r['n']:<3} ({r['correct_rate']})"
        )


def run(
    split: str,
    *,
    config: GenerationConfig,
    dry_run: bool = False,
    limit: int | None = None,
    source: str | None = None,
    rerun_test: bool = False,
) -> None:
    from generation.answer import Answerer
    from generation.llm import QuotaExhausted, generator_client, judge_client
    from retrieval.embed import Embedder
    from retrieval.pipeline import Retriever
    from retrieval.rerank import Reranker
    from retrieval.store import open_store

    items_file = RESULTS_DIR / f"generation_{split}_items.jsonl"
    if split == "test" and not dry_run and items_file.exists() and not rerun_test:
        raise SystemExit(
            f"{items_file.relative_to(ROOT)} exists: the test split has been run. "
            "Pass --rerun-test only if the run did not complete."
        )
    items = [i for i in items_in(split) if source is None or source_kind(i) == source]
    items = items[:limit] if limit else items
    retriever = Retriever(open_store(), Embedder(), Reranker())
    generator = generator_client()
    answerer = Answerer(retriever, generator, config)
    known_by = date.today()

    if dry_run:
        total, gated, methods, cached = 0, 0, set(), 0
        for item in items:
            result = answerer.answer(item.question, item.ref_date, known_by=known_by, dry_run=True)
            if result.abstained:
                gated += 1
                continue
            total += result.prompt_tokens["input_tokens"]
            methods.add(result.prompt_tokens["method"])
            cached += result.prompt_tokens["cached"]
        prompts = len(items) - gated
        print(
            f"dry run, {split}: {len(items)} items, {gated} stopped by the gate, "
            f"{prompts} generator prompts ({cached} already cached), "
            f"{total} input tokens ({'/'.join(sorted(methods)) or 'n/a'}), "
            f"mean {total // max(prompts, 1)} per prompt; model {generator.model}. "
            f"The judge adds one call per answered answerable item."
        )
        return

    judge_llm = judge_client()
    records = []
    for index, item in enumerate(items, 1):
        try:
            result = answerer.answer(item.question, item.ref_date, known_by=known_by)
        except QuotaExhausted as error:
            raise SystemExit(
                f"stopped at item {index}/{len(items)}: {error}\n"
                f"{index - 1} items are cached; re-running resumes from item {index}."
            ) from None
        record = {
            "key": item_key(item),
            "source": source_kind(item),
            "category": item.category,
            "answerable": item.answerable,
            "gold_paths": item.gold_paths,
            **result.to_dict(),
            "cited_paths": result.cited_paths,
            "judge": None,
        }
        if item.answerable and not result.abstained:
            record["judge"] = judge(judge_llm, item, result.rendered)
        record["score"] = score_item(item, record)
        records.append(record)
        verdict = (record["judge"] or {}).get("verdict", "-")
        status = f"abstain:{result.abstain_reason}" if result.abstained else "answered"
        print(f"[{index}/{len(items)}] {item_key(item):45} {status:24} judge={verdict}")

    models = {r["llm"]["model"] for r in records if r.get("llm")}
    judges = {
        r["judge"].get("model") for r in records if r.get("judge") and r["judge"].get("model")
    }
    meta = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "split": split,
        "config": config.name,
        "generator_model": "/".join(sorted(models)),
        "judge_model": "/".join(sorted(judges)),
        "abstain_threshold": config.abstain_threshold if config.gate else "",
    }
    rows = summarize(records, meta)
    if limit is None and source is None:
        with items_file.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"meta": meta, "config": asdict(config)}) + "\n")
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_summary(rows)
        print(
            f"wrote {items_file.relative_to(ROOT)}; appended {len(rows)} rows to "
            f"{SUMMARY_FILE.relative_to(ROOT)}"
        )
    else:
        print("partial run (--limit/--source): results not written")
    print_summary(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generation evaluation (LLM, cached).")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--dry-run", action="store_true", help="count tokens, no LLM calls")
    parser.add_argument("--limit", type=int, help="first N items only (results not written)")
    parser.add_argument("--source", choices=("hand", "templated"))
    parser.add_argument("--no-gate", action="store_true", help="ablation: no confidence gate")
    parser.add_argument(
        "--no-verify", action="store_true", help="ablation: keep claims without valid citations"
    )
    parser.add_argument("--rerun-test", action="store_true")
    args = parser.parse_args()
    config = GenerationConfig()
    if args.no_gate:
        config = replace(config, name=config.name + "-gate", gate=False)
    if args.no_verify:
        config = replace(config, name=config.name + "-verify", verify_citations=False)
    run(
        args.split,
        config=config,
        dry_run=args.dry_run,
        limit=args.limit,
        source=args.source,
        rerun_test=args.rerun_test,
    )


if __name__ == "__main__":
    main()

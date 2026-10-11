"""Tune the abstention gate threshold on the dev split only (no LLM calls).

For every scored dev item (answerable and unanswerable), build the generator context and take
its confidence: the best cross-encoder score of a servable chunk valid on the reference date.
The gate abstains when confidence < threshold. Candidate thresholds are 0 and the midpoints
between consecutive distinct confidences. Selection: highest abstention F1 (positive class =
should abstain); ties go to fewer false abstentions on answerable items, then to the lower
threshold. The gate is a coarse first filter, so the generator still abstains on
unanswerable questions whose wording retrieves close provisions (e.g. a future-dated rate).

Writes per-item confidences and every candidate to eval/results/abstention_tuning_dev.csv;
copy the winner into generation.config.TUNED_ABSTAIN_THRESHOLD.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from eval.retrieval import git_hash
from eval.split import item_key, items_in, source_kind
from generation.config import GenerationConfig
from generation.context import build_context

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "abstention_tuning_dev.csv"


def f1_counts(confidences: list[tuple[float, bool]], threshold: float) -> dict:
    """confidences: (confidence, answerable). Positive class = abstain."""
    tp = sum(1 for c, ans in confidences if c < threshold and not ans)
    fp = sum(1 for c, ans in confidences if c < threshold and ans)
    fn = sum(1 for c, ans in confidences if c >= threshold and not ans)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4)}


def choose(confidences: list[tuple[float, bool]]) -> tuple[float, list[dict]]:
    values = sorted({round(c, 6) for c, _ in confidences})
    thresholds = [0.0] + [round((a + b) / 2, 6) for a, b in zip(values, values[1:], strict=False)]
    candidates = [{"threshold": t, **f1_counts(confidences, t)} for t in thresholds]
    best = min(candidates, key=lambda c: (-c["f1"], c["fp"], c["threshold"]))
    return best["threshold"], candidates


def main() -> None:
    from retrieval.embed import Embedder
    from retrieval.pipeline import Retriever
    from retrieval.rerank import Reranker
    from retrieval.store import open_store

    retriever = Retriever(open_store(), Embedder(), Reranker())
    config = GenerationConfig(gate=False)
    items = items_in("dev")
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_hash()
    per_item = []
    for item in items:
        context = build_context(retriever, item.question, item.ref_date, config)
        per_item.append((item, context.confidence))
    threshold, candidates = choose([(conf, item.answerable) for item, conf in per_item])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# timestamp", stamp, "git_hash", commit, "chosen", threshold])
        writer.writerow(["kind", "key", "source", "category", "answerable", "confidence"])
        for item, conf in sorted(per_item, key=lambda pair: pair[1]):
            writer.writerow(
                [
                    "item",
                    item_key(item),
                    source_kind(item),
                    item.category,
                    item.answerable,
                    round(conf, 6),
                ]
            )
        writer.writerow(["kind", "threshold", "tp", "fp", "fn", "f1"])
        for c in candidates:
            writer.writerow(["candidate", c["threshold"], c["tp"], c["fp"], c["fn"], c["f1"]])
    n_abstain = sum(1 for item in items if not item.answerable)
    print(f"dev items: {len(items)} ({n_abstain} unanswerable)")
    for item, conf in sorted(per_item, key=lambda pair: pair[1])[:12]:
        flag = "ANSWERABLE" if item.answerable else "abstain"
        print(f"  {conf:.4f}  {flag:10} {item_key(item)}")
    best = next(c for c in candidates if c["threshold"] == threshold)
    print(f"chosen threshold {threshold}: {best}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

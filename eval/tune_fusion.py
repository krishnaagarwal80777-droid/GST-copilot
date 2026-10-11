"""Tune dense+BM25 fusion on the dev split only (eval/split.json).

Grid: weighted RRF over k x BM25 weight (dense weight 1), and dense-first with a BM25
tie-break over tie_eps. Each candidate is scored as the full pipeline (exact match + date
filter) and as plain dense+bm25.

Selection: candidates whose full-pipeline dev MRR@10 is within one item of the best (1/n,
since one item moving rank changes MRR by at most 1/n) are treated as tied; among them the
highest dense+bm25 dev MRR@10 wins, then the simpler candidate (earlier in the grid). The full
pipeline is dominated by exact-match pins and the date filter, so fusion differences there are
within noise; dense+bm25 is where fusion is measurable. (An earlier rule that ranked on
full-pipeline MRR alone picked a candidate one item ahead there but worst on dense+bm25.)
Writes every candidate to eval/results/fusion_tuning_dev.csv; copy the winner into
retrieval.config.TUNED_FUSION.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from eval.retrieval import Evaluation, git_hash, split_items
from retrieval.config import RetrievalConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "fusion_tuning_dev.csv"
FULL = RetrievalConfig("+date_filter", bm25=True, exact_match=True, date_filter=True)
PLAIN = RetrievalConfig("dense+bm25", bm25=True)


def grid() -> list[dict]:
    candidates = [{"fusion": "rrf", "rrf_k": 60, "w_dense": 1.0, "w_bm25": 1.0, "tie_eps": 0.0}]
    for k in (60, 30, 10):
        for w_bm25 in (1.0, 0.5, 0.25):
            candidate = {
                "fusion": "rrf",
                "rrf_k": k,
                "w_dense": 1.0,
                "w_bm25": w_bm25,
                "tie_eps": 0.0,
            }
            if candidate not in candidates:
                candidates.append(candidate)
    for eps in (0.0, 0.005, 0.01, 0.02, 0.04):
        candidates.append(
            {"fusion": "dense_first", "rrf_k": 60, "w_dense": 1.0, "w_bm25": 1.0, "tie_eps": eps}
        )
    return candidates


def overall(per_category: dict) -> tuple[float, float, float]:
    scores = sum(per_category.values(), [])
    return tuple(round(mean(score[i] for score in scores), 4) for i in range(3))


def main() -> None:
    session = Evaluation(rebuild=False)
    items = split_items("dev")
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_hash()
    rows = []
    for order, candidate in enumerate(grid()):
        full = overall(session.item_scores(replace(FULL, **candidate), items))
        plain = overall(session.item_scores(replace(PLAIN, **candidate), items))
        rows.append(
            {
                "timestamp": stamp,
                "git_hash": commit,
                "split": "dev",
                "n": len(items),
                "order": order,
                **candidate,
                "full_r@5": full[0],
                "full_r@10": full[1],
                "full_mrr@10": full[2],
                "dense+bm25_r@5": plain[0],
                "dense+bm25_r@10": plain[1],
                "dense+bm25_mrr@10": plain[2],
            }
        )
        print(
            f"{order:2} {candidate['fusion']:11} k={candidate['rrf_k']:2} "
            f"w_bm25={candidate['w_bm25']:<4} eps={candidate['tie_eps']:<5} "
            f"full MRR={full[2]:.3f} R@5={full[0]:.3f} | "
            f"dense+bm25 MRR={plain[2]:.3f} R@5={plain[0]:.3f}"
        )
    top_full = max(row["full_mrr@10"] for row in rows)
    tied = [row for row in rows if row["full_mrr@10"] >= top_full - 1 / len(items) - 1e-9]
    best = max(tied, key=lambda row: (row["dense+bm25_mrr@10"], -row["order"]))
    for row in rows:
        row["selected"] = row is best
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ("fusion", "rrf_k", "w_dense", "w_bm25", "tie_eps")
    print("selected:", {key: best[key] for key in keys})


if __name__ == "__main__":
    main()

"""Retrieval evaluation (no LLM): recall@5, recall@10 and MRR@10 per config and category.

Relevance
- Structured configs: a result is relevant only if it is the gold path's version valid on
  the item's ref_date (record id), so retrieving the right section in the wrong version
  does not count. This is what the date filter is measured against.
- Naive config (raw 500-token chunks, no paths or versions): a chunk is relevant to a gold
  provision if it contains an 8-word shingle that the gold version introduced (absent from the
  version before it; for a first version, absent from the version after it). A provision with
  one version matches on any of its shingles. Rate items need a 6-word shingle of the schedule
  description quoted in the gold answer, in that notification's file.

Only scored items with gold paths count (pending_source and abstain items are excluded).
Results are appended to eval/results/retrieval.csv with config, git hash and timestamp.
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from eval.harness import load_eval_items, scored_items
from gst_copilot.models import EvalItem
from retrieval.config import CONFIGS, RetrievalConfig
from retrieval.corpus import RATE_NOTIFICATIONS
from retrieval.embed import Embedder
from retrieval.naive import shingles
from retrieval.pipeline import NAIVE_TABLE, PROVISIONS_TABLE, Retriever, build_indexes
from retrieval.store import open_store

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval" / "results" / "retrieval.csv"
FIELDS = (
    "timestamp",
    "git_hash",
    "config",
    "category",
    "n",
    "recall@5",
    "recall@10",
    "mrr@10",
    "backend",
    "relevance",
)
NOTIFICATION_FILES = {
    f"CTR/{notification.number.replace('/', '-')}": notification.filename
    for notification in RATE_NOTIFICATIONS
}


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _body(text: str) -> str:
    """Index text without its "<source>, <citation> - <heading>: " prefix."""
    return text.split(": ", 1)[1] if ": " in text else text


def gold_versions(item: EvalItem, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Gold path -> record id of its version valid on the item's ref_date."""
    day = item.ref_date.isoformat()
    gold: dict[str, str] = {}
    for path in item.gold_paths:
        valid = {
            row["record_id"]
            for row in rows
            if row["path"] == path and row["effective_from"] <= day <= row["effective_to"]
        }
        if len(valid) != 1:
            raise ValueError(f"{item.question!r}: {path} has {len(valid)} versions on {day}")
        gold[path] = valid.pop()
    return gold


def naive_matchers(item: EvalItem, rows: list[dict[str, Any]]) -> dict[str, tuple[str | None, set]]:
    """Gold path -> (required source file or None, shingles that mark a relevant chunk)."""
    matchers: dict[str, tuple[str | None, set]] = {}
    versions = gold_versions(item, rows)
    for path, record_id in versions.items():
        if path.startswith("CTR/"):
            quoted = re.findall(r'"([^"]+)"', item.gold_answer)
            marks = set().union(*(shingles(text, 6) for text in quoted)) if quoted else set()
            source = NOTIFICATION_FILES["/".join(path.split("/")[:2])]
            matchers[path] = (source, marks)
            continue
        texts: dict[str, str] = {}
        for row in rows:
            if row["path"] == path:
                texts.setdefault(row["record_id"], "")
                texts[row["record_id"]] += " " + _body(row["text"])
        starts = {row["record_id"]: row["effective_from"] for row in rows if row["path"] == path}
        chain = sorted(texts, key=lambda rid: starts[rid])
        position = chain.index(record_id)
        gold = shingles(texts[record_id], 8)
        if len(chain) == 1:
            marks = gold
        elif position == 0:
            # First version: the text that the next version replaced.
            marks = gold - shingles(texts[chain[1]], 8)
        else:
            # Later versions: the text this version introduced.
            marks = gold - shingles(texts[chain[position - 1]], 8)
        if not marks:
            raise ValueError(f"{path}: version {record_id} has no text of its own to match")
        matchers[path] = (None, marks)
    return matchers


def score_item(ranked_hits: list[list[str]], n_gold: int) -> tuple[float, float, float]:
    """ranked_hits[i] = gold paths hit by result i. Returns (recall@5, recall@10, rr@10)."""

    def recall(k: int) -> float:
        found = set().union(*ranked_hits[:k]) if ranked_hits[:k] else set()
        return len(found) / n_gold

    first = next((rank for rank, hits in enumerate(ranked_hits[:10], 1) if hits), None)
    return recall(5), recall(10), (1.0 / first if first else 0.0)


def evaluate(configs: tuple[RetrievalConfig, ...] = CONFIGS, *, rebuild: bool = True) -> list[dict]:
    store = open_store()
    embedder = Embedder()
    if rebuild:
        counts = build_indexes(store, embedder)
        print(f"indexed with {store.backend}: {counts}")
    retriever = Retriever(store, embedder)
    provision_rows = store.rows(PROVISIONS_TABLE)
    naive_text = {row["record_id"]: row for row in store.rows(NAIVE_TABLE)}
    items = [item for item in scored_items(load_eval_items()) if item.gold_paths]
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_hash()

    results: list[dict] = []
    for config in configs:
        per_category: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for item in items:
            ranked = retriever.search(item.question, config, on=item.ref_date, k=10)
            if config.index == "naive":
                matchers = naive_matchers(item, provision_rows)
                hits = []
                for chunk_id in ranked:
                    chunk = naive_text[chunk_id]
                    hit = []
                    for path, (source, marks) in matchers.items():
                        if source is not None and chunk["path"] != source:
                            continue
                        n = len(next(iter(marks))) if marks else 8
                        chunk_marks = shingles(chunk["text"], n)
                        if marks & chunk_marks:
                            hit.append(path)
                    hits.append(hit)
            else:
                gold = gold_versions(item, provision_rows)
                hits = [[path for path, rid in gold.items() if rid == result] for result in ranked]
            per_category[item.category].append(score_item(hits, len(item.gold_paths)))
        for category, scores in [("overall", sum(per_category.values(), []))] + sorted(
            per_category.items()
        ):
            results.append(
                {
                    "timestamp": stamp,
                    "git_hash": commit,
                    "config": config.name,
                    "category": category,
                    "n": len(scores),
                    "recall@5": round(mean(score[0] for score in scores), 4),
                    "recall@10": round(mean(score[1] for score in scores), 4),
                    "mrr@10": round(mean(score[2] for score in scores), 4),
                    "backend": store.backend,
                    "relevance": "text_overlap" if config.index == "naive" else "version",
                }
            )
    return results


def write_results(results: list[dict], path: Path = RESULTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(results)


def main() -> None:
    results = evaluate()
    write_results(results)
    print(f"{'config':14} {'category':15} {'n':>3} {'R@5':>6} {'R@10':>6} {'MRR':>6}")
    for row in results:
        print(
            f"{row['config']:14} {row['category']:15} {row['n']:>3} {row['recall@5']:>6.3f} "
            f"{row['recall@10']:>6.3f} {row['mrr@10']:>6.3f}"
        )
    print(f"appended {len(results)} rows to {RESULTS_FILE.relative_to(ROOT)} ({date.today()})")


if __name__ == "__main__":
    main()

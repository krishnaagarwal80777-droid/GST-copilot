"""Retrieval evaluation (no LLM): recall@5, recall@10 and MRR@10 per config and category.

Relevance
- Structured configs: a result is relevant only if it is the gold path's version valid on
  the item's ref_date (record id), so retrieving the right section in the wrong version
  does not count. This is what the date filter is measured against.
- Naive config (raw 500-token chunks, no paths or versions): a chunk is relevant to a gold
  provision if it contains an 8-word shingle that the gold version introduced (absent from the
  version before it; for a first version, absent from the version after it). A provision with
  one version, or a version with no text of its own (its text is contained in its neighbour's),
  matches on any of its shingles. Rate items need a 6-word shingle of the schedule
  description quoted in the gold answer, in that notification's file.

Only scored items with gold paths count (pending_source and abstain items are excluded).
Rows are reported per source (hand-written, templated, and all) and category. Results are
appended to eval/results/retrieval.csv with config, git hash and timestamp; rows written before
templated items existed carry source=hand.
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from eval.split import items_in, source_kind
from gst_copilot.models import EvalItem
from retrieval.config import CONFIGS, RetrievalConfig, ablation_configs
from retrieval.corpus import RATE_NOTIFICATIONS
from retrieval.embed import Embedder
from retrieval.naive import shingles
from retrieval.pipeline import NAIVE_TABLE, PROVISIONS_TABLE, Retriever, build_indexes
from retrieval.rerank import Reranker
from retrieval.store import open_store

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "eval" / "results" / "retrieval.csv"
FIELDS = (
    "timestamp",
    "git_hash",
    "split",
    "source",
    "config",
    "index_size",
    "index_records",
    "category",
    "n",
    "recall@5",
    "recall@10",
    "mrr@10",
    "fusion",
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
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return result.stdout.strip() + ("-dirty" if dirty else "")
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
            # The version's text is contained in its neighbour's (a pure insertion or
            # omission), so no shingle is its own: match on any of its shingles, as for a
            # provision with one version.
            marks = gold
        if not marks:
            raise ValueError(f"{path}: version {record_id} has no text to match")
        matchers[path] = (None, marks)
    return matchers


def score_item(ranked_hits: list[list[str]], n_gold: int) -> tuple[float, float, float]:
    """ranked_hits[i] = gold paths hit by result i. Returns (recall@5, recall@10, rr@10)."""

    def recall(k: int) -> float:
        found = set().union(*ranked_hits[:k]) if ranked_hits[:k] else set()
        return len(found) / n_gold

    first = next((rank for rank, hits in enumerate(ranked_hits[:10], 1) if hits), None)
    return recall(5), recall(10), (1.0 / first if first else 0.0)


def fusion_label(config: RetrievalConfig) -> str:
    # With split_bm25 the Act/Rules and rate-schedule BM25 rankings are fused as two lists.
    split = config.bm25 and config.split_bm25 and config.index == "structured"
    lists = ",bm25=act_rules+rates" if split else ""
    lists += ",typed_citations" if split and config.typed_citations else ""
    if not (config.dense and config.bm25):
        return f"rrf(k={config.rrf_k},w_bm25={config.w_bm25}{lists})" if split else ""
    if config.fusion == "dense_first":
        return f"dense_first(tie_eps={config.tie_eps}{lists})"
    return f"rrf(k={config.rrf_k},w_dense={config.w_dense},w_bm25={config.w_bm25}{lists})"


class Evaluation:
    """One store/model session that can score many configs on a split."""

    def __init__(self, *, rebuild: bool = False, reranker: bool = False) -> None:
        self.store = open_store()
        self.embedder = Embedder()
        if rebuild:
            counts = build_indexes(self.store, self.embedder)
            print(f"indexed with {self.store.backend}: {counts}")
        self.retriever = Retriever(self.store, self.embedder, Reranker() if reranker else None)
        self.provision_rows = self.store.rows(PROVISIONS_TABLE)
        naive = self.store.rows(NAIVE_TABLE)
        self.naive_text = {row["record_id"]: row for row in naive}
        records = len({row["record_id"] for row in self.provision_rows})
        self.sizes = {
            "structured": (len(self.provision_rows), records),
            "naive": (len(naive), len(naive)),
        }

    def item_scores(self, config: RetrievalConfig, items) -> dict[str, list]:
        per_category: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for item, score in self.scored(config, items):
            per_category[item.category].append(score)
        return per_category

    def scored(self, config: RetrievalConfig, items) -> list[tuple[EvalItem, tuple]]:
        out = []
        for item in items:
            ranked = self.retriever.search(item.question, config, on=item.ref_date, k=10)
            if config.index == "naive":
                matchers = naive_matchers(item, self.provision_rows)
                hits = []
                for chunk_id in ranked:
                    chunk = self.naive_text[chunk_id]
                    hit = []
                    for path, (source, marks) in matchers.items():
                        if source is not None and chunk["path"] != source:
                            continue
                        n = len(next(iter(marks))) if marks else 8
                        if marks & shingles(chunk["text"], n):
                            hit.append(path)
                    hits.append(hit)
            else:
                gold = gold_versions(item, self.provision_rows)
                hits = [[path for path, rid in gold.items() if rid == result] for result in ranked]
            out.append((item, score_item(hits, len(item.gold_paths))))
        return out

    def rows(self, config: RetrievalConfig, items, *, split: str, stamp: str, commit: str):
        scored = self.scored(config, items)
        size, records = self.sizes[config.index]
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for item, score in scored:
            for source in ("all", source_kind(item)):
                groups[(source, "overall")].append(score)
                groups[(source, item.category)].append(score)
        sources = ("all", "hand", "templated")
        order = sorted(groups, key=lambda k: (sources.index(k[0]), k[1] != "overall", k[1]))
        out = []
        for source, category in order:
            scores = groups[(source, category)]
            out.append(
                {
                    "timestamp": stamp,
                    "git_hash": commit,
                    "split": split,
                    "source": source,
                    "config": config.name,
                    "index_size": size,
                    "index_records": records,
                    "category": category,
                    "n": len(scores),
                    "recall@5": round(mean(score[0] for score in scores), 4),
                    "recall@10": round(mean(score[1] for score in scores), 4),
                    "mrr@10": round(mean(score[2] for score in scores), 4),
                    "fusion": fusion_label(config),
                    "backend": self.store.backend,
                    "relevance": "text_overlap" if config.index == "naive" else "version",
                }
            )
        return out


def split_items(split: str, source: str | None = None) -> list[EvalItem]:
    """Scored items with gold paths in the dev or test split (eval/split.json); ``source``
    ("hand" or "templated") keeps only that kind."""
    return [
        item
        for item in items_in(split)
        if item.gold_paths and (source is None or source_kind(item) == source)
    ]


def evaluate(
    configs: tuple[RetrievalConfig, ...] | None = None,
    *,
    split: str = "test",
    rebuild: bool = True,
) -> list[dict]:
    configs = configs or CONFIGS
    session = Evaluation(rebuild=rebuild, reranker=any(config.rerank for config in configs))
    items = split_items(split)
    stamp, commit = datetime.now(UTC).isoformat(timespec="seconds"), git_hash()
    results: list[dict] = []
    for config in configs:
        results.extend(session.rows(config, items, split=split, stamp=stamp, commit=commit))
    return results


def write_results(results: list[dict], path: Path = RESULTS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(results)


def print_table(results: list[dict]) -> None:
    print(
        f"{'config':14} {'size':>5} {'source':9} {'category':15} {'n':>3} "
        f"{'R@5':>6} {'R@10':>6} {'MRR':>6}"
    )
    for row in results:
        print(
            f"{row['config']:14} {row['index_size']:>5} {row['source']:9} {row['category']:15} "
            f"{row['n']:>3} "
            f"{row['recall@5']:>6.3f} {row['recall@10']:>6.3f} {row['mrr@10']:>6.3f}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval evaluation (no LLM).")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--configs", nargs="*", help="config names to run (default: all)")
    parser.add_argument("--no-rebuild", action="store_true", help="reuse the existing index")
    parser.add_argument(
        "--no-typed-citations",
        action="store_true",
        help="ablation: citations match records of any doc type",
    )
    parser.add_argument(
        "--no-split-bm25",
        action="store_true",
        help="ablation: one BM25 index over Act, Rules and rate-schedule rows",
    )
    args = parser.parse_args()
    ladder = ablation_configs(
        split_bm25=not args.no_split_bm25, typed_citations=not args.no_typed_citations
    )
    configs = tuple(c for c in ladder if not args.configs or c.name in args.configs)
    results = evaluate(configs, split=args.split, rebuild=not args.no_rebuild)
    write_results(results)
    print_table(results)
    print(f"appended {len(results)} rows to {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

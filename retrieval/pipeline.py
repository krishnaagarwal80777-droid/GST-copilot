"""Retrieval pipeline (no LLM): exact match, BM25, dense, RRF fusion, date filter — all flags.

Results are provision versions (record ids, ``<path>@v<N>``), best first. Chunked records
(rate schedules) are collapsed to their best-ranked chunk. With ``date_filter`` every path —
exact match, BM25 and dense — sees only versions valid on the reference date and enacted by
the knowledge date.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from retrieval.bm25 import Bm25Index
from retrieval.config import RetrievalConfig
from retrieval.corpus import build_rows
from retrieval.embed import Embedder
from retrieval.exact_match import exact_paths, parse_citations
from retrieval.fusion import rrf
from retrieval.naive import naive_rows
from retrieval.store import VectorStore, row_dict

PROVISIONS_TABLE = "provisions"
NAIVE_TABLE = "naive"


def build_indexes(store: VectorStore, embedder: Embedder) -> dict[str, int]:
    """(Re)build both tables from source; embeddings come from the disk cache when present."""
    rows = [row_dict(row) for row in build_rows()]
    store.write(PROVISIONS_TABLE, rows, embedder.encode([row["text"] for row in rows]))
    raw = naive_rows(embedder.tokenizer)
    store.write(NAIVE_TABLE, raw, embedder.encode([row["text"] for row in raw]))
    return {PROVISIONS_TABLE: len(rows), NAIVE_TABLE: len(raw)}


def _valid(row: dict[str, Any], on: date, known_by: date) -> bool:
    day = on.isoformat()
    return (
        row["effective_from"] <= day <= row["effective_to"]
        and row["in_force_since"] <= known_by.isoformat()
    )


def _dedupe(record_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(record_ids))


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder
        self._tables: dict[str, tuple[list[dict[str, Any]], Bm25Index | None]] = {}

    def _table(self, name: str, with_bm25: bool) -> tuple[list[dict[str, Any]], Bm25Index | None]:
        rows, bm25 = self._tables.get(name, (None, None))
        if rows is None:
            rows = self.store.rows(name)
        if with_bm25 and bm25 is None:
            bm25 = Bm25Index([row["text"] for row in rows])
        self._tables[name] = (rows, bm25)
        return rows, bm25

    def search(
        self,
        query: str,
        config: RetrievalConfig,
        *,
        on: date,
        known_by: date | None = None,
        k: int = 10,
    ) -> list[str]:
        known_by = known_by or date.today()
        table = NAIVE_TABLE if config.index == "naive" else PROVISIONS_TABLE
        rows, bm25 = self._table(table, config.bm25)
        filter_on = on if config.date_filter else None
        mask = [_valid(row, on, known_by) for row in rows] if config.date_filter else None

        rankings: list[list[str]] = []
        if config.dense:
            vector = self.embedder.encode([query])[0]
            hits = self.store.search(
                table, vector, config.candidates, on=filter_on, known_by=known_by
            )
            rankings.append(_dedupe([hit["record_id"] for hit in hits]))
        if config.bm25 and bm25 is not None:
            order = bm25.rank(query, mask=mask, k=config.candidates)
            rankings.append(_dedupe([rows[index]["record_id"] for index in order]))
        fused = rankings[0] if len(rankings) == 1 else rrf(rankings, k=config.rrf_k)

        if config.exact_match and config.index == "structured":
            pinned = self._exact(query, rows, mask)
            fused = pinned + [record_id for record_id in fused if record_id not in pinned]
        return fused[:k]

    @staticmethod
    def _exact(query: str, rows: list[dict[str, Any]], mask: list[bool] | None) -> list[str]:
        """Rows for every cited provision, citation order, newest version first per path."""
        candidates = [row for index, row in enumerate(rows) if mask is None or mask[index]]
        indexed_paths = sorted({row["path"] for row in candidates})
        wants_proviso = "proviso" in query.lower()
        pinned: list[str] = []
        for cited in parse_citations(query):
            paths = exact_paths(cited, indexed_paths)
            if wants_proviso:
                paths = sorted(paths, key=lambda path: "/proviso" not in path)
            for path in paths:
                versions = sorted(
                    (row for row in candidates if row["path"] == path),
                    key=lambda row: row["effective_from"],
                    reverse=True,
                )
                pinned.extend(row["record_id"] for row in versions)
        return _dedupe(pinned)

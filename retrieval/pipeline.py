"""Retrieval pipeline (no LLM): exact match, BM25, dense, RRF fusion, date filter — all flags.

Results are provision versions (record ids, ``<path>@v<N>``), best first. Chunked records
(rate schedules) are collapsed to their best-ranked chunk. With ``date_filter`` every path —
exact match, BM25 and dense — sees only versions valid on the reference date and enacted by
the knowledge date.

With ``split_bm25`` the structured index has two BM25 indexes, one over Act and Rules rows and
one over rate-schedule rows, and each contributes its own ranking to the fusion. In a single
index the 2,723 short schedule rows set the average document length and the term statistics
for the 1,054 long Act and Rules rows.

With ``typed_citations`` a citation matches only records of its doc type: exact match pins only
rows of the cited type, and a query citing only rules/sections (or only notifications) skips
the BM25 index of the other doc types.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from retrieval.bm25 import Bm25Index
from retrieval.config import RetrievalConfig
from retrieval.corpus import build_rows
from retrieval.embed import Embedder
from retrieval.exact_match import citation_doc_type, cited_doc_types, exact_paths, parse_citations
from retrieval.fusion import dense_first, weighted_rrf
from retrieval.naive import naive_rows
from retrieval.store import VectorStore, row_dict

PROVISIONS_TABLE = "provisions"
NAIVE_TABLE = "naive"
RATE_ACT_ID = "CTR"  # act_id of rate-schedule rows (retrieval.corpus.rate_rows)


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
    def __init__(self, store: VectorStore, embedder: Embedder, reranker=None) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._tables: dict[str, list[dict[str, Any]]] = {}
        # (table, split) -> BM25 indexes, each with the table positions of its rows.
        self._bm25: dict[tuple[str, bool], list[tuple[list[int], Bm25Index]]] = {}
        # (table, split) -> doc types of the rows in each BM25 index, in the same order.
        self._bm25_types: dict[tuple[str, bool], list[set[str]]] = {}

    def _table(self, name: str) -> list[dict[str, Any]]:
        if name not in self._tables:
            self._tables[name] = self.store.rows(name)
        return self._tables[name]

    def _bm25_indexes(self, name: str, split: bool) -> list[tuple[list[int], Bm25Index]]:
        """One index over the whole table, or (split) one over Act and Rules rows and one over
        rate-schedule rows. Only the structured table is split."""
        split = split and name == PROVISIONS_TABLE
        if (name, split) not in self._bm25:
            rows = self._table(name)
            everything = list(range(len(rows)))
            if split:
                rates = [i for i in everything if rows[i]["act_id"] == RATE_ACT_ID]
                parts = [[i for i in everything if rows[i]["act_id"] != RATE_ACT_ID], rates]
            else:
                parts = [everything]
            parts = [part for part in parts if part]
            self._bm25[(name, split)] = [
                (part, Bm25Index([rows[i]["text"] for i in part])) for part in parts
            ]
            self._bm25_types[(name, split)] = [
                {rows[i]["doc_type"] for i in part} for part in parts
            ]
        return self._bm25[(name, split)]

    def _bm25_rankings(
        self, name: str, query: str, config: RetrievalConfig, mask: list[bool] | None
    ) -> list[list[str]]:
        """Record ids ranked by each BM25 index of the table (one ranking per index)."""
        rows = self._table(name)
        indexes = self._bm25_indexes(name, config.split_bm25)
        types = self._bm25_types[(name, config.split_bm25 and name == PROVISIONS_TABLE)]
        cited = cited_doc_types(query) if config.typed_citations and len(indexes) > 1 else set()
        rankings: list[list[str]] = []
        for (positions, index), doc_types in zip(indexes, types, strict=True):
            if cited and not cited & doc_types:
                continue
            part_mask = None if mask is None else [mask[i] for i in positions]
            order = index.rank(query, mask=part_mask, k=config.candidates)
            rankings.append(_dedupe([rows[positions[i]]["record_id"] for i in order]))
        return rankings

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
        rows = self._table(table)
        filter_on = on if config.date_filter else None
        mask = [_valid(row, on, known_by) for row in rows] if config.date_filter else None

        dense_ids: list[str] = []
        similarity: dict[str, float] = {}
        if config.dense:
            vector = self.embedder.encode([query])[0]
            hits = self.store.search(
                table, vector, config.candidates, on=filter_on, known_by=known_by
            )
            for hit in hits:
                if hit["record_id"] not in similarity:
                    similarity[hit["record_id"]] = 1.0 - float(hit["_distance"])
                    dense_ids.append(hit["record_id"])
        bm25_rankings = self._bm25_rankings(table, query, config, mask) if config.bm25 else []
        bm25_weights = [config.w_bm25] * len(bm25_rankings)

        if config.dense and config.bm25:
            if config.fusion == "dense_first":
                bm25_ids = (
                    bm25_rankings[0]
                    if len(bm25_rankings) == 1
                    else weighted_rrf(bm25_rankings, bm25_weights, k=config.rrf_k)
                )
                fused = dense_first(dense_ids, similarity, bm25_ids, tie_eps=config.tie_eps)
            else:
                fused = weighted_rrf(
                    [dense_ids, *bm25_rankings], [config.w_dense, *bm25_weights], k=config.rrf_k
                )
        elif len(bm25_rankings) > 1:
            fused = weighted_rrf(bm25_rankings, bm25_weights, k=config.rrf_k)
        else:
            fused = dense_ids or (bm25_rankings[0] if bm25_rankings else [])

        if config.rerank and self.reranker is not None:
            head = self._rerank(query, fused[: config.rerank_top], rows, mask)
            fused = head + fused[config.rerank_top :]

        if config.exact_match and config.index == "structured":
            pinned = self._exact(query, rows, mask, typed=config.typed_citations)
            fused = pinned + [record_id for record_id in fused if record_id not in pinned]
        return fused[:k]

    def rows_for(
        self, record_ids: list[str], *, on: date, known_by: date | None = None
    ) -> list[dict[str, Any]]:
        """Structured-index rows (chunks) of these records that are valid on the date."""
        known_by = known_by or date.today()
        rows = self._table(PROVISIONS_TABLE)
        wanted = set(record_ids)
        return [row for row in rows if row["record_id"] in wanted and _valid(row, on, known_by)]

    def exact_hits(
        self, query: str, *, on: date, known_by: date | None = None, typed: bool = False
    ) -> list[str]:
        """Record ids the exact-match path pins for this query (servable, date-valid)."""
        known_by = known_by or date.today()
        rows = self._table(PROVISIONS_TABLE)
        mask = [_valid(row, on, known_by) for row in rows]
        return self._exact(query, rows, mask, typed=typed)

    def _rerank(
        self, query: str, record_ids: list[str], rows: list[dict[str, Any]], mask
    ) -> list[str]:
        """Cross-encoder order; a chunked record scores as its best chunk."""
        texts: list[str] = []
        owners: list[str] = []
        wanted = set(record_ids)
        for index, row in enumerate(rows):
            if row["record_id"] in wanted and (mask is None or mask[index]):
                texts.append(row["text"])
                owners.append(row["record_id"])
        best: dict[str, float] = {}
        for owner, score in zip(owners, self.reranker.scores(query, texts), strict=True):
            best[owner] = max(score, best.get(owner, float("-inf")))
        return sorted(record_ids, key=lambda rid: -best.get(rid, float("-inf")))

    @staticmethod
    def _exact(
        query: str, rows: list[dict[str, Any]], mask: list[bool] | None, *, typed: bool = False
    ) -> list[str]:
        """Rows for every cited provision, citation order, newest version first per path;
        ``typed``: only rows of the citation's doc type."""
        # Exact-match hits are citations, so only servable rows can be pinned.
        candidates = [
            row
            for index, row in enumerate(rows)
            if (mask is None or mask[index]) and row.get("servable", "1") == "1"
        ]
        wants_proviso = "proviso" in query.lower()
        pinned: list[str] = []
        for cited in parse_citations(query):
            if typed:
                doc_type = citation_doc_type(cited)
                of_type = [row for row in candidates if row["doc_type"] == doc_type]
            else:
                of_type = candidates
            paths = exact_paths(cited, sorted({row["path"] for row in of_type}))
            if wants_proviso:
                paths = sorted(paths, key=lambda path: "/proviso" not in path)
            for path in paths:
                versions = sorted(
                    (row for row in of_type if row["path"] == path),
                    key=lambda row: row["effective_from"],
                    reverse=True,
                )
                pinned.extend(row["record_id"] for row in versions)
        return _dedupe(pinned)

"""Generator context: the top servable chunks valid on the reference date, with IDs.

Candidates are the records the retrieval pipeline returns (exact match pinned first); each is
expanded to its index rows (chunks; a rate schedule has many) that are servable and valid on
the reference date. Every candidate chunk is scored with the cross-encoder. The context is the
best chunk of each exact-match record, then the highest-scoring remaining chunks, up to
``context_chunks``. Confidence = the best cross-encoder score among servable candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from generation.config import GenerationConfig
from retrieval.config import CONFIGS
from retrieval.pipeline import Retriever


@dataclass(frozen=True)
class Chunk:
    id: str  # index row id: "<path>@v<N>" or "<path>@v<N>#<block>" for rate schedules
    record_id: str
    path: str
    doc_type: str
    effective_from: str
    effective_to: str
    text: str
    score: float
    pinned: bool


@dataclass(frozen=True)
class Context:
    chunks: list[Chunk]
    confidence: float  # best cross-encoder score over servable candidates (0 if none)
    candidates: int  # servable candidate chunks scored


def retrieval_config(name: str):
    return next(config for config in CONFIGS if config.name == name)


def build_context(
    retriever: Retriever,
    question: str,
    on: date,
    config: GenerationConfig,
    *,
    known_by: date | None = None,
) -> Context:
    if retriever.reranker is None:
        raise ValueError("generation context needs the cross-encoder (Retriever(reranker=...))")
    rconfig = retrieval_config(config.retrieval)
    rconfig = replace(rconfig, split_bm25=config.split_bm25 and rconfig.bm25)
    # The date filter is not optional here: the generator only ever sees valid versions.
    record_ids = retriever.search(
        question, rconfig, on=on, known_by=known_by, k=config.candidate_records
    )
    pinned = set(retriever.exact_hits(question, on=on, known_by=known_by)) & set(record_ids)
    rows = [
        row
        for row in retriever.rows_for(record_ids, on=on, known_by=known_by)
        if row.get("servable", "1") == "1"
    ]
    if not rows:
        return Context([], 0.0, 0)
    scores = retriever.reranker.scores(question, [row["text"] for row in rows])
    chunks = sorted(
        (
            Chunk(
                id=row["id"],
                record_id=row["record_id"],
                path=row["path"],
                doc_type=row["doc_type"],
                effective_from=row["effective_from"],
                effective_to=row["effective_to"],
                text=row["text"],
                score=float(score),
                pinned=row["record_id"] in pinned,
            )
            for row, score in zip(rows, scores, strict=True)
        ),
        key=lambda chunk: (-chunk.score, chunk.id),
    )
    chosen: list[Chunk] = []
    seen_pinned: set[str] = set()
    for chunk in chunks:  # best chunk of each pinned record first, in score order
        if chunk.pinned and chunk.record_id not in seen_pinned:
            chosen.append(chunk)
            seen_pinned.add(chunk.record_id)
    chosen = chosen[: config.context_chunks]
    for chunk in chunks:
        if len(chosen) >= config.context_chunks:
            break
        if chunk not in chosen:
            chosen.append(chunk)
    return Context(chosen, max(chunk.score for chunk in chunks), len(chunks))

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gst_copilot.models import ProvisionRecord
from retrieval.bm25 import bm25_rank
from retrieval.exact_match import exact_match


@dataclass(frozen=True)
class RetrievalResult:
    path: str
    score: float
    matched_by: str


def retrieve(
    query: str, records: Sequence[ProvisionRecord], *, limit: int = 5
) -> list[RetrievalResult]:
    """Hybrid baseline retrieval: exact match first, then BM25 fallback."""
    exact_hits = exact_match(query, records)
    if exact_hits:
        ranked: list[RetrievalResult] = [
            RetrievalResult(path=record.path, score=float(1000 - index), matched_by="exact_match")
            for index, record in enumerate(exact_hits)
        ]
        seen = {item.path for item in ranked}
        for record, score in bm25_rank(query, records):
            if record.path in seen:
                continue
            ranked.append(RetrievalResult(path=record.path, score=score, matched_by="bm25"))
        ranked.sort(key=lambda item: (-item.score, item.path))
        return ranked[:limit]

    ranked = [
        RetrievalResult(path=record.path, score=score, matched_by="bm25")
        for record, score in bm25_rank(query, records)
    ]
    return ranked[:limit]

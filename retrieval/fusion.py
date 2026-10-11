"""Reciprocal rank fusion over ranked lists of ids."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[str]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def weighted_rrf(
    rankings: Sequence[Sequence[str]], weights: Sequence[float], *, k: int = 60
) -> list[str]:
    """Weighted RRF: score(id) = sum of weight_i / (k + rank_i)."""
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def dense_first(
    dense: Sequence[str],
    similarity: dict[str, float],
    bm25: Sequence[str],
    *,
    tie_eps: float = 0.0,
) -> list[str]:
    """Dense order, with BM25 rank breaking near-ties; BM25-only ids follow all dense hits.

    Dense hits are grouped into similarity bands of width tie_eps (from the top hit down);
    within a band, the better BM25 rank comes first. tie_eps=0 keeps the dense order exactly.
    """
    bm25_rank = {item: rank for rank, item in enumerate(bm25)}
    missing = len(bm25)
    top = similarity[dense[0]] if dense else 0.0

    def key(position_item: tuple[int, str]) -> tuple[int, int, int]:
        position, item = position_item
        band = int((top - similarity[item]) // tie_eps) if tie_eps > 0 else position
        return band, bm25_rank.get(item, missing), position

    ordered = [item for _, item in sorted(enumerate(dense), key=key)]
    seen = set(ordered)
    return ordered + [item for item in bm25 if item not in seen]

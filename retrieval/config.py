"""Retrieval feature flags. Every pipeline feature is a flag so ablations run from one command."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    index: Literal["structured", "naive"] = "structured"
    dense: bool = True
    bm25: bool = False
    exact_match: bool = False
    date_filter: bool = False
    # Fusion of dense and BM25 rankings (used when both are on).
    #   rrf:         score = w_dense / (rrf_k + rank_dense) + w_bm25 / (rrf_k + rank_bm25)
    #   dense_first: dense order; BM25 rank only reorders dense hits whose cosine similarity is
    #                within tie_eps of each other; BM25-only hits follow all dense hits.
    fusion: Literal["rrf", "dense_first"] = "rrf"
    rrf_k: int = 60
    w_dense: float = 1.0
    w_bm25: float = 1.0
    tie_eps: float = 0.0
    # Separate BM25 indexes for Act/Rules rows and rate-schedule rows (structured index only);
    # each index adds its own ranking to the fusion with weight w_bm25.
    split_bm25: bool = False
    rerank: bool = False
    rerank_top: int = 20
    candidates: int = 50


# Fusion chosen on the dev split by eval/tune_fusion.py (eval/results/fusion_tuning_dev.csv,
# commit ff8c5c9): RRF with k=10 and equal weights. Plain RRF (k=60, equal weights) is the
# untuned default; on dev dense+bm25 MRR@10 rose from 0.561 to 0.607 (24 items).
TUNED_FUSION: dict = {"fusion": "rrf", "rrf_k": 10, "w_dense": 1.0, "w_bm25": 1.0, "tie_eps": 0.0}


def with_fusion(config: RetrievalConfig, fusion: dict) -> RetrievalConfig:
    return replace(config, **fusion)


def ablation_configs(
    fusion: dict | None = None, *, split_bm25: bool = True
) -> tuple[RetrievalConfig, ...]:
    """Cumulative ablation ladder reported by `make eval-retrieval`. ``split_bm25`` applies to
    every rung that uses BM25 (`python -m eval.retrieval --no-split-bm25` turns it off)."""
    fused = TUNED_FUSION if fusion is None else fusion
    ladder = (
        RetrievalConfig("naive", index="naive"),
        RetrievalConfig("dense"),
        RetrievalConfig("bm25", dense=False, bm25=True),
        with_fusion(RetrievalConfig("dense+bm25", bm25=True), fused),
        with_fusion(RetrievalConfig("+exact_match", bm25=True, exact_match=True), fused),
        with_fusion(
            RetrievalConfig("+date_filter", bm25=True, exact_match=True, date_filter=True), fused
        ),
        with_fusion(
            RetrievalConfig(
                "+reranker", bm25=True, exact_match=True, date_filter=True, rerank=True
            ),
            fused,
        ),
    )
    return tuple(replace(config, split_bm25=split_bm25 and config.bm25) for config in ladder)


CONFIGS: tuple[RetrievalConfig, ...] = ablation_configs()

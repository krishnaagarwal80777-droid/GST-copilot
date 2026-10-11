"""Retrieval feature flags. Every pipeline feature is a flag so ablations run from one command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    index: Literal["structured", "naive"] = "structured"
    dense: bool = True
    bm25: bool = False
    exact_match: bool = False
    date_filter: bool = False
    rrf_k: int = 60
    candidates: int = 50


# Cumulative ablation ladder reported by `make eval-retrieval`.
CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig("naive", index="naive"),
    RetrievalConfig("dense"),
    RetrievalConfig("bm25", dense=False, bm25=True),
    RetrievalConfig("dense+bm25", bm25=True),
    RetrievalConfig("+exact_match", bm25=True, exact_match=True),
    RetrievalConfig("+date_filter", bm25=True, exact_match=True, date_filter=True),
)

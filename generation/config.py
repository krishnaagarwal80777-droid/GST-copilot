"""Generation feature flags (one flag per feature, so ablations run from the eval command)."""

from __future__ import annotations

from dataclasses import dataclass

# Confidence gate: abstain when no servable context chunk scores at least this with the
# bge-reranker-v2-m3 cross-encoder (sigmoid scores in [0, 1]). Tuned on the dev split by
# eval/tune_abstention.py (eval/results/abstention_tuning_dev.csv): on 94 dev items the 3
# unanswerable ones score <= 0.437 and every answerable one >= 0.758; the midpoint is used.
# Three negatives is thin evidence: treat this as a coarse filter, not a calibrated one.
TUNED_ABSTAIN_THRESHOLD: float = 0.597412


@dataclass(frozen=True)
class GenerationConfig:
    name: str = "full"
    retrieval: str = "+reranker"  # retrieval.config.CONFIGS name used to pick candidates
    split_bm25: bool = True  # separate BM25 indexes for Act/Rules and rate-schedule rows
    typed_citations: bool = True  # a cited rule/section/notification matches only its doc type
    context_chunks: int = 6  # servable chunks handed to the generator
    candidate_records: int = 10  # records from retrieval whose chunks compete for the context
    # Abstain before generation when the reference date is after the knowledge horizon of
    # the question's source type (generation.horizon).
    date_gate: bool = True
    gate: bool = True  # abstain before generation when confidence < abstain_threshold
    abstain_threshold: float = TUNED_ABSTAIN_THRESHOLD
    verify_citations: bool = True  # drop citation ids not in context/servable, then claims

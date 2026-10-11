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
    context_chunks: int = 6  # servable chunks handed to the generator
    candidate_records: int = 10  # records from retrieval whose chunks compete for the context
    gate: bool = True  # abstain before generation when confidence < abstain_threshold
    abstain_threshold: float = TUNED_ABSTAIN_THRESHOLD
    verify_citations: bool = True  # drop citation ids not in context/servable, then claims

"""Citation verification as a data check: every citation id must be a servable context chunk.

Citation ids that are not in the context, or not servable, are removed; a claim left with no
citation is unsupported and dropped. (Checking that a chunk entails its claim, e.g. by NLI, is
not done here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from generation.context import Chunk
from generation.schemas import Claim, GenerationOutput


@dataclass(frozen=True)
class Verified:
    claims: list[Claim]
    dropped_claims: list[Claim] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)  # ids not in context/servable
    ref_date_matches: bool = True


def verify(output: GenerationOutput, chunks: list[Chunk], ref_date: str) -> Verified:
    # Context chunks are servable by construction (generation.context); the id check is the
    # servability check.
    allowed = {chunk.id for chunk in chunks}
    kept: list[Claim] = []
    dropped: list[Claim] = []
    bad_ids: list[str] = []
    for claim in output.claims:
        valid = [cid for cid in dict.fromkeys(claim.citation_ids) if cid in allowed]
        bad_ids.extend(cid for cid in claim.citation_ids if cid not in allowed)
        if valid and claim.text.strip():
            kept.append(claim.model_copy(update={"citation_ids": valid}))
        else:
            dropped.append(claim)
    return Verified(kept, dropped, bad_ids, output.ref_date_used.strip() == ref_date)

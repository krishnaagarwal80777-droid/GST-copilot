"""Fixed answer template. The LLM writes claims; this module writes the prose around them."""

from __future__ import annotations

from datetime import date

from generation.context import Chunk
from generation.prompts import chunk_label
from generation.schemas import Claim

DISCLAIMER = (
    "This is general information about the law as of the reference date, generated from the "
    "cited provisions. It is not professional tax or legal advice."
)
SECTIONS = (("conclusion", "Conclusion"), ("condition", "Conditions"), ("exception", "Exceptions"))


def _cites(claim: Claim, by_id: dict[str, Chunk]) -> str:
    # An id outside the context (only possible with verification off) is shown as written.
    labels = dict.fromkeys(
        chunk_label(by_id[cid].path) if cid in by_id else cid for cid in claim.citation_ids
    )
    return "; ".join(labels)


def render_answer(
    claims: list[Claim], assumptions: list[str], ref_date: date, chunks: list[Chunk]
) -> str:
    by_id = {chunk.id: chunk for chunk in chunks}
    lines: list[str] = []
    for kind, title in SECTIONS:
        lines.append(f"**{title}**")
        selected = [claim for claim in claims if claim.kind == kind]
        lines.extend(f"- {c.text} [{_cites(c, by_id)}]" for c in selected)
        if not selected:
            lines.append("- None stated in the cited provisions.")
        lines.append("")
    lines.append("**Assumptions made**")
    lines.extend(f"- {text}" for text in assumptions or ["None."])
    lines += ["", f"**Reference date used:** {ref_date.isoformat()}", "", f"_{DISCLAIMER}_"]
    return "\n".join(lines)


def render_abstention(reason: str, ref_date: date, chunks: list[Chunk]) -> str:
    lines = [
        "**Not enough information**",
        f"- {reason}",
        "",
        "**Closest provisions**",
    ]
    nearest = list(dict.fromkeys(chunk_label(chunk.path) for chunk in chunks))[:5]
    lines.extend(f"- {label}" for label in nearest or ["None found."])
    lines += ["", f"**Reference date used:** {ref_date.isoformat()}", "", f"_{DISCLAIMER}_"]
    return "\n".join(lines)

"""Prompts for the generator (Gemini) and the answer-correctness judge (Groq)."""

from __future__ import annotations

from datetime import date

from generation.context import Chunk
from retrieval.corpus import citation_label

GENERATOR_SYSTEM = """\
You answer questions about Indian GST law (CGST Act, CGST Rules, central tax rate \
notifications) for a stated reference date, using ONLY the provisions in the context. Each \
context chunk has an ID and is the version of that provision in force on the reference date.

Return JSON with exactly these fields:
- "abstain": true if the context does not contain what is needed to answer (the question is \
outside these sources, asks about something not in the context, or asks about a future or \
pre-GST period); otherwise false.
- "ref_date_used": the reference date you answered for, as YYYY-MM-DD.
- "claims": a list of short statements. Each claim has:
  - "kind": "conclusion" (the direct answer, first), "condition" (a requirement the answer \
depends on) or "exception" (a proviso, exclusion or carve-out);
  - "text": the statement, following the wording of the provision;
  - "citation_ids": the IDs of the context chunks that state it. Use only IDs that appear in \
the context, copied exactly. A claim with no supporting chunk must not be made.
- "assumptions": facts you assumed about the question that the provision depends on (may be \
empty).

Rules:
- Put the rule first, then its provisos and explanations.
- Do no arithmetic. Quote rates, amounts, periods and dates exactly as the provisions state \
them; never compute a figure.
- Do not rely on outside knowledge of the law, later amendments or earlier versions: the \
context is the law on the reference date.
- If abstaining, return an empty "claims" list.
"""

JUDGE_SYSTEM = """\
You grade answers to Indian GST questions against a reference. Return JSON: \
{"verdict": "correct" | "partially_correct" | "incorrect", "rationale": "<one or two \
sentences>"}.

- "correct": the answer's conclusion matches the reference for the stated reference date and \
nothing it states contradicts the reference.
- "partially_correct": the conclusion matches but a stated condition, exception or figure is \
wrong or an essential part of the reference is missing.
- "incorrect": the conclusion is wrong, contradicts the reference, states the law of a \
different date, or the answer declines to answer.
Judge substance, not wording or length. Rates and figures must match exactly.
"""


def chunk_label(chunk_path: str) -> str:
    """Human citation for an index path."""
    if chunk_path.startswith("CTR/"):
        parts = chunk_path.split("/")
        label = f"Notification No. {parts[1].replace('-', '/')}-Central Tax (Rate)"
        return label + (f", Schedule {parts[2][3:]}" if len(parts) > 2 else "")
    source = "CGST Act" if chunk_path.split("/")[1].startswith("s") else "CGST Rules"
    return f"{source}, {citation_label(chunk_path)}"


def _validity(chunk: Chunk) -> str:
    end = "" if chunk.effective_to.startswith("9999") else chunk.effective_to
    return f"in force {chunk.effective_from} to {end or 'date'}"


def generator_prompt(question: str, ref_date: date, chunks: list[Chunk]) -> str:
    blocks = [
        f"[ID: {chunk.id}] {chunk_label(chunk.path)} ({_validity(chunk)})\n{chunk.text}"
        for chunk in chunks
    ]
    return (
        f"Reference date: {ref_date.isoformat()}\n"
        f"Question: {question}\n\n"
        "Context:\n\n" + "\n\n".join(blocks)
    )


# Direction-neutral: an empty diff side means that version's text is contained in the other's
# (the amendment added text if this is the earlier version, removed it if the later one).
NONE_OWN = "(none: no text is unique to that version)"


def judge_prompt(
    question: str, ref_date: date, reference: str, answer: str, distinguishing: dict | None = None
) -> str:
    parts = [
        f"Reference date: {ref_date.isoformat()}",
        f"Question: {question}",
        f"Reference answer (the law on the reference date):\n{reference}",
    ]
    if distinguishing:
        parts.append(
            "The provision was amended near this date. Text only in the version in force on "
            "the reference date:\n"
            f"{distinguishing.get('only_in_this') or NONE_OWN}\n"
            "Text only in the other version (stating this as the law on the reference date "
            "is wrong):\n"
            f"{distinguishing.get('only_in_other') or NONE_OWN}"
        )
    parts.append(f"Answer to grade:\n{answer}")
    return "\n\n".join(parts)

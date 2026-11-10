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

# Judge v2: the judge also sees the gold provision text in force on the reference date.
JUDGE_SYSTEM_V2 = JUDGE_SYSTEM + (
    "You are also given the gold provision text in force on the reference date. Extra "
    "statements supported by the provision text are not errors; omissions of material "
    "conditions are.\n"
)
PROVISION_SAME_AS_REFERENCE = "(the same text as the reference answer above)"


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
    question: str,
    ref_date: date,
    reference: str,
    answer: str,
    distinguishing: dict | None = None,
    provision: str | None = None,
) -> str:
    parts = [
        f"Reference date: {ref_date.isoformat()}",
        f"Question: {question}",
        f"Reference answer (the law on the reference date):\n{reference}",
    ]
    if provision is not None:  # judge v2
        parts.append(f"Gold provision text (in force on the reference date):\n{provision}")
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


# --- Generator prompt variants (GenerationConfig.prompt_variant) ----------------------------
# baseline:         GENERATOR_SYSTEM as is.
# strict_citations: every claim cites, and quotes the deciding words of the cited provision.
# few_shot:         two worked examples. Their provisions and goods are in no eval item (gold
#                   paths of every item, dev, test or pending; question and gold-answer words);
#                   chunk text is copied from the index rows.
# provision_first:  a "governing" object (governing chunk ID, its in-force dates, whether the
#                   reference date falls inside them) is filled before the claims.
PROMPT_VARIANTS = ("baseline", "strict_citations", "few_shot", "provision_first")

STRICT_CITATION_RULES = """\
- Every claim must carry at least one citation ID; write no statement without one.
- In every claim, quote inside double quotes the exact words of the cited chunk that decide \
it, copied verbatim (for example the rate, the condition or the exception as worded).
"""

PROVISION_FIRST_RULES = """\
- Before the claims, fill "governing": "citation_id" is the ID of the context chunk that \
governs the question, "in_force" is the period that chunk states it is in force, and \
"covers_ref_date" says whether the reference date falls inside that period. Then answer from \
that chunk first.
"""


def _example_chunk(chunk_id: str, doc_type: str, effective_from: str, text: str) -> Chunk:
    path = chunk_id.split("@", 1)[0]
    return Chunk(
        chunk_id,
        chunk_id.split("#")[0],
        path,
        doc_type,
        effective_from,
        "9999-12-31",
        text,
        0.0,
        False,
    )


_FEW_SHOT = (
    (
        "How is the value of a supply of goods determined when it cannot be determined under "
        "any of the preceding valuation rules?",
        date(2023, 4, 1),
        [
            _example_chunk(
                "CGST/r30@v1",
                "rule",
                "2021-05-18",
                "Central Goods and Services Tax Rules, 2017, rule 30 - Value of supply of goods "
                "or services or both based on cost: 30. Value of supply of goods or services or "
                "both based on cost.-Where the value of a supply of goods or services or both is "
                "not determinable by any of the preceding rules of this Chapter, the value shall "
                "be one hundred and ten percent of the cost of production or manufacture or the "
                "cost of acquisition of such goods or the cost of provision of such services.",
            ),
            _example_chunk(
                "CGST/r76@v1",
                "rule",
                "2021-05-18",
                "Central Goods and Services Tax Rules, 2017, rule 76 - Claim of reduction in "
                "output tax liability more than once: 76. Claim of reduction in output tax "
                "liability more than once .-The duplication of claims for reduction in output "
                "tax liability in the details of outward supplies shall be communicated to the "
                "registered person in FORM GST MIS -1 electronically through the common portal.",
            ),
        ],
        {
            "abstain": False,
            "ref_date_used": "2023-04-01",
            "claims": [
                {
                    "kind": "conclusion",
                    "text": "Where the value is not determinable by any of the preceding rules "
                    "of the Chapter, the value is one hundred and ten percent of the cost of "
                    "production or manufacture or the cost of acquisition of such goods.",
                    "citation_ids": ["CGST/r30@v1"],
                }
            ],
            "assumptions": [],
        },
    ),
    (
        "What is the rate of central tax on intra-State supply of ferro-alloys?",
        date(2025, 12, 1),
        [
            _example_chunk(
                "CTR/9-2025/schII@v1#261",
                "notification",
                "2025-09-22",
                "Notification No. 9/2025-Central Tax (Rate), Schedule II - central tax rate 9 per "
                "cent: S. No. 261. 7202 Ferro-alloys",
            ),
            _example_chunk(
                "CTR/9-2025/schII@v1#173",
                "notification",
                "2025-09-22",
                "Notification No. 9/2025-Central Tax (Rate), Schedule II - central tax rate 9 per "
                "cent: S. No. 173. 4806 40 10 Glassine papers",
            ),
        ],
        {
            "abstain": False,
            "ref_date_used": "2025-12-01",
            "claims": [
                {
                    "kind": "conclusion",
                    "text": "Ferro-alloys (heading 7202) are at S. No. 261 of Schedule II of "
                    "Notification No. 9/2025-Central Tax (Rate), which carries central tax at "
                    "9 per cent.",
                    "citation_ids": ["CTR/9-2025/schII@v1#261"],
                }
            ],
            "assumptions": [],
        },
    ),
)


def _few_shot_block() -> str:
    import json

    parts = ["Examples (the context of each example is given with it):"]
    for number, (question, ref_date, chunks, output) in enumerate(_FEW_SHOT, 1):
        parts.append(
            f"Example {number} input:\n{generator_prompt(question, ref_date, chunks)}\n\n"
            f"Example {number} output:\n{json.dumps(output, ensure_ascii=False)}"
        )
    return "\n\n".join(parts) + "\n"


def generator_system(variant: str = "baseline") -> str:
    """System prompt of a generator prompt variant."""
    if variant == "baseline":
        return GENERATOR_SYSTEM
    if variant == "strict_citations":
        return GENERATOR_SYSTEM + STRICT_CITATION_RULES
    if variant == "few_shot":
        return GENERATOR_SYSTEM + "\n" + _few_shot_block()
    if variant == "provision_first":
        return GENERATOR_SYSTEM + PROVISION_FIRST_RULES
    raise ValueError(f"unknown prompt variant {variant!r}; choose from {PROMPT_VARIANTS}")

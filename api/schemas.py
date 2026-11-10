"""Request and response models for POST /ask."""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from retrieval.corpus import RATE_NOTIFICATIONS

# The earliest effective date in the corpus: the commencement of the 2017 rate notifications.
CORPUS_START: date = min(n.effective_from for n in RATE_NOTIFICATIONS)
# Transport limit only; the question-length guardrail (api.config) refuses well below this.
MAX_REQUEST_CHARS = 20_000

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso_date_string(value: Any) -> Any:
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        raise ValueError("expected a date as YYYY-MM-DD")
    return value


# An ISO "YYYY-MM-DD" string only: not a timestamp, a datetime or another format.
IsoDate = Annotated[date, BeforeValidator(_iso_date_string)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    reference_date: IsoDate
    knowledge_date: IsoDate | None = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is blank")
        return value

    @model_validator(mode="after")
    def check_dates(self) -> AskRequest:
        if self.reference_date < CORPUS_START:
            raise ValueError(f"reference_date is before {CORPUS_START.isoformat()} (GST start)")
        if self.knowledge_date is not None:
            if self.knowledge_date < CORPUS_START:
                raise ValueError(f"knowledge_date is before {CORPUS_START.isoformat()}")
            if self.knowledge_date > date.today():
                raise ValueError("knowledge_date is in the future")
        return self


class ClaimOut(BaseModel):
    text: str
    kind: str
    citation_ids: list[str]
    citation_numbers: list[int]  # Citation.number of each id, in the same order


class Citation(BaseModel):
    number: int  # 1-based, in order of first citation
    id: str  # chunk id: "<path>@v<N>", or "<path>@v<N>#<row>" for a rate-schedule row
    version_id: str  # "<path>@v<N>"
    path: str
    label: str
    doc_type: str
    effective_from: date
    effective_to: date | None  # None: still in force
    source_file: str | None  # filename in data/raw/ (see manifest.json)
    source_url: str | None  # None when the file was added by hand
    text: str


class ToolCallOut(BaseModel):
    """One calculator call (generation.tools_router): what the model asked for and what Python
    computed. ``result`` values are strings, exactly as the answer must quote them."""

    name: str  # "rate_lookup" or "gst_split"
    arguments: dict[str, Any]
    ok: bool  # False when the arguments were invalid
    error: str | None
    status: str | None  # ToolResult.status: ok, ambiguous, not_found, refused
    result: dict[str, Any] | None
    steps: list[str]
    citations: list[str]  # index row ids the result rests on
    notes: list[str]
    candidates: list[dict[str, Any]]  # every matching row of an ambiguous lookup
    latency_ms: float


class AskResponse(BaseModel):
    request_id: str
    answer: str  # rendered from the fixed template
    claims: list[ClaimOut]
    citations: list[Citation]
    assumptions: list[str]  # stated by the generator; empty on abstention
    abstained: bool
    # generation.answer.AbstainReason, or "guardrail" for a refused question
    abstain_reason: str | None
    guardrail: str | None  # generation.guardrails.Guardrail that refused the question
    reference_date: date
    knowledge_date: date | None
    timings_ms: dict[str, float]  # stage -> ms for the stages that ran, plus "total"
    model: str | None  # the generator that answered; None when no LLM call was made
    # Calculator calls made for this answer, in order (empty when no tool was used).
    tool_calls: list[ToolCallOut] = Field(default_factory=list)


class Health(BaseModel):
    status: str
    pipeline_loaded: bool
    store: str | None

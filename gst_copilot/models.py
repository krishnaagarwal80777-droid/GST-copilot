from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

DocumentType = Literal["act", "rule", "notification", "circular", "faq"]
NodeType = Literal[
    "section",
    "subsection",
    "clause",
    "sub_clause",
    "proviso",
    "explanation",
    "rule",
    "paragraph",
]
EdgeType = Literal[
    "REFERS_TO",
    "SUBJECT_TO",
    "NOTWITHSTANDING",
    "DEFINED_IN",
    "CHILD_OF",
    "AMENDS",
    "CLARIFIES",
]


class ProvisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    doc_type: DocumentType
    act_id: str
    path: str
    node_type: NodeType
    heading: str | None = None
    text: str
    context_prefix: str = ""
    verified: bool = True
    verified_by: str | None = None
    # False for retrieval distractors (unversioned snapshot text): retrievable, never citable.
    servable: bool = True
    effective_from: date
    effective_to: date | None = None
    in_force_since: date
    amended_by: str | None = None
    source_url: str | None = None
    page: int | None = None
    char_span: tuple[int, int] | None = None

    @model_validator(mode="after")
    def validate_temporal_fields(self) -> "ProvisionRecord":
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to must be on or after effective_from")
        if self.char_span is not None and self.char_span[1] < self.char_span[0]:
            raise ValueError("char_span end must be on or after its start")
        return self


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: EdgeType
    source_url: str | None = None
    confidence: float | None = None


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    ref_date: date
    gold_paths: list[str]
    gold_answer: str
    category: str
    answerable: bool
    expected_tool_output: dict[str, Any] | None = None
    source: Literal["faq", "icai", "hand"]
    group_id: str | None = None

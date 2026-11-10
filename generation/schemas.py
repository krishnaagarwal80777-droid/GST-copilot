"""Structured generator output: claims with citation ids, never free-text citations.

``kind`` places a claim in the answer template (Conclusion / Conditions / Exceptions).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ClaimKind = Literal["conclusion", "condition", "exception"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    citation_ids: list[str] = Field(default_factory=list)
    kind: ClaimKind = "conclusion"


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: list[Claim] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ref_date_used: str = ""
    abstain: bool = False


# Gemini responseSchema (OpenAPI subset) for GenerationOutput.
RESPONSE_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "citation_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "kind": {"type": "STRING", "enum": ["conclusion", "condition", "exception"]},
                },
                "required": ["text", "citation_ids", "kind"],
                "propertyOrdering": ["kind", "text", "citation_ids"],
            },
        },
        "assumptions": {"type": "ARRAY", "items": {"type": "STRING"}},
        "ref_date_used": {"type": "STRING"},
        "abstain": {"type": "BOOLEAN"},
    },
    "required": ["claims", "assumptions", "ref_date_used", "abstain"],
    "propertyOrdering": ["abstain", "ref_date_used", "claims", "assumptions"],
}

# provision_first prompt variant: the governing provision is identified before the claims.
RESPONSE_SCHEMA_PROVISION_FIRST: dict = {
    **RESPONSE_SCHEMA,
    "properties": {
        "governing": {
            "type": "OBJECT",
            "properties": {
                "citation_id": {"type": "STRING"},
                "in_force": {"type": "STRING"},
                "covers_ref_date": {"type": "BOOLEAN"},
            },
            "required": ["citation_id", "in_force", "covers_ref_date"],
            "propertyOrdering": ["citation_id", "in_force", "covers_ref_date"],
        },
        **RESPONSE_SCHEMA["properties"],
    },
    "required": ["governing", *RESPONSE_SCHEMA["required"]],
    "propertyOrdering": ["governing", *RESPONSE_SCHEMA["propertyOrdering"]],
}


def response_schema(variant: str = "baseline") -> dict:
    return RESPONSE_SCHEMA_PROVISION_FIRST if variant == "provision_first" else RESPONSE_SCHEMA

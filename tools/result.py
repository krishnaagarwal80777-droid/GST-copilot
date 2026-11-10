"""Common output of the calculator tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "ambiguous", "not_found", "refused"]


class ToolResult(BaseModel):
    """``result`` holds the numbers (as strings, exactly as they are to be quoted); ``steps``
    explain how they were obtained; ``citations`` are index row ids of the provisions the
    result rests on (only servable rows); ``notes`` state assumptions and missing citations.
    ``candidates`` lists every matching row when a lookup is ambiguous (no guess is made)."""

    tool: str
    status: ToolStatus
    result: dict[str, Any] | None = None
    steps: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)

    def text(self) -> str:
        """Plain-text rendering used as the tool's context block for the generator."""
        lines = [f"Tool {self.tool}: status {self.status}"]
        if self.result:
            lines += [f"{key}: {value}" for key, value in self.result.items()]
        if self.candidates:
            lines.append("Candidates (more than one row matches; none was chosen):")
            lines += ["- " + ", ".join(f"{k}: {v}" for k, v in c.items()) for c in self.candidates]
        lines += [f"Step: {step}" for step in self.steps]
        lines += [f"Note: {note}" for note in self.notes]
        return "\n".join(lines)

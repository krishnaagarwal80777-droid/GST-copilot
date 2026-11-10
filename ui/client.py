"""API calls and answer formatting for the Streamlit UI (no Streamlit imports, so testable)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

API_URL = os.environ.get("GST_API_URL", "http://localhost:8000")
# The first request loads the retrieval models; later ones take seconds.
TIMEOUT_S = 600.0
SECTIONS = (("conclusion", "Conclusion"), ("condition", "Conditions"), ("exception", "Exceptions"))


@dataclass(frozen=True)
class ApiResult:
    data: dict[str, Any] | None = None
    error: str | None = None


def ask(
    question: str,
    reference_date: date,
    knowledge_date: date | None = None,
    *,
    url: str = API_URL,
    transport: httpx.BaseTransport | None = None,
) -> ApiResult:
    body: dict[str, Any] = {"question": question, "reference_date": reference_date.isoformat()}
    if knowledge_date is not None:
        body["knowledge_date"] = knowledge_date.isoformat()
    try:
        with httpx.Client(timeout=TIMEOUT_S, transport=transport) as client:
            response = client.post(f"{url}/ask", json=body)
    except httpx.HTTPError as error:
        return ApiResult(error=f"Could not reach the API at {url} ({type(error).__name__}).")
    if response.status_code == 422:
        details = response.json().get("detail", [])
        messages = [d.get("msg", str(d)) for d in details] if isinstance(details, list) else []
        return ApiResult(error="Invalid input: " + ("; ".join(messages) or str(details)))
    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else ""
        return ApiResult(error=f"API error {response.status_code}: {detail}")
    return ApiResult(data=response.json())


def claims_markdown(response: dict[str, Any]) -> str:
    """Claims grouped as in the answer template, each followed by its citation numbers."""
    lines: list[str] = []
    for kind, title in SECTIONS:
        claims = [claim for claim in response["claims"] if claim["kind"] == kind]
        if not claims:
            continue
        lines.append(f"**{title}**")
        for claim in claims:
            marks = "".join(f" [{n}]" for n in claim["citation_numbers"])
            lines.append(f"- {claim['text']}{marks}")
        lines.append("")
    if response.get("assumptions"):
        lines.append("**Assumptions made**")
        lines.extend(f"- {text}" for text in response["assumptions"])
    return "\n".join(lines).strip()


def calculation_markdown(response: dict[str, Any]) -> str:
    """The calculator calls of a response: tool, arguments, result, steps and notes."""
    lines: list[str] = []
    for number, call in enumerate(response.get("tool_calls") or [], 1):
        args = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
        status = call["status"] if call["ok"] else f"failed: {call['error']}"
        lines.append(
            f"**{number}. `{call['name']}`({args})** - {status}, {call['latency_ms']:.0f} ms"
        )
        for key, value in (call.get("result") or {}).items():
            lines.append(f"- {key}: `{value}`")
        for candidate in call.get("candidates") or []:
            lines.append(
                f"- candidate: Notification {candidate['notification']} Schedule "
                f"{candidate['schedule']} S. No. {candidate['s_no']} ({candidate['description']}): "
                f"{candidate['rate']}"
            )
        lines += [f"  - step: {step}" for step in call.get("steps") or []]
        lines += [f"  - note: {note}" for note in call.get("notes") or []]
        lines.append("")
    return "\n".join(lines).strip()


def validity(citation: dict[str, Any]) -> str:
    if citation["effective_to"] is None:
        return f"in force from {citation['effective_from']} (still in force)"
    return f"in force {citation['effective_from']} to {citation['effective_to']}"

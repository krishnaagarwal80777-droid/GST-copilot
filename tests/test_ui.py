from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx

from ui.client import ask, citation_title, claims_markdown, validity

ROOT = Path(__file__).resolve().parents[1]


def transport(status: int, payload: dict, seen: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def test_ask_sends_iso_dates_and_omits_unset_knowledge_date() -> None:
    seen: list = []
    result = ask("q", date(2026, 1, 15), url="http://api", transport=transport(200, {"a": 1}, seen))
    assert result.data == {"a": 1} and result.error is None
    assert seen == [{"question": "q", "reference_date": "2026-01-15"}]
    ask(
        "q",
        date(2026, 1, 15),
        date(2025, 12, 31),
        url="http://api",
        transport=transport(200, {}, seen),
    )
    assert seen[1]["knowledge_date"] == "2025-12-31"


def test_ask_reports_errors() -> None:
    invalid = {"detail": [{"msg": "Value error, knowledge_date is in the future"}]}
    result = ask("q", date(2026, 1, 15), url="http://api", transport=transport(422, invalid))
    assert result.error == "Invalid input: Value error, knowledge_date is in the future"
    down = ask("q", date(2026, 1, 15), url="http://api", transport=transport(503, {"detail": "x"}))
    assert down.error == "API error 503: x"

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    unreachable = ask(
        "q", date(2026, 1, 15), url="http://api", transport=httpx.MockTransport(refuse)
    )
    assert unreachable.error.startswith("Could not reach the API at http://api")


def test_claims_markdown_and_validity() -> None:
    response = {
        "claims": [
            {"kind": "condition", "text": "B.", "citation_numbers": [2]},
            {"kind": "conclusion", "text": "A.", "citation_numbers": [1, 2]},
        ],
        "assumptions": ["Assumed."],
    }
    assert claims_markdown(response) == (
        "**Conclusion**\n- A. [1] [2]\n\n**Conditions**\n- B. [2]\n\n**Assumptions made**\n"
        "- Assumed."
    )
    assert validity({"effective_from": "2021-05-18", "effective_to": None}) == (
        "in force from 2021-05-18 (still in force)"
    )
    closed = {"effective_from": "2017-07-01", "effective_to": "2025-09-21"}
    assert validity(closed) == "in force 2017-07-01 to 2025-09-21"


def test_streamlit_app_renders_without_the_api(monkeypatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("observability.config.LOG_PATH", tmp_path / "none.jsonl")
    app = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Ask", "Observability"]
    assert app.button  # the form's Ask button


def test_citation_title_for_tools_and_provisions() -> None:
    tool = {"number": 2, "doc_type": "tool", "path": "TOOL/gst_split", "label": "Calculation"}
    assert citation_title(tool) == "[2] calculator output (gst_split)"
    rule = {
        "number": 1,
        "doc_type": "rule",
        "path": "CGST/r36/ss4",
        "label": "CGST Rules, rule 36(4)",
        "effective_from": "2021-05-18",
        "effective_to": None,
    }
    assert citation_title(rule) == (
        "[1] CGST Rules, rule 36(4) (in force from 2021-05-18 (still in force))"
    )

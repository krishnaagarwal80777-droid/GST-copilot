"""Calculator routing with a scripted generator (no LLM): the planner's function calls are
executed by Python, the answer cites TOOL- chunks, and numbers not in a tool output or cited
text abstain. The rate lookups read the real index rows (data/raw); retrieval is in memory."""

from __future__ import annotations

import json
from datetime import date

from fastapi.testclient import TestClient
from test_api import FakeRetriever

from api.app import create_app
from api.config import ApiSettings
from api.service import AskService
from generation.answer import Answerer
from generation.config import GenerationConfig
from generation.llm import LLMResponse, ToolTurn
from generation.tools_router import numbers, unsupported_numbers
from observability.log import RequestLogger

DAY = date(2025, 12, 1)  # before the rates horizon (2026-01-31)


class ScriptedGenerator:
    """complete_tools returns the scripted planner turns in order (then NO_TOOL); complete
    returns the scripted final answer."""

    model = "stub-model"

    def __init__(self, turns: list[list[dict]], final: dict | None = None) -> None:
        self.turns = list(turns)
        self.final = final
        self.planner_calls: list[list[dict]] = []
        self.final_prompts: list[tuple[str, str]] = []

    def complete_tools(self, system, contents, tools):
        self.planner_calls.append(contents)
        calls = self.turns.pop(0) if self.turns else []
        parts = [{"functionCall": call} for call in calls] or [{"text": "NO_TOOL"}]
        usage = {"promptTokenCount": 400, "candidatesTokenCount": 20}
        return ToolTurn(parts, "stub", self.model, False, usage, 0.01)

    def complete(self, system, prompt, *, schema=None):
        self.final_prompts.append((system, prompt))
        usage = {"promptTokenCount": 1500, "candidatesTokenCount": 90}
        return LLMResponse(json.dumps(self.final), "stub", self.model, False, usage, 0.01)


def claims(*items: tuple[str, list[str]]) -> dict:
    return {
        "abstain": False,
        "ref_date_used": DAY.isoformat(),
        "claims": [{"kind": "conclusion", "text": t, "citation_ids": ids} for t, ids in items],
        "assumptions": [],
    }


def answerer(generator, tools: bool = True, score: float = 0.9) -> Answerer:
    config = GenerationConfig(tools=tools, strict_claims=True)
    return Answerer(FakeRetriever(score), generator, config, horizons={})


LOOKUP = {"name": "rate_lookup", "args": {"query": "ferro-alloys"}}
SPLIT = {"name": "gst_split", "args": {"taxable_value": "50000", "rate": "18", "supply": "intra"}}
QUESTION = "What CGST and SGST apply to ferro-alloys worth Rs 50,000 sold within Maharashtra?"


def test_chained_lookup_and_split_answer_quotes_the_tool_numbers() -> None:
    final = claims(
        ("Ferro-alloys are at central tax 9 per cent (combined 18 per cent).", ["TOOL-1"]),
        ("CGST is 4500.00 and SGST is 4500.00, total tax 9000.00.", ["TOOL-2"]),
    )
    generator = ScriptedGenerator([[LOOKUP], [SPLIT]], final)
    result = answerer(generator).answer(QUESTION, DAY)
    assert not result.abstained, result.rendered
    assert [c["name"] for c in result.tool_calls] == ["rate_lookup", "gst_split"]
    lookup, split = (c["output"] for c in result.tool_calls)
    assert (lookup["result"]["rate"], lookup["result"]["combined_rate"]) == ("9", "18")
    assert lookup["citations"] == ["CTR/9-2025/schII@v1#261"]
    assert (split["result"]["cgst"], split["result"]["sgst"]) == ("4500.00", "4500.00")
    # The second planner turn saw the lookup result; the answer prompt has both tool chunks.
    assert "functionResponse" in json.dumps(generator.planner_calls[1])
    system, prompt = generator.final_prompts[0]
    assert "TOOL- are calculator outputs" in system
    assert "[ID: TOOL-1] Calculation: rate_lookup" in prompt and "cgst: 4500.00" in prompt
    assert "Calculation: gst_split" in result.rendered
    assert {"plan", "tools"} <= set(result.timings_ms) and len(result.plan_llm) == 2


def test_a_number_not_in_any_tool_output_abstains() -> None:
    final = claims(("CGST is 4600.00.", ["TOOL-2"]))
    result = answerer(ScriptedGenerator([[LOOKUP], [SPLIT]], final)).answer(QUESTION, DAY)
    assert result.abstained and result.abstain_reason == "unsupported_number"
    assert "4600.00" in result.rendered


def test_ambiguous_lookup_abstains_with_candidates_and_no_answer_call() -> None:
    generator = ScriptedGenerator([[{"name": "rate_lookup", "args": {"hsn": "7317"}}]])
    result = answerer(generator).answer("What is the rate on HSN 7317?", DAY)
    assert result.abstained and result.abstain_reason == "ambiguous_lookup"
    assert "S. No. 412" in result.rendered and "S. No. 295" in result.rendered
    assert generator.final_prompts == []


def test_lookup_after_the_rates_horizon_is_refused() -> None:
    generator = ScriptedGenerator([[LOOKUP]])
    result = answerer(generator).answer("Rate on ferro-alloys?", date(2026, 3, 1))
    assert result.abstained and result.abstain_reason == "horizon"
    assert "after 2026-01-31" in result.rendered


def test_invalid_arguments_are_a_failed_call() -> None:
    bad = {"name": "gst_split", "args": {"taxable_value": "100", "rate": "18", "supply": "export"}}
    result = answerer(ScriptedGenerator([[bad]])).answer("Tax on 100 at 18%?", DAY)
    assert result.abstained and result.abstain_reason == "tool_error"
    assert result.tool_calls[0]["ok"] is False


def test_tool_answer_skips_the_confidence_gate() -> None:
    # A pure calculation retrieves nothing relevant (score 0.1 < threshold), yet the tool
    # output grounds the answer; the retrieved rows are left out of the answer context.
    final = claims(("IGST is 1800.00.", ["TOOL-1"]))
    split = {
        "name": "gst_split",
        "args": {"taxable_value": "10000", "rate": "18", "supply": "inter"},
    }
    generator = ScriptedGenerator([[split]], final)
    result = answerer(generator, score=0.1).answer("IGST on 10,000 at 18%?", DAY)
    assert not result.abstained
    assert "placeholder" not in generator.final_prompts[0][1]


def test_tools_off_never_calls_the_planner_and_keeps_records_unchanged() -> None:
    generator = ScriptedGenerator([[LOOKUP]], claims(("A claim.", ["CGST/r36/ss4@v5"])))
    result = answerer(generator, tools=False).answer("When is rule 36(4) applicable?", DAY)
    assert generator.planner_calls == [] and not result.abstained
    record = result.to_dict()
    assert "tool_calls" not in record and "plan_llm" not in record
    assert "plan" not in result.timings_ms


def test_number_check() -> None:
    assert numbers("CGST ₹4,500.00 at 9% under S. No. 261 of 9/2025") == [
        "4,500.00",
        "9",
        "261",
        "9",
        "2025",
    ]
    assert unsupported_numbers(["CGST is 4,500.00 (9%)."], ["cgst: 4500.00", "rate: 9"]) == []
    assert unsupported_numbers(["Total 9,001"], ["total: 9000.00"]) == ["9,001"]


def test_api_returns_the_calculation_and_logs_tool_calls(tmp_path) -> None:
    final = claims(("CGST is 4500.00 and SGST is 4500.00.", ["TOOL-2"]))
    generator = ScriptedGenerator([[LOOKUP], [SPLIT]], final)
    settings = ApiSettings()
    log = RequestLogger(tmp_path / "requests.jsonl")
    service = AskService(answerer(generator), settings, log)
    client = TestClient(create_app(service))
    body = {"question": QUESTION, "reference_date": DAY.isoformat()}
    data = client.post("/ask", json=body).json()
    assert not data["abstained"], data["answer"]
    names = [call["name"] for call in data["tool_calls"]]
    assert names == ["rate_lookup", "gst_split"]
    split = data["tool_calls"][1]
    assert split["result"]["total"] == "59000.00" and split["steps"] and split["ok"]
    assert any(c["label"] == "Calculation: gst_split" for c in data["citations"])
    line = json.loads((tmp_path / "requests.jsonl").read_text(encoding="utf-8"))
    assert [(t["name"], t["success"]) for t in line["tool_calls"]] == [
        ("rate_lookup", True),
        ("gst_split", True),
    ]
    assert all(t["latency_ms"] >= 0 for t in line["tool_calls"])
    assert line["planner_calls"] == 2 and "50000" not in json.dumps(line)

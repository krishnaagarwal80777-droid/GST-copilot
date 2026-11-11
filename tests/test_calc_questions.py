"""eval/questions_calc.jsonl: the hand-computed expected results agree with the tools when
the expected calls are run (rate lookups read the real index rows from data/raw)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from tools.gst_split import gst_split
from tools.rate_lookup import rate_lookup

ITEMS = [
    json.loads(line)
    for line in (Path(__file__).resolve().parents[1] / "eval" / "questions_calc.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


def test_fifteen_items_with_unique_ids() -> None:
    assert len(ITEMS) == 15 and len({item["id"] for item in ITEMS}) == 15


@pytest.mark.parametrize("item", ITEMS, ids=[item["id"] for item in ITEMS])
def test_expected_results_match_the_tools(item) -> None:
    on = date.fromisoformat(item["reference_date"])
    expected = item["expected"]
    outputs = []
    for call in expected["tools"]:
        args = call["arguments"]
        if call["name"] == "gst_split":
            outputs.append(
                gst_split(
                    args["taxable_value"], args["rate"], args["supply"], inclusive=args["inclusive"]
                )
            )
        else:
            outputs.append(rate_lookup(args.get("query"), hsn=args.get("hsn"), on=on))
    if not outputs:
        assert expected["result"] is None
        return
    last = outputs[-1]
    if expected["outcome"] == "abstain":
        if expected["abstain_reason"] == "horizon":
            assert last.status == "refused"
        else:
            assert last.status == "ambiguous"
            got = sorted((c["schedule"], c["s_no"], c["rate"]) for c in last.candidates)
            want = sorted(
                (c["schedule"], c["s_no"], c["rate"]) for c in expected["result"]["candidates"]
            )
            assert got == want
        return
    assert last.status == "ok"
    assert {k: last.result[k] for k in expected["result"]} == expected["result"]
    # A chained question passes the lookup's combined rate to gst_split unchanged.
    if len(outputs) == 2:
        assert outputs[0].result["combined_rate"] == expected["tools"][1]["arguments"]["rate"]
    for number in expected["numbers"]:
        assert number in last.result.values() or number == last.result.get("rate")


def test_requested_fields_name_expected_result_fields() -> None:
    for item in ITEMS:
        fields = item["requested_fields"]
        if item["expected"]["outcome"] == "abstain" or not item["expected"]["tools"]:
            assert fields == [], item["id"]
        else:
            assert fields and set(fields) <= set(item["expected"]["result"]), item["id"]

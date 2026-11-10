"""Calculation eval: eval/questions_calc.jsonl through the API path (POST /ask on the FastAPI
app with the API settings: guardrails, strict_claims and calculator routing on).

A small set (15 questions written for this, separate from the main eval and its split) that
checks the routing, not the law:
- tool selection: the tools called, in order, equal the expected ones (no tool for calc-15);
- arguments: each expected call was made with equal arguments (numbers compared as Decimal,
  descriptions as content words, HSN as digits), over items that expect a tool;
- final numbers: every expected number appears (as a value) in the answer's claims, over items
  that expect an answer with numbers;
- outcome: answered or abstained as expected (with the expected abstention reason).

    python -m eval.calc   ->  eval/results/calc_items.jsonl, eval/results/calc.csv (appended)
"""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from eval.retrieval import git_hash
from generation.tools_router import numbers
from tools.rate_lookup import words

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "questions_calc.jsonl"
ITEMS_OUT = ROOT / "eval" / "results" / "calc_items.jsonl"
SUMMARY_OUT = ROOT / "eval" / "results" / "calc.csv"


def load(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dec(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "").strip()).normalize()
    except InvalidOperation:
        return None


def same_arguments(name: str, got: dict[str, Any], want: dict[str, Any]) -> bool:
    if name == "gst_split":
        return (
            _dec(got.get("taxable_value")) == _dec(want["taxable_value"])
            and _dec(got.get("rate")) == _dec(want["rate"])
            and got.get("supply") == want["supply"]
            and bool(got.get("inclusive", False)) == bool(want.get("inclusive", False))
        )
    if "hsn" in want:
        return re.sub(r"\D", "", str(got.get("hsn") or "")) == want["hsn"] and not got.get("query")
    return (
        bool(got.get("query"))
        and words(got["query"]) == words(want["query"])
        and not got.get("hsn")
    )


def score(item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected = item["expected"]
    calls = response.get("tool_calls") or []
    names = [call["name"] for call in calls]
    want_names = [call["name"] for call in expected["tools"]]
    selection = names == want_names
    arguments = None
    if expected["tools"]:
        arguments = selection and all(
            same_arguments(w["name"], c["arguments"], w["arguments"])
            for c, w in zip(calls, expected["tools"], strict=True)
        )
    if expected["outcome"] == "abstain":
        outcome = response["abstained"] and response["abstain_reason"] == expected["abstain_reason"]
    else:
        outcome = not response["abstained"]
    final_numbers = None
    if expected["outcome"] == "answer" and expected["numbers"]:
        written = {_dec(n) for claim in response["claims"] for n in numbers(claim["text"])}
        final_numbers = not response["abstained"] and all(
            _dec(n) in written for n in expected["numbers"]
        )
    return {
        "tool_selection": selection,
        "arguments": arguments,
        "final_numbers": final_numbers,
        "outcome": outcome,
    }


def rate(values: list[bool | None]) -> tuple[int, int]:
    scored = [v for v in values if v is not None]
    return sum(scored), len(scored)


def main() -> None:
    from fastapi.testclient import TestClient

    from api.app import create_app, default_answerer
    from api.config import ApiSettings
    from api.service import AskService
    from observability.log import RequestLogger

    settings = ApiSettings()
    service = AskService(lambda: default_answerer(settings), settings, RequestLogger())
    client = TestClient(create_app(service))
    items = load()
    records = []
    for item in items:
        body = {"question": item["question"], "reference_date": item["reference_date"]}
        response = client.post("/ask", json=body)
        response.raise_for_status()
        data = response.json()
        scores = score(item, data)
        records.append({"id": item["id"], **scores, "response": data})
        print(
            f"{item['id']}: tools {[c['name'] for c in data['tool_calls']]} "
            f"abstained={data['abstained']}({data['abstain_reason']}) {scores}"
        )
    with ITEMS_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "git_hash": git_hash()}
    summary["n"] = len(records)
    for key in ("tool_selection", "arguments", "final_numbers", "outcome"):
        hits, total = rate([r[key] for r in records])
        summary[key] = f"{hits}/{total}"
        summary[f"{key}_rate"] = round(hits / total, 3) if total else ""
    new = not SUMMARY_OUT.exists()
    with SUMMARY_OUT.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        if new:
            writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

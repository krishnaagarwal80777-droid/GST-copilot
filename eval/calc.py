"""Calculation eval: eval/questions_calc.jsonl through the API path (POST /ask on the FastAPI
app with the API settings: guardrails, strict_claims and calculator routing on).

A small set (15 questions written for this, separate from the main eval and its split) that
checks the routing, not the law:
- tool selection: the tools called, in order, equal the expected ones (no tool for calc-15);
- arguments: each expected call was made with equal arguments (numbers compared as Decimal,
  descriptions as content words, HSN as digits), over items that expect a tool;
- final numbers (strict): every number in the item's ``expected.numbers`` appears (as a
  value) in the answer's claims, over items that expect an answer with numbers;
- requested amounts: every field the question asks for (``requested_fields``, read from its
  wording; values from ``expected.result``) appears in the answer's claims, over items that
  request at least one field. The column was added after the first run;
- outcome: answered or abstained as expected (with the expected abstention reason).

    python -m eval.calc [--tag v2]   ->  eval/results/calc_items[_<tag>].jsonl, calc.csv
    python -m eval.calc --rescore eval/results/calc_items.jsonl --note "..."   (no LLM calls)
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
METRICS = ("tool_selection", "arguments", "final_numbers", "requested_amounts", "outcome")


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
    written = {_dec(n) for claim in response["claims"] for n in numbers(claim["text"])}
    final_numbers = None
    if expected["outcome"] == "answer" and expected["numbers"]:
        final_numbers = not response["abstained"] and all(
            _dec(n) in written for n in expected["numbers"]
        )
    requested = None
    if expected["outcome"] == "answer" and item.get("requested_fields"):
        wanted = [expected["result"][field] for field in item["requested_fields"]]
        requested = not response["abstained"] and all(_dec(v) in written for v in wanted)
    return {
        "tool_selection": selection,
        "arguments": arguments,
        "final_numbers": final_numbers,
        "requested_amounts": requested,
        "outcome": outcome,
    }


def rate(values: list[bool | None]) -> tuple[int, int]:
    scored = [v for v in values if v is not None]
    return sum(scored), len(scored)


def summarize(records: list[dict[str, Any]], note: str) -> dict[str, Any]:
    summary = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "git_hash": git_hash()}
    summary["n"] = len(records)
    for key in METRICS:
        hits, total = rate([r[key] for r in records])
        summary[key] = f"{hits}/{total}"
        summary[f"{key}_rate"] = round(hits / total, 3) if total else ""
    summary["note"] = note
    return summary


def append_summary(summary: dict[str, Any], path: Path = SUMMARY_OUT) -> None:
    """Append a row; when columns were added since the file was written, earlier rows keep
    their values and the new columns are empty on them."""
    fields = list(summary)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(dict.fromkeys([*(reader.fieldnames or []), *summary]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*rows, summary])


def rescore(items_file: Path, note: str) -> None:
    """Score recorded responses again with the current questions file (no LLM calls)."""
    items = {item["id"]: item for item in load()}
    records = []
    for line in items_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records.append(
            {
                "id": record["id"],
                **score(items[record["id"]], record["response"]),
                "response": record["response"],
            }
        )
    for record in records:
        print(record["id"], {k: record[k] for k in METRICS})
    summary = summarize(records, note)
    append_summary(summary)
    print(json.dumps(summary, indent=1))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calculation eval through the API path.")
    parser.add_argument("--tag", default="", help="run label: calc_items_<tag>.jsonl")
    parser.add_argument("--note", default="", help="note stored on the summary row")
    parser.add_argument("--rescore", type=Path, help="score a recorded items file again")
    args = parser.parse_args()
    if args.rescore:
        rescore(args.rescore, args.note)
        return

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
    items_out = ITEMS_OUT.with_name(f"calc_items_{args.tag}.jsonl") if args.tag else ITEMS_OUT
    with items_out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, args.note)
    append_summary(summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

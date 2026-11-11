"""Calculator routing on the dev rate questions (generator calls only, no judge).

The 24 dev numeric items (rate lookups) go through the API service with the API settings
(input guardrails, strict_claims, calculator routing on). For each item: did the planner call
rate_lookup, did the lookup return a single row, and does the answer state the gold central
tax rate (the leading "<rate> per cent" of the gold answer, compared as a value with the
numbers in the answer's claims)? The frozen dev v4 run (tools off, judge v2) had all 24
correct; an item whose answer abstains or no longer states the gold rate would differ.

    python -m eval.tool_routing   ->  eval/results/tool_routing_dev.jsonl and .csv (appended)
"""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from eval.retrieval import git_hash
from eval.split import item_key, items_in
from generation.tools_router import numbers

ROOT = Path(__file__).resolve().parents[1]
ITEMS_OUT = ROOT / "eval" / "results" / "tool_routing_dev.jsonl"
SUMMARY_OUT = ROOT / "eval" / "results" / "tool_routing_dev.csv"
V4 = ROOT / "eval" / "results" / "generation_dev_v4_judge-v2_items.jsonl"
GOLD_RATE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+per\s+cent")


def gold_rate(answer: str) -> Decimal | None:
    match = GOLD_RATE.match(answer)
    return Decimal(match.group(1)).normalize() if match else None


def main() -> None:
    from api.app import default_answerer
    from api.config import ApiSettings
    from api.schemas import AskRequest
    from api.service import AskService
    from observability.log import RequestLogger

    settings = ApiSettings()
    service = AskService(lambda: default_answerer(settings), settings, RequestLogger())
    v4 = {r["key"]: r for r in map(json.loads, V4.read_text(encoding="utf-8").splitlines()[1:])}
    items = [i for i in items_in("dev") if i.category == "numeric"]
    records = []
    for item in items:
        key = item_key(item)
        data = service.ask(AskRequest(question=item.question, reference_date=item.ref_date))
        data = data.model_dump(mode="json")
        lookups = [c for c in data["tool_calls"] if c["name"] == "rate_lookup"]
        gold = gold_rate(item.gold_answer)
        written = {
            Decimal(n.replace(",", "")).normalize()
            for c in data["claims"]
            for n in numbers(c["text"])
        }
        found = lookups[-1]["result"]["rate"] if lookups and lookups[-1]["result"] else None
        record = {
            "key": key,
            "routed_to_rate_lookup": bool(lookups),
            "lookup_status": lookups[-1]["status"] if lookups else None,
            "lookup_rate": found,
            "gold_rate": str(gold),
            "lookup_rate_is_gold": found is not None and Decimal(found).normalize() == gold,
            "answer_states_gold_rate": not data["abstained"] and gold in written,
            "abstained": data["abstained"],
            "abstain_reason": data["abstain_reason"],
            "v4_correct": v4[key]["score"]["correct"],
            "response": data,
        }
        record["differs_from_v4"] = record["v4_correct"] != record["answer_states_gold_rate"]
        records.append(record)
        print(
            f"{key}: routed={record['routed_to_rate_lookup']} status={record['lookup_status']} "
            f"rate={found} gold={gold} answer_has_gold={record['answer_states_gold_rate']} "
            f"abstained={data['abstained']}({data['abstain_reason']})"
        )
    with ITEMS_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    routed = [r for r in records if r["routed_to_rate_lookup"]]
    summary = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "n": len(records),
        "routed_to_rate_lookup": len(routed),
        "lookup_single_row": sum(r["lookup_status"] == "ok" for r in routed),
        "lookup_rate_is_gold": sum(r["lookup_rate_is_gold"] for r in routed),
        "answer_states_gold_rate_routed": sum(r["answer_states_gold_rate"] for r in routed),
        "answer_states_gold_rate_all": sum(r["answer_states_gold_rate"] for r in records),
        "differs_from_v4": sum(r["differs_from_v4"] for r in records),
    }
    new = not SUMMARY_OUT.exists()
    with SUMMARY_OUT.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        if new:
            writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

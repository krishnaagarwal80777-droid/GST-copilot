"""Judge audit sheet for the dev split: items for a human to re-grade against the LLM judge.

Rows: every dev item the judge marked incorrect or partially_correct in the given run, plus
a fixed-seed random sample of the items it marked correct. Items that were not judged
(abstentions, and unanswerable items that were answered) are not included. The
``human_verdict`` column is left empty for the reviewer. Only the dev split can be audited:
reading test answers item by item would turn the test split into a second dev split.

    python -m eval.judge_audit --tag v3   ->  eval/results/judge_audit_dev.csv
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from eval.generation import ROOT, items_path
from eval.split import item_key, items_in

SPLIT = "dev"
OUT = ROOT / "eval" / "results" / f"judge_audit_{SPLIT}.csv"
SEED = 20261110
CORRECT_SAMPLE = 10
FIELDS = (
    "item_key",
    "question",
    "gold_answer",
    "generated_answer",
    "citations",
    "judge_verdict",
    "judge_reason",
    "human_verdict",
)
ORDER = {"incorrect": 0, "partially_correct": 1, "correct": 2}


def audit_rows(tag: str, *, seed: int = SEED, sample: int = CORRECT_SAMPLE) -> list[dict[str, str]]:
    lines = items_path(SPLIT, tag).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines[1:]]
    judged = [r for r in records if (r.get("judge") or {}).get("verdict") in ORDER]
    flagged = [r for r in judged if r["judge"]["verdict"] != "correct"]
    correct = sorted((r for r in judged if r["judge"]["verdict"] == "correct"), key=_key)
    picked = random.Random(seed).sample(correct, min(sample, len(correct)))
    gold = {item_key(item): item.gold_answer for item in items_in(SPLIT)}
    chosen = sorted(flagged + picked, key=lambda r: (ORDER[r["judge"]["verdict"]], r["key"]))
    return [
        {
            "item_key": record["key"],
            "question": record["question"],
            "gold_answer": gold[record["key"]],
            "generated_answer": record["rendered"],
            "citations": "; ".join(record["cited_paths"]),
            "judge_verdict": record["judge"]["verdict"],
            "judge_reason": record["judge"].get("rationale", ""),
            "human_verdict": "",
        }
        for record in chosen
    ]


def _key(record: dict) -> str:
    return record["key"]


def write(rows: list[dict[str, str]], path: Path = OUT) -> None:
    # utf-8-sig so Excel opens the file with the right encoding.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Write the dev judge audit sheet.")
    parser.add_argument("--tag", default="", help='generation run to audit, e.g. "v3"')
    args = parser.parse_args()
    rows = audit_rows(args.tag)
    write(rows)
    counts = dict(Counter(row["judge_verdict"] for row in rows))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows {counts} (seed {SEED})")


if __name__ == "__main__":
    main()

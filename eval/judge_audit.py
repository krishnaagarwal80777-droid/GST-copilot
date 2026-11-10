"""Judge audit sheet for the dev split: items for a human to re-grade against the LLM judge.

Rows: every dev item the judge marked incorrect or partially_correct in the given run, plus
a fixed-seed random sample of the items it marked correct. Items that were not judged
(abstentions, and unanswerable items that were answered) are not included. The
``human_verdict`` column is left empty for the reviewer. Only the dev split can be audited:
reading test answers item by item would turn the test split into a second dev split.

    python -m eval.judge_audit --tag v3   ->  eval/results/judge_audit_dev.csv

Once ``human_verdict`` is filled in the sheet is never rewritten. ``--agreement v2`` re-grades
the sheet's answers with another judge version and reports agreement and Cohen's kappa
against the human verdicts, beside the judge that built the sheet; the new verdicts go to
``judge_audit_dev_judge-<version>.csv``.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from eval.generation import ROOT, VERDICTS, items_path
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


def read(path: Path = OUT) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(rows: list[dict[str, str]], path: Path = OUT) -> None:
    if path.exists() and any(row.get("human_verdict") for row in read(path)):
        raise SystemExit(f"{path.name} has human verdicts: not overwriting it")
    # utf-8-sig so Excel opens the file with the right encoding.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def cohen_kappa(a: list[str], b: list[str], labels: tuple[str, ...] = VERDICTS) -> float:
    """Unweighted Cohen's kappa of two label sequences."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label] for label in labels) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def agreement(version: str, path: Path = OUT) -> list[dict[str, str]]:
    """Re-grade the audit sheet's answers with judge ``version``; the sheet is only read."""
    from eval.generation import judge
    from generation.llm import judge_client

    rows = read(path)
    items = {item_key(item): item for item in items_in(SPLIT)}
    client = judge_client()
    out = []
    for row in rows:
        verdict = judge(client, items[row["item_key"]], row["generated_answer"], version)
        out.append(
            {
                "item_key": row["item_key"],
                "human_verdict": row["human_verdict"],
                "judge_v1": row["judge_verdict"],
                f"judge_{version}": verdict["verdict"],
                f"judge_{version}_reason": verdict.get("rationale", ""),
            }
        )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write or score the dev judge audit sheet.")
    parser.add_argument("--tag", default="", help='generation run to audit, e.g. "v3"')
    parser.add_argument("--agreement", metavar="VERSION", help='re-grade with a judge, e.g. "v2"')
    args = parser.parse_args()
    if args.agreement:
        version = args.agreement
        rows = agreement(version)
        target = OUT.with_name(f"judge_audit_{SPLIT}_judge-{version}.csv")
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        human = [row["human_verdict"] for row in rows]
        for name in ("judge_v1", f"judge_{version}"):
            labels = [row[name] for row in rows]
            agree = sum(h == x for h, x in zip(human, labels, strict=True))
            print(
                f"{name}: agreement {agree}/{len(rows)}, "
                f"kappa {cohen_kappa(human, labels):.3f}, verdicts {dict(Counter(labels))}"
            )
        print(f"wrote {target.relative_to(ROOT)}")
        return
    rows = audit_rows(args.tag)
    write(rows)
    counts = dict(Counter(row["judge_verdict"] for row in rows))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows {counts} (seed {SEED})")


if __name__ == "__main__":
    main()

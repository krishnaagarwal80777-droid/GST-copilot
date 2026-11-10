"""Input-guardrail eval: refusal precision and recall on eval/guardrail_questions.jsonl.

30 questions written for this set (10 out of scope, 10 prompt injection, 10 GST), none taken
from the eval questions. No LLM call is made. Two configurations:

- keywords: length, injection patterns and keyword scope only (no retrieval);
- full: keywords, then the retrieval-confidence scope check for questions no keyword decides
  (needs the retriever: pgvector and the local models).

A refusal is any guardrail firing. Refusal precision = refused questions that should be refused
/ refused; recall = refused / should be refused (out of scope + injection). "Right type" counts a
refusal whose guardrail matches the expected one. Results go to eval/results/guardrails.csv
(one row per configuration) and guardrails_items.csv (per question).

The scope threshold equals the confidence-gate threshold, so a GST question refused as out of
scope on confidence would otherwise have been a gate abstention: a false refusal changes the
message, not whether an answer is given.

The keyword and injection patterns were written before this set and were not changed after
scoring it.

    python -m eval.guardrails [--keywords-only]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from generation.guardrails import (
    GuardrailConfig,
    check_after_retrieval,
    check_before_retrieval,
    keyword_scope,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "guardrail_questions.jsonl"
ITEMS_OUT = ROOT / "eval" / "results" / "guardrails_items.csv"
SUMMARY_OUT = ROOT / "eval" / "results" / "guardrails.csv"


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    reference_date: date
    expected: str  # "out_of_scope" | "injection" | "none"


def load_items(path: Path = QUESTIONS) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(
                Item(
                    row["id"],
                    row["question"],
                    date.fromisoformat(row["reference_date"]),
                    row["expected"],
                )
            )
    return items


def scores(expected: list[str], predicted: list[str | None]) -> dict[str, float | int]:
    should = [e != "none" for e in expected]
    refused = [p is not None for p in predicted]
    tp = sum(s and r for s, r in zip(should, refused, strict=True))
    out: dict[str, float | int] = {
        "refused": sum(refused),
        "should_refuse": sum(should),
        "true_refusals": tp,
        "precision": round(tp / sum(refused), 3) if any(refused) else float("nan"),
        "recall": round(tp / sum(should), 3) if any(should) else float("nan"),
        "right_type": sum(e == p for e, p in zip(expected, predicted, strict=True) if p),
    }
    for kind in ("out_of_scope", "injection"):
        rows = [(e, p) for e, p in zip(expected, predicted, strict=True) if e == kind]
        out[f"recall_{kind}"] = round(sum(p == kind for _, p in rows) / len(rows), 3)
        fired = [(e, p) for e, p in zip(expected, predicted, strict=True) if p == kind]
        out[f"precision_{kind}"] = (
            round(sum(e == kind for e, _ in fired) / len(fired), 3) if fired else float("nan")
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keywords-only", action="store_true", help="skip the retrieval check")
    args = parser.parse_args()

    config = GuardrailConfig()
    items = load_items()
    answerer = None
    if not args.keywords_only:
        from api.app import default_answerer
        from api.config import ApiSettings

        answerer = default_answerer(ApiSettings())
        print(f"store: {answerer.retriever.store.backend}")

    rows = []
    for item in items:
        keyword = check_before_retrieval(item.question, config)
        full, confidence = keyword, None
        if keyword is None and answerer is not None and keyword_scope(item.question) == "unknown":
            context, _ = answerer.retrieve(item.question, item.reference_date)
            confidence = round(context.confidence, 4)
            full = check_after_retrieval(item.question, context.confidence, config)
        rows.append(
            {
                "id": item.id,
                "expected": item.expected,
                "keyword_scope": keyword_scope(item.question),
                "keywords": keyword or "",
                "full": (full or "") if answerer is not None else "",
                "confidence": "" if confidence is None else confidence,
                "question": item.question,
            }
        )

    ITEMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ITEMS_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    expected = [row["expected"] for row in rows]
    configs = ["keywords"] + (["full"] if answerer is not None else [])
    print(f"{len(items)} questions; scope threshold {config.scope_threshold}")
    summary = []
    for name in configs:
        predicted = [row[name] or None for row in rows]
        result = scores(expected, predicted)
        summary.append({"config": name, "questions": len(items), **result})
        print(f"\n[{name}]")
        for key, value in result.items():
            print(f"  {key:24} {value}")
        misses = [
            f"{row['id']} expected {row['expected']}, got {row[name] or 'answered'}"
            for row in rows
            if (row[name] or "none") != row["expected"]
        ]
        print("  errors: " + ("; ".join(misses) if misses else "none"))
    with SUMMARY_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n{SUMMARY_OUT.relative_to(ROOT)}, {ITEMS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

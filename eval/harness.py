from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gst_copilot.models import EvalItem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_FILE = ROOT / "eval" / "questions.jsonl"
DEFAULT_RESULTS_FILE = ROOT / "eval" / "results.jsonl"
# Items whose gold answer depends on Act text newer than the latest consolidated Act source
# (cgst-act_cbic-taxinfo_2022-09-28.pdf). They stay in the file but are never scored.
PENDING_SOURCE_GROUP = "pending_source"


def load_eval_items(path: str | Path = DEFAULT_QUESTION_FILE) -> list[EvalItem]:
    """Load a JSONL evaluation dataset into Pydantic EvalItem objects."""
    dataset = Path(path)
    items: list[EvalItem] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid JSON on line {line_number}: {raw_line!r}") from exc
            items.append(EvalItem.model_validate(payload))
    return items


def scored_items(items: list[EvalItem]) -> list[EvalItem]:
    """Items that count towards metrics (everything except the pending_source group)."""
    return [item for item in items if item.group_id != PENDING_SOURCE_GROUP]


def summarize_eval_items(items: list[EvalItem]) -> dict[str, Any]:
    scored = scored_items(items)
    by_category = Counter(item.category for item in scored)
    return {
        "num_items": len(items),
        "num_scored": len(scored),
        "pending_source": len(items) - len(scored),
        "answerable": sum(1 for item in scored if item.answerable),
        "unanswerable": sum(1 for item in scored if not item.answerable),
        "by_category": dict(sorted(by_category.items())),
    }


def run_eval(
    question_file: str | Path = DEFAULT_QUESTION_FILE,
    results_file: str | Path = DEFAULT_RESULTS_FILE,
    *,
    config_name: str = "baseline",
) -> dict[str, Any]:
    """Run a baseline evaluation over a dataset and persist a JSONL summary row."""
    dataset = Path(question_file)
    items = load_eval_items(dataset)
    summary = summarize_eval_items(items)
    record = {
        "config": config_name,
        "dataset": str(dataset),
        "run_at": datetime.now(UTC).isoformat(),
        **summary,
    }

    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    return record


def main() -> None:
    result = run_eval()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

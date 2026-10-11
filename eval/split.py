"""Deterministic dev/test split of the scored eval items (eval/split.json).

60/40 dev/test, stratified by category; items sharing a group_id (date pairs, rate pairs) stay
in the same split. Groups are shuffled with a fixed seed and assigned to test until the
category's test share reaches 40% of its items. pending_source items are never scored, so they
are left out. Tune only on dev; the test split is for the final numbers.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from eval.harness import load_eval_items, scored_items

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "eval" / "split.json"
SEED = 20261011
TEST_SHARE = 0.4


def item_key(item) -> str:
    """Stable id for an eval item: group plus reference date (unique within the set)."""
    return f"{item.group_id}@{item.ref_date.isoformat()}"


def make_split(seed: int = SEED, test_share: float = TEST_SHARE) -> dict:
    items = scored_items(load_eval_items())
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups[item.category][item.group_id].append(item)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for category in sorted(groups):
        group_ids = sorted(groups[category])
        rng.shuffle(group_ids)
        total = sum(len(groups[category][gid]) for gid in group_ids)
        in_test = 0
        for gid in group_ids:
            size = len(groups[category][gid])
            # Add to test while that keeps the test share closest to the target.
            if abs(in_test + size - test_share * total) < abs(in_test - test_share * total):
                split, in_test = "test", in_test + size
            else:
                split = "dev"
            for item in groups[category][gid]:
                assignment[item_key(item)] = split
    return {
        "seed": seed,
        "test_share": test_share,
        "note": "group_id-preserving, stratified by category; pending_source items excluded",
        "items": dict(sorted(assignment.items())),
    }


def load_split(path: Path = SPLIT_FILE) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def items_in(split: str, path: Path = SPLIT_FILE) -> list:
    assignment = load_split(path)
    return [item for item in scored_items(load_eval_items()) if assignment[item_key(item)] == split]


def main() -> None:
    split = make_split()
    SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    by_key = {item_key(item): item for item in scored_items(load_eval_items())}
    for key, part in split["items"].items():
        counts[(by_key[key].category, part)] += 1
    for category in sorted({category for category, _ in counts}):
        print(
            f"{category:15} dev={counts[(category, 'dev')]:2} test={counts[(category, 'test')]:2}"
        )


if __name__ == "__main__":
    main()

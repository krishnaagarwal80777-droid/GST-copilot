"""Deterministic dev/test split of the scored eval items (eval/split.json).

60/40 dev/test, stratified by source and category; items sharing a group_id (date pairs, rate
pairs) stay in the same split. Groups are shuffled with a fixed seed and assigned to test until
the stratum's test share reaches 40% of its items. pending_source items are never scored, so
they are left out. Tune only on dev; the test split is for the final numbers.

Hand-written strata are processed first, with the same random stream as before templated items
existed, so adding templated items never moves a hand-written item between splits.

Items already in eval/split.json keep their split (``frozen``): when new items become scored
(e.g. pending_source items once a newer Act source arrived), only the new groups are assigned,
greedily toward the 40% target of their stratum, in an order shuffled with the same seed.
Nothing already assigned moves, so dev tuning and test numbers stay comparable.
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


def source_kind(item) -> str:
    """'templated' for generated items, 'hand' for everything written by hand."""
    return "templated" if item.source == "templated" else "hand"


def make_split(
    seed: int = SEED, test_share: float = TEST_SHARE, frozen: dict[str, str] | None = None
) -> dict:
    if frozen:
        return _extend_split(frozen, seed, test_share)
    items = scored_items(load_eval_items())
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups[(source_kind(item), item.category)][item.group_id].append(item)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    # "hand" sorts before "templated": hand-written strata first, in category order, as before.
    for stratum in sorted(groups):
        group_ids = sorted(groups[stratum])
        rng.shuffle(group_ids)
        total = sum(len(groups[stratum][gid]) for gid in group_ids)
        in_test = 0
        for gid in group_ids:
            size = len(groups[stratum][gid])
            # Add to test while that keeps the test share closest to the target.
            if abs(in_test + size - test_share * total) < abs(in_test - test_share * total):
                split, in_test = "test", in_test + size
            else:
                split = "dev"
            for item in groups[stratum][gid]:
                assignment[item_key(item)] = split
    return {
        "seed": seed,
        "test_share": test_share,
        "note": (
            "group_id-preserving, stratified by source and category; pending_source items excluded"
        ),
        "items": dict(sorted(assignment.items())),
    }


def _extend_split(frozen: dict[str, str], seed: int, test_share: float) -> dict:
    """Keep every frozen assignment; assign only groups with no assigned item."""
    items = scored_items(load_eval_items())
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        groups[(source_kind(item), item.category)][item.group_id].append(item)
    assignment = {key: part for key, part in frozen.items()}
    live = {item_key(item) for item in items}
    gone = set(assignment) - live
    if gone:
        raise ValueError(f"split.json lists items that are no longer scored: {sorted(gone)}")
    rng = random.Random(seed)
    for stratum in sorted(groups):
        members = groups[stratum]
        total = sum(len(group) for group in members.values())
        in_test = sum(
            1 for group in members.values() for i in group if assignment.get(item_key(i)) == "test"
        )
        new_ids = sorted(
            gid
            for gid, group in members.items()
            if not any(item_key(i) in assignment for i in group)
        )
        partial = [
            gid
            for gid, group in members.items()
            if gid not in new_ids and not all(item_key(i) in assignment for i in group)
        ]
        if partial:
            raise ValueError(f"groups split across assigned and unassigned items: {partial}")
        rng.shuffle(new_ids)
        for gid in new_ids:
            size = len(members[gid])
            if abs(in_test + size - test_share * total) < abs(in_test - test_share * total):
                split, in_test = "test", in_test + size
            else:
                split = "dev"
            for item in members[gid]:
                assignment[item_key(item)] = split
    return {
        "seed": seed,
        "test_share": test_share,
        "note": (
            "group_id-preserving, stratified by source and category; pending_source items "
            "excluded; earlier assignments frozen, new groups added toward the target share"
        ),
        "items": dict(sorted(assignment.items())),
    }


def load_split(path: Path = SPLIT_FILE) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def items_in(split: str, path: Path = SPLIT_FILE) -> list:
    assignment = load_split(path)
    return [item for item in scored_items(load_eval_items()) if assignment[item_key(item)] == split]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reshuffle", action="store_true", help="ignore split.json and re-split from scratch"
    )
    args = parser.parse_args()
    frozen = None if args.reshuffle or not SPLIT_FILE.exists() else load_split()
    split = make_split(frozen=frozen)
    SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    by_key = {item_key(item): item for item in scored_items(load_eval_items())}
    for key, part in split["items"].items():
        counts[(source_kind(by_key[key]), by_key[key].category, part)] += 1
    for source, category in sorted({(s, c) for s, c, _ in counts}):
        dev, test = counts[(source, category, "dev")], counts[(source, category, "test")]
        print(f"{source:9} {category:15} dev={dev:3} test={test:3}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from eval.harness import load_eval_items, scored_items
from eval.split import SPLIT_FILE, item_key, load_split, make_split
from ingest.index import load_processed_records

ROOT = Path(__file__).resolve().parents[1]
DISTRACTORS = ROOT / "data" / "processed" / "distractors.jsonl"


def test_split_file_is_reproducible_and_covers_every_scored_item() -> None:
    assignment = load_split()
    assert assignment == make_split()["items"]
    assert set(assignment) == {item_key(item) for item in scored_items(load_eval_items())}
    assert set(assignment.values()) == {"dev", "test"}


def test_split_keeps_groups_together_and_stratifies_by_category() -> None:
    assignment = load_split(SPLIT_FILE)
    items = scored_items(load_eval_items())
    by_group = defaultdict(set)
    for item in items:
        by_group[item.group_id].add(assignment[item_key(item)])
    assert all(len(parts) == 1 for parts in by_group.values())

    per_category = defaultdict(Counter)
    for item in items:
        per_category[item.category][assignment[item_key(item)]] += 1
    for category, counts in per_category.items():
        total = counts["dev"] + counts["test"]
        assert counts["dev"] and counts["test"], category
        assert abs(counts["test"] / total - 0.4) <= 0.15, (category, counts)


@pytest.fixture(scope="module")
def distractors():
    records = load_processed_records(DISTRACTORS)
    if not records:
        pytest.skip("run `make ingest` to build data/processed/distractors.jsonl")
    return records


def test_distractors_are_never_servable_and_never_shadow_verified_paths(distractors) -> None:
    assert all(not record.servable and not record.verified for record in distractors)
    verified = {
        record.path
        for name in ("core_provisions.jsonl", "provisions.jsonl")
        for record in load_processed_records(ROOT / "data" / "processed" / name)
        if record.verified
    }
    assert not verified & {record.path for record in distractors}


def test_distractors_cover_the_whole_act_and_unverified_rules(distractors) -> None:
    sections = {record.path.split("/")[1] for record in distractors if record.doc_type == "act"}
    # Every section 1-174 appears except 42, 43 and 43A, which the source shows as omitted
    # ("[***]", Finance Act 2022 w.e.f. 2022-10-01) and so have no text.
    omitted = {"s42", "s43", "s43A"}
    assert {f"s{number}" for number in range(1, 175)} - omitted <= sections
    assert not omitted & sections
    # 17(5) is split at clauses so the verified (a), (aa), (ab) and (b) are not duplicated.
    paths = {record.path for record in distractors}
    assert "CGST/s17/ss5/cl(c)" in paths and "CGST/s17/ss5/cl(a)" not in paths
    assert any(record.doc_type == "rule" for record in distractors)

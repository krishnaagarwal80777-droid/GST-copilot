from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from eval.harness import DEFAULT_QUESTION_FILE, TEMPLATED_QUESTION_FILE, load_eval_items
from eval.split import item_key, load_split
from eval.templated import ACT_SOURCE_HORIZON, hand_boundaries, indexed_versions
from ingest.rate_schedule import parse_rows
from retrieval.corpus import RATE_NOTIFICATIONS

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not TEMPLATED_QUESTION_FILE.exists(), reason="run `python -m eval.templated` first"
)


@pytest.fixture(scope="module")
def templated():
    return load_eval_items(TEMPLATED_QUESTION_FILE)


@pytest.fixture(scope="module")
def versions():
    return indexed_versions()


def test_templated_items_are_tagged_and_paired(templated) -> None:
    # 100 until 2026-10-12; the CBIC section pages added 4 post-2022 Act version pairs.
    assert 80 <= len(templated) <= 110
    groups = defaultdict(list)
    for item in templated:
        assert item.source == "templated" and item.answerable and item.template
        groups[item.group_id].append(item)
    for group_id, members in groups.items():
        assert len(members) == 2, group_id
        before, after = sorted(members, key=lambda item: item.ref_date)
        assert before.question == after.question
        assert before.gold_answer != after.gold_answer


def test_version_pairs_quote_the_version_valid_on_each_date(templated, versions) -> None:
    for item in templated:
        if item.template["kind"] != "version_change":
            continue
        (path,) = item.gold_paths
        valid = [
            v
            for v in versions[path]
            if v.effective_from <= item.ref_date
            and (v.effective_to is None or item.ref_date <= v.effective_to)
        ]
        assert [v.id for v in valid] == [item.template["record_id"]], item_key(item)
        assert item.gold_answer == " ".join(valid[0].text.split())
        if path.startswith("CGST/s"):
            assert item.ref_date <= ACT_SOURCE_HORIZON


def test_no_templated_boundary_repeats_a_hand_written_pair(templated) -> None:
    covered = hand_boundaries(load_eval_items(DEFAULT_QUESTION_FILE))
    for item in templated:
        if item.template["kind"] == "version_change":
            from datetime import date

            boundary = date.fromisoformat(item.template["boundary"])
            assert (item.gold_paths[0], boundary) not in covered


def test_rate_pairs_match_parsed_rows_and_change_rate(templated) -> None:
    rows = {
        (row.notification, row.schedule, row.s_no): row
        for notification in RATE_NOTIFICATIONS
        for row in parse_rows(notification)
    }
    for item in templated:
        if item.template["kind"] != "rate_change":
            continue
        before, after = item.template["before"], item.template["after"]
        assert before["rate"] != after["rate"]
        side = before if item.ref_date.year == 2017 else after
        row = rows[(side["notification"], side["schedule"], side["s_no"])]
        assert item.gold_paths == [row.path]
        assert item.gold_answer.startswith(f"{row.rate} per cent.")
        (quoted,) = re.findall(r'"([^"]+)"', item.gold_answer)
        assert quoted == row.description


def test_rate_parser_reads_known_rows() -> None:
    # Rows quoted by the hand-written rate items (eval/questions.jsonl).
    one, nine = (n for n in RATE_NOTIFICATIONS if n.number in ("1/2017", "9/2025"))
    rows_1 = parse_rows(one)
    rows_9 = parse_rows(nine)
    assert any(
        r.schedule == "IV" and r.description.startswith("Dish washing machines, household")
        for r in rows_1
    )
    ghee = [r for r in rows_9 if r.description.startswith("Butter and other fats and oils")]
    assert [(r.schedule, r.tariff, r.rate) for r in ghee] == [("I", "0405", "2.5")]
    for rows in (rows_1, rows_9):  # serial numbers run 1..N without gaps in every schedule
        by_schedule = defaultdict(list)
        for r in rows:
            by_schedule[r.schedule].append(r.s_no)
        for numbers in by_schedule.values():
            assert numbers == list(range(1, len(numbers) + 1))


def test_split_covers_templated_items_and_keeps_groups_together(templated) -> None:
    split = load_split()
    by_group = defaultdict(set)
    for item in templated:
        by_group[item.group_id].add(split[item_key(item)])
    assert all(len(parts) == 1 for parts in by_group.values())
    test_share = sum(1 for item in templated if split[item_key(item)] == "test") / len(templated)
    assert 0.3 <= test_share <= 0.5


def test_hand_written_assignments_are_unchanged() -> None:
    # Assignments as committed before templated items existed (eval/split.json at d24dac9).
    import subprocess

    try:
        old = subprocess.run(
            ["git", "show", "d24dac9:eval/split.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history unavailable")
    previous = json.loads(old)["items"]
    current = load_split()
    assert {key: current[key] for key in previous} == previous

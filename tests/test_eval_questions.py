from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from eval.harness import PENDING_SOURCE_GROUP, load_eval_items, scored_items, summarize_eval_items
from ingest.index import load_processed_records
from retrieval.provisions import get_provision

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "processed" / "core_provisions.jsonl"


@pytest.fixture(scope="module")
def items():
    return load_eval_items(ROOT / "eval" / "questions.jsonl")


@pytest.fixture(scope="module")
def core():
    return load_processed_records(CORE)


def test_eval_set_shape(items) -> None:
    assert len(items) == 50
    assert Counter(item.category for item in items) == {
        "date_sensitive": 16,
        "numeric": 10,
        "direct_lookup": 10,
        "interpretive": 9,
        "abstain": 5,
    }
    assert all(item.group_id for item in items)
    assert Counter(item.category for item in scored_items(items)) == {
        "date_sensitive": 14,
        "numeric": 10,
        "direct_lookup": 9,
        "interpretive": 8,
        "abstain": 5,
    }
    for item in items:
        assert item.answerable == (item.category != "abstain")
        assert bool(item.gold_paths) == item.answerable


def test_act_items_past_the_source_horizon_are_pending_and_unscored(items) -> None:
    # The newest consolidated Act source is cgst-act_cbic-taxinfo_2022-09-28.pdf.
    for item in items:
        act = any(path.startswith("CGST/s") for path in item.gold_paths)
        pending = act and item.ref_date.isoformat() > "2022-09-28"
        assert (item.group_id == PENDING_SOURCE_GROUP) == pending, item.question
    summary = summarize_eval_items(items)
    assert (summary["num_items"], summary["num_scored"], summary["pending_source"]) == (50, 46, 4)


def test_linked_items_are_same_question_on_two_dates(items) -> None:
    groups = defaultdict(list)
    for item in scored_items(items):
        groups[item.group_id].append(item)
    for group_id, members in groups.items():
        if members[0].category in {"date_sensitive", "numeric"}:
            assert len(members) == 2, group_id
            assert members[0].question == members[1].question, group_id
            assert members[0].ref_date < members[1].ref_date, group_id
            assert members[0].gold_answer != members[1].gold_answer, group_id
        else:
            assert len(members) == 1, group_id


def test_gold_paths_resolve_on_the_reference_date(items, core) -> None:
    for item in items:
        for path in item.gold_paths:
            if path.startswith("CTR/"):
                assert re.fullmatch(r"CTR/\d+-\d{4}/sch[IVX]+", path), path
                continue
            served = get_provision(path, item.ref_date, records=core)
            assert served is not None, (item.group_id, path)
            assert served.effective_from <= item.ref_date, (item.group_id, path)


def test_date_pairs_cross_a_version_boundary(items, core) -> None:
    groups = defaultdict(list)
    for item in scored_items(items):
        if item.category == "date_sensitive":
            groups[item.group_id].append(item)
    for group_id, (before, after) in groups.items():
        changed = [
            path
            for path in set(before.gold_paths) | set(after.gold_paths)
            if get_provision(path, before.ref_date, records=core)
            != get_provision(path, after.ref_date, records=core)
        ]
        assert changed, group_id


# Phrases each gold answer quotes, checked against the version served on its ref_date.
QUOTED_IN_SOURCE = [
    ("CGST/s17/ss5/cl(a)", "2019-01-31", "motor vehicles and other conveyances"),
    ("CGST/s17/ss5/cl(a)", "2019-02-01", "not more than thirteen persons (including the driver)"),
    (
        "CGST/s16/ss4",
        "2022-09-30",
        "for the month of September following the end of financial year",
    ),
    ("CGST/s16/ss4", "2022-10-01", "thirtieth day of November following the end of financial year"),
    ("CGST/s16/ss4/proviso1", "2018-12-31", "for the month of March, 2019"),
    ("CGST/r36/ss4", "2021-12-31", "shall not exceed 5 per cent. of the eligible credit"),
    ("CGST/r36/ss4", "2022-01-01", "communicated to the registered person in FORM GSTR-2B"),
    ("CGST/r36/ss4", "2021-05-31", "cumulatively for the period April and May, 2021"),
    ("CGST/r36/ss4", "2021-06-01", "cumulatively for the period April, May and June, 2021"),
    (
        "CGST/r89",
        "2021-09-24",
        "before the expiry of a period of two years from the date of payment",
    ),
    ("CGST/r89", "2021-12-31", "may file an application electronically in FORM GST RFD"),
    ("CGST/r89", "2022-01-01", "may file, subject to the provisions of rule 10B, an application"),
    ("CGST/s9/ss4", "2019-01-31", "by a supplier, who is not registered, to a registered person"),
    ("CGST/s9/ss4", "2019-02-01", "specify a class of registered persons"),
    ("CGST/r86B", "2026-03-01", "goods specified under rule 31D"),
    ("CGST/r86B", "2026-03-01", "ninety-nine per cent. of such tax liability"),
    ("CGST/r36/ss4", "2025-01-01", "as amended in FORM GSTR-1A if any"),
    ("CGST/s17/ss5/cl(ab)", "2023-04-01", "insured by him"),
    # Items re-dated to the 2022-09-28 source horizon.
    (
        "CGST/s16/ss4",
        "2022-09-28",
        "for the month of September following the end of financial year",
    ),
    (
        "CGST/r36/ss4",
        "2022-09-28",
        "the details of such invoices or debit notes have been communicated",
    ),
    ("CGST/s17/ss5/cl(ab)", "2022-09-28", "insured by him"),
    ("CGST/s9/ss3", "2022-09-28", "paid on reverse charge basis by the recipient"),
]


@pytest.mark.parametrize(("path", "on", "phrase"), QUOTED_IN_SOURCE)
def test_gold_answer_phrases_come_from_the_source_version(core, path, on, phrase) -> None:
    served = get_provision(path, on, records=core)
    assert phrase in " ".join(served.text.split()), (path, on)


def test_rule_89_before_1A_has_no_section_77_refund(core) -> None:
    served = get_provision("CGST/r89", "2021-09-23", records=core)
    assert "(1A)" not in served.text
    assert "section 77" not in served.text


def test_naive_relevance_marks_only_text_the_gold_version_introduced() -> None:
    from eval.retrieval import naive_matchers
    from retrieval.corpus import provision_rows
    from retrieval.naive import shingles
    from retrieval.store import row_dict

    rows = [row_dict(row) for row in provision_rows()]
    item = next(
        i
        for i in load_eval_items(ROOT / "eval" / "questions.jsonl")
        if i.group_id == "ds-r89-1A" and i.ref_date.isoformat() == "2021-09-24"
    )
    ((_, marks),) = naive_matchers(item, rows).values()
    introduced = shingles(
        "may, before the expiry of a period of two years from the date of payment", 8
    )
    unchanged = shingles("may file an application electronically in FORM GST RFD", 8)
    assert marks & introduced
    assert not marks & unchanged

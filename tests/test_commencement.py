from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ingest.all_rule_versions import _read_events
from ingest.commencement import (
    Appointment,
    UnresolvedScopeError,
    instruction_label,
    parse_commencement_text,
    resolve_pending,
)
from ingest.notifications import (
    instruction_commencement,
    is_pending_commencement,
    parse_notification_text,
    read_pdf_text,
)
from retrieval.provisions import get_provision

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NOTIFICATION_35_2021 = RAW / "ct-35-2021_2021-09-24.pdf"
# Placeholder for the mechanism tests only; real appointed dates come from data/raw.
SENTINEL = date.max


@pytest.fixture(scope="module")
def notification_35_2021() -> dict:
    return parse_notification_text(
        read_pdf_text(NOTIFICATION_35_2021), source_file=NOTIFICATION_35_2021.name
    )


def _by_label(notification: dict) -> dict[tuple[str, str, str], dict]:
    return {
        (str(row["rule"]), *instruction_label(str(row["instruction"]))): row
        for row in notification["amendments"]
    }


def test_notification_35_2021_deferred_instructions_are_pending(notification_35_2021) -> None:
    rows = _by_label(notification_35_2021)

    for key in (("10A", "1", ""), ("23", "3", ""), ("89", "6", "i"), ("96", "7", "")):
        assert rows[key]["commencement"] == "pending", key
        assert rows[key]["effective_date"] is None, key
    # Siblings in the same notification keep their own dates.
    assert rows[("89", "6", "ii")]["commencement"] == "fixed"
    assert rows[("89", "6", "ii")]["effective_date"] == "2021-09-24"
    assert rows[("45", "4", "i")]["effective_date"] == "2021-10-01"


def test_pending_phrase_inside_quoted_inserted_text_does_not_defer() -> None:
    assert is_pending_commencement("(1) In rule 10A, with effect from the date as may be notified,")
    assert not is_pending_commencement(
        "(2) after sub-rule (1), the following shall be inserted, namely:- "
        "“with effect from the date as may be notified”"
    )


def test_engine_never_builds_versions_from_pending_instructions() -> None:
    ledger = ROOT / "data" / "processed" / "amendments.jsonl"
    events = [event for event in _read_events(ledger) if event["notification_number"] == "35/2021"]
    # Still pending (outside 38/2021's scope): never an event.
    assert {event["rule"] for event in events}.isdisjoint({"10A", "96C"})
    # Deferred, then appointed by 38/2021: an event only from the appointed date.
    deferred = [
        event
        for event in events
        if event["rule"] in {"10B", "23", "96"} or "rule 10B" in event["instruction"]
    ]
    assert deferred
    assert {event["effective_date"] for event in deferred} == {date(2022, 1, 1)}

    # Nothing built from 35/2021's deferred paragraphs is served on its publication date.
    for path in ("CGST/r10A", "CGST/r23", "CGST/r96"):
        served = get_provision(path, date(2021, 9, 24), include_unverified=True)
        assert served is None or served.amended_by is None, path
    rule89 = get_provision("CGST/r89", date(2021, 9, 24), include_unverified=True)
    assert "(1A)" in rule89.text
    assert "subject to the provisions of rule 10B" not in rule89.text.split("(1A)")[0]


def test_resolver_dates_only_the_instructions_in_scope(notification_35_2021) -> None:
    notification = json.loads(json.dumps(notification_35_2021))
    appointment = Appointment(
        notification_number="sentinel",
        source_file="sentinel.pdf",
        appointed_date=SENTINEL,
        target_notification="35/2021",
        scope=(("6", "i"),),
    )
    resolve_pending([notification], [appointment])
    rows = _by_label(notification)

    assert rows[("89", "6", "i")]["commencement"] == "resolved"
    assert rows[("89", "6", "i")]["effective_date"] == SENTINEL.isoformat()
    assert rows[("89", "6", "i")]["commenced_by"] == "sentinel"
    assert rows[("89", "6", "ii")]["effective_date"] == "2021-09-24"
    assert rows[("23", "3", "")]["commencement"] == "pending"


def test_resolver_rejects_scope_that_matches_nothing(notification_35_2021) -> None:
    appointment = Appointment(
        notification_number="sentinel",
        source_file="sentinel.pdf",
        appointed_date=SENTINEL,
        target_notification="35/2021",
        scope=(("4", "i"),),  # para (4) is fixed (rule 45), not pending
    )
    with pytest.raises(UnresolvedScopeError):
        resolve_pending([json.loads(json.dumps(notification_35_2021))], [appointment])


def test_act_commencement_notifications_are_not_rule_appointments() -> None:
    source = RAW / "ct-18-2022_2022-09-28.pdf"
    assert (
        parse_commencement_text(
            read_pdf_text(source), notification_number="18/2022", source_file=source.name
        )
        == []
    )


LEDGER = ROOT / "data" / "processed" / "amendments.jsonl"
REPORT = ROOT / "data" / "processed" / "rule_version_report.json"


@pytest.fixture(scope="module")
def ledger() -> dict[str, dict]:
    rows = [json.loads(line) for line in LEDGER.open(encoding="utf-8") if line.strip()]
    return {row["notification_number"]: row for row in rows}


def _parse(name: str) -> dict:
    source = next(RAW.glob(name))
    return parse_notification_text(read_pdf_text(source), source_file=source.name)


def test_unparseable_date_is_pending_not_the_publication_date() -> None:
    # 20/2024 para 5 ends "... shall be inserted with effect from the 1st day of November, 2024."
    # Cut before the year (as the old splitter did), the date is unreadable and must not
    # silently become the 2024-10-08 publication date.
    row = next(a for a in _parse("ct-20-2024_*.pdf")["amendments"] if a["rule"] == "66")
    truncated = row["instruction"][: row["instruction"].rindex("2024")]
    assert instruction_commencement(
        truncated, issue_date=date(2024, 10, 8), default=date(2024, 10, 8)
    ) == (None, "pending", "unparsed_date")
    assert row["effective_date"] == "2024-11-01"  # the full instruction keeps its stated date


def test_unparsed_and_pending_instructions_are_listed_in_the_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    listed = report["pending_commencement"]
    assert report["instructions_unparsed_date"] == sum(
        row["reason"] == "unparsed_date" for row in listed
    )
    assert {
        (row["notification_number"], row["rule"]) for row in listed if row["status"] == "pending"
    } >= {
        ("35/2021", "10A"),
        ("35/2021", "96C"),
    }


def test_35_2021_paragraph_targets_and_new_rule_insertions(notification_35_2021) -> None:
    by_para: dict[str, list[dict]] = {}
    for row in notification_35_2021["amendments"]:
        by_para.setdefault(instruction_label(row["instruction"])[0], []).append(row)

    # Para (7) amends rule 96; rule 10B appears only inside its quoted inserted clause.
    assert {row["rule"] for row in by_para["7"]} == {"96"}
    for para, after, rule in (("2", "10A", "10B"), ("8", "96B", "96C")):
        (row,) = by_para[para]
        assert (row["rule"], row["operation"], row["after_rule"]) == (rule, "insert_rule", after)

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    inserted = {(row["notification_number"], row["rule"]) for row in report["new_rules_inserted"]}
    not_in_base = {
        (row["notification_number"], row["rule"]) for row in report["target_rule_not_in_base"]
    }
    assert ("35/2021", "10B") in inserted
    assert not any(number == "35/2021" for number, _ in not_in_base)


def test_notification_38_2021_commences_35_2021_by_sub_rule_of_rule_2(ledger) -> None:
    source = next(RAW.glob("ct-38-2021_*.pdf"))
    (appointment,) = parse_commencement_text(
        read_pdf_text(source), notification_number="38/2021", source_file=source.name
    )
    assert appointment.target_notification == "35/2021"
    assert appointment.appointed_date == date(2022, 1, 1)
    assert set(appointment.scope) == {("2", ""), ("3", ""), ("6", "i"), ("7", "")}

    rows = _by_label(ledger["35/2021"])
    for key in (("10B", "2", ""), ("23", "3", ""), ("89", "6", "i"), ("96", "7", "")):
        assert rows[key]["effective_date"] == "2022-01-01", key
        assert rows[key]["commenced_by"] == "38/2021", key
    # Sub-rules (1) and (8) of rule 2 are outside 38/2021's scope and stay pending.
    assert rows[("10A", "1", "")]["commencement"] == "pending"
    assert rows[("96C", "8", "")]["commencement"] == "pending"


def test_notification_09_2025_commences_12_2024_by_paragraph_not_rule_number(ledger) -> None:
    source = next(RAW.glob("ct-09-2025_*.pdf"))
    appointments = parse_commencement_text(
        read_pdf_text(source), notification_number="09/2025", source_file=source.name
    )
    assert [(a.target_notification, a.appointed_date, a.scope) for a in appointments] == [
        ("12/2024", date(2025, 2, 11), (("2", ""), ("24", ""), ("27", ""), ("32", ""))),
        ("12/2024", date(2025, 4, 1), (("8", ""), ("37", ""), ("38", "ii"))),
    ]

    by_para: dict[str, set] = {}
    for row in ledger["12/2024"]["amendments"]:
        para, sub = instruction_label(row["instruction"])
        by_para.setdefault(para, set()).add(
            (row["rule"], sub, row["effective_date"], row["commencement"])
        )
    # Para 2 amends rule 8 and para 8 amends rule 39: dates follow the paragraph, not the rule.
    assert {(rule, when) for rule, _, when, _ in by_para["2"]} == {("8", "2025-02-11")}
    assert {(rule, when) for rule, _, when, _ in by_para["24"]} == {("138", "2025-02-11")}
    assert {(rule, when) for rule, _, when, _ in by_para["8"]} == {("39", "2025-04-01")}
    assert {status for *_, status in by_para["2"] | by_para["8"] | by_para["24"]} == {"resolved"}
    # Paragraphs amending forms are not rule instructions; they are recorded, not dropped.
    assert {row["paragraph"] for row in ledger["09/2025"]["unmatched_scope"]} == {
        "27",
        "32",
        "37",
        "38(ii)",
    }


def test_45_2023_is_superseded_by_51_2023_and_never_builds(ledger) -> None:
    assert ledger["45/2023"]["superseded_by"] == "51/2023"
    assert ledger["45/2023"]["amendments"]
    assert {row["commencement"] for row in ledger["45/2023"]["amendments"]} == {"superseded"}
    assert not [
        event for event in _read_events(LEDGER) if event["notification_number"] == "45/2023"
    ]
    assert {row["effective_date"] for row in ledger["51/2023"]["amendments"]} == {"2023-10-01"}

import json
from datetime import date

import pytest

from gst_copilot.models import ProvisionRecord
from ingest.all_rule_versions import (
    AMENDMENTS_PATH,
    RULES_PDF,
    AmendmentApplicationError,
    _apply_generic_instruction,
    _extract_rule_bases,
    _inserted_text,
    _node_span,
    _read_events,
    _read_pdf_text,
    run_all_rule_versions,
)
from ingest.index import load_processed_records
from ingest.rule_versions import build_rule_36_provisions
from retrieval.provisions import get_provision


def _rule36_versions() -> list[ProvisionRecord]:
    return build_rule_36_provisions()


def test_all_rule_base_extraction_finds_multiple_rules() -> None:
    bases = _extract_rule_bases(_read_pdf_text(RULES_PDF))

    assert len(bases) > 100
    assert bases["8"][0] == "Application for registration"
    assert bases["36"][0] == "Documentary requirements and conditions for claiming input tax credit"
    assert bases["21"][0] == "Registration to be cancelled in certain cases"
    assert bases["21A"][0] == "Suspension of registration"
    assert bases["138"][0].startswith("Information to be furnished prior")
    _node_span(bases["59"][1], "sub-rule", "2")


def test_generic_word_substitution_is_scoped_to_its_subrule() -> None:
    text = '8. Example.-(1) Keep "same" here. (2) Replace "same" here.'
    instruction = (
        'in sub-rule (2), for the words "same", the words "new" shall be substituted'
    )

    updated = _apply_generic_instruction(text, instruction)

    assert 'Keep "same" here.' in updated
    assert 'Replace "new" here.' in updated


@pytest.mark.parametrize(
    ("opening_quote", "closing_quote"),
    [("\u201c", "\u201d"), ("\u2015", "\u2016")],
)
def test_word_insertion_accepts_normalized_quote_glyphs(
    opening_quote: str, closing_quote: str
) -> None:
    text = "10A. Example.-Details of bank account are recorded."
    instruction = (
        f"after the words {opening_quote}details of bank account{closing_quote}, "
        f"the words {opening_quote}verified taxpayer account{closing_quote} "
        "shall be inserted"
    )

    updated = _apply_generic_instruction(text, instruction)

    assert "details of bank account verified taxpayer account" in updated


def test_inserted_text_tolerates_one_unmatched_opening_quote() -> None:
    instruction = (
        "after rule 110, the following rule shall be inserted, namely: "
        "\u201c110A. Example.-This text remains inside the inserted rule."
    )

    assert _inserted_text(instruction) == (
        "110A. Example.-This text remains inside the inserted rule"
    )


def test_source_notifications_40_2021_and_13_2025_quotes_parse() -> None:
    events = _read_events(AMENDMENTS_PATH)
    event_40 = next(
        event
        for event in events
        if event["notification_number"] == "40/2021" and event["rule"] == "36"
    )
    event_13 = next(
        event
        for event in events
        if event["notification_number"] == "13/2025" and event["rule"] == "110A"
    )

    assert _inserted_text(event_40["instruction"]).startswith("(4) No input tax credit")
    assert _inserted_text(event_13["instruction"]) == "110A"


def test_structural_operations_replace_insert_and_omit_legal_nodes() -> None:
    rule = "8. Example.-(1) Old wording. (2) Keep this."
    substitution = (
        "in rule 8, for sub-rule (1), the following shall be substituted, "
        'namely: "(1) Replacement wording."'
    )
    replaced = _apply_generic_instruction(rule, substitution)
    assert "(1) Replacement wording." in replaced
    assert "(1) Old wording." not in replaced
    assert "(2) Keep this." in replaced

    clause_rule = "21. Example.-(a) First clause; or (b) Second clause."
    insertion = (
        "in rule 21, after clause (b), the following clause shall be inserted, "
        'namely: "(c) Third clause."'
    )
    inserted = _apply_generic_instruction(clause_rule, insertion)
    assert inserted.index("(b) Second clause.") < inserted.index("(c) Third clause.")

    before_insertion = (
        "in rule 21, before clause (b), the following clause shall be inserted, "
        'namely: "(c) Before the second clause."'
    )
    inserted_before = _apply_generic_instruction(clause_rule, before_insertion)
    assert inserted_before.index("(c) Before the second clause.") < inserted_before.index(
        "(b) Second clause."
    )

    rule_with_clause = "59. Example.-(6) Main text; (a) Keep this; (b) Remove this."
    omitted_clause = _apply_generic_instruction(
        rule_with_clause,
        "in rule 59, in sub-rule (6), (b) shall be omitted",
        "6",
    )
    assert "(a) Keep this" in omitted_clause
    assert "(b) Remove this" not in omitted_clause

    misplaced_omission = (
        "3. For refund under rule 96; (c) shall be omitted; "
        "(6) In rule 89 of the said rules."
    )
    with pytest.raises(AmendmentApplicationError, match="quoted omission text"):
        _apply_generic_instruction(rule_with_clause, misplaced_omission)

    proviso_rule = "36. Example.-(1) Main text. Provided that the condition applies."
    omission = "in sub-rule (1), the proviso shall be omitted"
    omitted = _apply_generic_instruction(proviso_rule, omission, "1")
    assert "Provided that" not in omitted
    assert "(1) Main text." in omitted

    explanation_rule = "59. Example.-(6) Main text. Provided that the condition applies."
    explanation_insertion = (
        "in rule 59, in sub-rule (6), after the proviso, the following explanation "
        'shall be inserted, namely: "Explanation. This is added."'
    )
    inserted_explanation = _apply_generic_instruction(
        explanation_rule, explanation_insertion, "6"
    )
    assert inserted_explanation.index("Provided that") < inserted_explanation.index(
        "Explanation."
    )


def test_report_separates_root_cascade_and_missing_base_targets(tmp_path) -> None:
    amendments_path = tmp_path / "amendments.jsonl"
    amendments = [
        {
            "notification_number": "90/2022",
            "issue_date": "2022-01-01",
            "amendments": [
                {
                    "rule": "8",
                    "subrule": "999",
                    "effective_date": "2022-01-01",
                    "instruction": (
                        'in sub-rule (999), for the words "missing", the words '
                        '"text" shall be substituted'
                    ),
                }
            ],
        },
        {
            "notification_number": "91/2022",
            "issue_date": "2022-02-01",
            "amendments": [
                {
                    "rule": "8",
                    "subrule": None,
                    "effective_date": "2022-02-01",
                    "instruction": (
                        'for the words "old", the words "new" shall be substituted'
                    ),
                }
            ],
        },
        {
            "notification_number": "92/2022",
            "issue_date": "2022-03-01",
            "amendments": [
                {
                    "rule": "999",
                    "subrule": None,
                    "effective_date": "2022-03-01",
                    "instruction": "",
                }
            ],
        },
    ]
    amendments_path.write_text(
        "".join(json.dumps(item) + "\n" for item in amendments),
        encoding="utf-8",
    )

    report = run_all_rule_versions(
        raw_pdf=RULES_PDF,
        amendments_path=amendments_path,
        output_path=tmp_path / "provisions.jsonl",
        report_path=tmp_path / "report.json",
    )

    assert report["instructions_root_failures"] == 1
    assert report["instructions_cascade_failures"] == 1
    assert report["instructions_target_rule_not_in_base"] == 1
    assert report["instructions_flagged"] == 2
    assert report["root_failures"][0]["status"] == "root_failure"
    assert report["cascade_failures"][0]["status"] == "cascade_failure"
    assert report["target_rule_not_in_base"][0]["status"] == "target_rule_not_in_base"
    assert len(report["top_10_root_failures"]) == 1
    generated = load_processed_records(tmp_path / "provisions.jsonl")
    rule_8_records = [record for record in generated if record.path == "CGST/r8"]
    assert rule_8_records
    assert all(not record.verified for record in rule_8_records)


def test_rule36_base_pdf_already_includes_notification_13_2021_proviso() -> None:
    records = _rule36_versions()
    rule_36 = get_provision(
        "CGST/r36",
        date(2021, 5, 31),
        records=records,
        knowledge_date=date(2026, 10, 8),
    )
    rule_36_4 = get_provision(
        "CGST/r36/ss4",
        date(2021, 5, 31),
        records=records,
        knowledge_date=date(2026, 10, 8),
    )

    assert rule_36 is not None
    assert rule_36.effective_from == date(2021, 5, 1)
    assert rule_36.amended_by == "Notification No. 13/2021-Central Tax"
    assert rule_36_4 is not None
    assert "April and May, 2021" in rule_36_4.text
    assert rule_36_4.effective_from == date(2021, 5, 1)
    assert rule_36_4.amended_by == "Notification No. 13/2021-Central Tax"
    assert rule_36_4.in_force_since == date(2021, 5, 1)


def test_rule36_4_changes_on_2022_01_01() -> None:
    records = _rule36_versions()
    before = get_provision(
        "CGST/r36/ss4",
        date(2021, 12, 31),
        records=records,
        knowledge_date=date(2026, 10, 8),
    )
    after = get_provision(
        "CGST/r36/ss4",
        date(2022, 1, 2),
        records=records,
        knowledge_date=date(2026, 10, 8),
    )

    assert before is not None and after is not None
    assert before.text != after.text
    assert "April, May and June, 2021" in before.text
    assert "FORM GSTR-2B" in after.text
    assert after.amended_by == "Notification No. 40/2021-Central Tax"


def test_rule36_3_on_2024_10_09_contains_section_74() -> None:
    record = get_provision(
        "CGST/r36/ss3",
        date(2024, 10, 9),
        knowledge_date=date(2026, 10, 8),
    )

    assert record is not None
    assert "under section 74" in record.text
    assert record.amended_by == "Notification No. 20/2024-Central Tax"


def test_rule36_2_omission_from_notification_19_2022() -> None:
    records = _rule36_versions()
    before = get_provision(
        "CGST/r36/ss2",
        date(2022, 9, 30),
        records=records,
        knowledge_date=date(2026, 10, 8),
    )
    after = get_provision(
        "CGST/r36/ss2",
        date(2022, 10, 1),
        records=records,
        knowledge_date=date(2026, 10, 8),
    )

    assert before is not None and after is not None
    assert "FORM GSTR-2 by such person" in before.text
    assert "FORM GSTR-2 by such person" not in after.text


def test_rule36_4_versions_have_exact_effective_timeline() -> None:
    records = _rule36_versions()
    timeline = [
        (record.effective_from.isoformat(), record.amended_by)
        for record in records
        if record.path == "CGST/r36/ss4"
    ]

    assert timeline == [
        ("2021-05-01", "Notification No. 13/2021-Central Tax"),
        ("2021-06-01", "Notification No. 27/2021-Central Tax"),
        ("2022-01-01", "Notification No. 40/2021-Central Tax"),
        ("2022-10-01", "Notification No. 19/2022-Central Tax"),
        ("2024-07-10", "Notification No. 12/2024-Central Tax"),
    ]


def test_get_provision_obeys_effective_and_knowledge_dates() -> None:
    records = _rule36_versions()

    assert (
        get_provision(
            "CGST/r36/ss4",
            date(2021, 4, 30),
            records=records,
            knowledge_date=date(2026, 10, 8),
        )
        is None
    )
    version_before_knowledge = get_provision(
            "CGST/r36/ss4",
            on="2022-01-02",
            records=records,
            knowledge_date="2021-12-28",
        )
    version_after_knowledge = get_provision(
        "CGST/r36/ss4",
        on="2022-01-02",
        records=records,
        knowledge_date="2021-12-30",
    )

    assert version_before_knowledge is not None
    assert version_before_knowledge.id == "CGST/r36/ss4@v2"
    assert version_after_knowledge is not None
    assert version_after_knowledge.id == "CGST/r36/ss4@v3"


def test_rule36_3_notification_20_2024_effective_date_boundary() -> None:
    records = _rule36_versions()
    before = get_provision(
        "CGST/r36/ss3",
        on="2024-10-07",
        records=records,
        knowledge_date="2026-10-08",
    )
    after = get_provision(
        "CGST/r36/ss3",
        on="2024-10-08",
        records=records,
        knowledge_date="2026-10-08",
    )

    assert before is not None and "under section 74" not in before.text
    assert after is not None and "under section 74" in after.text


def _ledger_instruction(number: str, rule: str, needle: str) -> str:
    with open(AMENDMENTS_PATH, encoding="utf-8") as handle:
        for line in handle:
            notification = json.loads(line)
            if notification.get("notification_number") != number:
                continue
            for amendment in notification["amendments"]:
                if str(amendment["rule"]) == rule and needle in amendment["instruction"]:
                    return amendment["instruction"]
    raise AssertionError(f"{number} rule {rule}: instruction with {needle!r} not found")


def test_insert_after_clause_adds_a_sibling_not_a_nested_item() -> None:
    # 20/2025 para 3: "in rule 86B, in the first proviso, after clause (e), the following clause
    # shall be inserted". Clause (e) has its own items (i)-(iv); (f) must follow (iv).
    base = _extract_rule_bases(_read_pdf_text(RULES_PDF))["86B"][1]
    instruction = _ledger_instruction("20/2025", "86B", "after clause (e)")
    updated = " ".join(_apply_generic_instruction(base, instruction).split())

    assert "(e) the registered person is \u2013 (i) Government Department;" in updated
    assert "(iv) a statutory body: (f) the registered person other than a manufacturer" in updated
    assert updated.index("(f) the registered person") < updated.index("Provided further")


def test_insert_after_roman_clause_lands_before_the_next_roman_sibling() -> None:
    # 12/2024 inserts (iia) after clause (ii) of rule 60(7); (ii) has its own (a)-(c).
    served = get_provision("CGST/r60", "2024-07-10", include_unverified=True)
    text = " ".join(served.text.split())
    assert (
        text.index("(c) for the third month of the quarter")
        < text.index("(iia) the additional details")
        < text.index("(iii) the details of the integrated tax")
    )
    assert "; ;" not in text

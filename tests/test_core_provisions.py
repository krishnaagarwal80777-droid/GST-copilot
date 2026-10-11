from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
import yaml

from ingest.core_provisions import (
    ACT_PDF,
    _apply_core_rule_edits,
    _attach_rule_sources,
    _manual_rule43_versions,
    _manual_rule89_versions,
    _read_pdf_text,
    build_act_core_provisions,
    build_core_provisions,
)
from ingest.index import load_processed_records

ROOT = Path(__file__).resolve().parents[1]


def test_core_provision_manifest_lists_requested_rules_and_act_sections() -> None:
    manifest = yaml.safe_load((ROOT / "eval" / "core_provisions.yaml").read_text(encoding="utf-8"))

    assert {row["rule"] for row in manifest["rules"]} == {"36", "42", "43", "86B", "89"}
    assert {row["section"] for row in manifest["act_provisions"]} == {"9(3)", "9(4)", "16", "17(5)"}


def test_section_17_5_a_versions_follow_the_source_footnote() -> None:
    records, footnotes = build_act_core_provisions(_read_pdf_text(ACT_PDF))
    versions = sorted(
        (record for record in records if record.path == "CGST/s17/ss5/cl(a)"),
        key=lambda record: record.effective_from,
    )

    assert len(versions) == 2
    assert versions[0].effective_from == date(2017, 7, 1)
    assert versions[0].effective_to == date(2019, 1, 31)
    assert "motor vehicles and other conveyances" in versions[0].text
    assert versions[1].effective_from == date(2019, 2, 1)
    assert "approved seating capacity of not more than thirteen persons" in versions[1].text

    footnote = next(note for note in footnotes["17"] if note["marker"] == "2")
    assert footnote["effective_from"] == "2019-02-01"
    assert (
        "s.9 of The Central Goods and Services Tax (Amendment) Act, 2018" in footnote["instrument"]
    )
    assert "motor vehicles and other conveyances" in footnote["old_text"]


def test_section_9_and_16_footnote_versions_are_bitemporal() -> None:
    records, footnotes = build_act_core_provisions(_read_pdf_text(ACT_PDF))
    by_path: dict[str, list] = {}
    for record in records:
        by_path.setdefault(record.path, []).append(record)

    section9_4 = sorted(by_path["CGST/s9/ss4"], key=lambda record: record.effective_from)
    section16_explanation = sorted(
        by_path["CGST/s16/ss2/cl(b)/explanation1"],
        key=lambda record: record.effective_from,
    )
    assert section9_4[0].effective_to == date(2019, 1, 31)
    assert section9_4[1].effective_from == date(2019, 2, 1)
    assert section16_explanation[0].effective_to == date(2019, 1, 31)
    assert section16_explanation[1].effective_from == date(2019, 2, 1)

    section16_old = next(note for note in footnotes["16"] if note["marker"] == "2")
    assert section16_old["effective_from"] == "2019-02-01"
    assert section16_old["old_text"].startswith("Explanation.")
    assert "s. 8" in section16_old["instrument"]


def test_core_rule_42_and_43_manual_versions_apply_notification_19_2022() -> None:
    processed = load_processed_records(ROOT / "data" / "processed" / "provisions.jsonl")
    requested_paths = {"CGST/r42", "CGST/r43"}
    reviewed = _apply_core_rule_edits(
        [record for record in processed if record.path in requested_paths]
    )

    for path, expected_count in (("CGST/r42", 2), ("CGST/r43", 2)):
        versions = sorted(
            (record for record in reviewed if record.path == path),
            key=lambda record: record.effective_from,
        )
        assert len(versions) == 2
        assert all(record.verified and record.verified_by == "manual" for record in versions)
        assert versions[0].effective_to == date(2022, 9, 30)
        assert versions[1].effective_from == date(2022, 10, 1)
        assert versions[1].amended_by == "Notification No. 19/2022-Central Tax"
        assert versions[0].text != versions[1].text
        if path == "CGST/r42":
            target = "at the invoice level in FORM GSTR-2"
            assert versions[0].text.count(target) == 1
            assert target not in versions[1].text
        else:
            assert versions[0].text.count("FORM GSTR-2[and FORM GSTR-3B]") == expected_count
            assert versions[1].text.count("FORM GSTR-2[and FORM GSTR-3B]") == 0


def test_rule_43_notification_38_2023_versions_match_source_boundaries() -> None:
    processed = load_processed_records(ROOT / "data" / "processed" / "provisions.jsonl")
    rule_records = [record for record in processed if record.path in {"CGST/r42", "CGST/r43"}]
    reviewed = _attach_rule_sources(
        _manual_rule43_versions(_apply_core_rule_edits(rule_records)),
        ROOT / "data" / "processed" / "amendments.jsonl",
    )
    versions = sorted(
        (record for record in reviewed if record.path == "CGST/r43"),
        key=lambda record: record.effective_from,
    )

    assert [(row.effective_from, row.effective_to, row.amended_by) for row in versions] == [
        (date(2021, 5, 18), date(2022, 9, 30), None),
        (
            date(2022, 10, 1),
            date(2023, 8, 3),
            "Notification No. 19/2022-Central Tax",
        ),
        (
            date(2023, 8, 4),
            date(2023, 9, 30),
            "Notification No. 38/2023-Central Tax",
        ),
        (
            date(2023, 10, 1),
            None,
            "Notification No. 38/2023-Central Tax",
        ),
    ]
    assert all(row.verified and row.verified_by == "manual" for row in versions)
    assert all(row.source_url for row in versions)
    assert all((ROOT / row.source_url).is_file() for row in versions)

    explanation1 = re.search(
        r"Explanation\s*1\s*:[\s\S]*?(?=Explanation\s*2\s*:)",
        versions[2].text,
        re.IGNORECASE,
    )
    assert explanation1 is not None
    assert "transportation of goods by a vessel" not in explanation1.group()
    assert "incoming passengers" not in versions[2].text
    assert "incoming passengers" in versions[3].text
    assert "sub-paragraph (a) of paragraph 8 of Schedule III" in versions[3].text


def test_rule_89_manual_timeline_covers_all_flagged_instructions() -> None:
    processed = load_processed_records(ROOT / "data" / "processed" / "provisions.jsonl")
    reviewed = _attach_rule_sources(
        _manual_rule89_versions(processed),
        ROOT / "data" / "processed" / "amendments.jsonl",
    )
    versions = sorted(
        (record for record in reviewed if record.path == "CGST/r89"),
        key=lambda record: record.effective_from,
    )

    assert len(versions) == 9
    assert [row.effective_from for row in versions] == [
        date(2021, 5, 18),
        date(2021, 9, 24),
        date(2022, 1, 1),
        date(2022, 7, 5),
        date(2022, 10, 1),
        date(2022, 12, 26),
        date(2023, 8, 4),
        date(2024, 7, 10),
        date(2024, 10, 8),
    ]
    assert all(row.verified and row.verified_by == "manual" for row in versions)
    assert all(row.source_url for row in versions)
    assert all((ROOT / row.source_url).is_file() for row in versions)
    assert versions[3].effective_to == date(2022, 9, 30)
    assert versions[6].effective_to == date(2024, 7, 9)
    assert versions[-2].effective_to == date(2024, 10, 7)
    assert "sub-rules (4A) or (4B)" not in versions[-1].text


@pytest.fixture(scope="module")
def act_build() -> tuple[list, dict]:
    return build_act_core_provisions(_read_pdf_text(ACT_PDF))


@pytest.fixture(scope="module")
def core_records() -> list:
    return build_core_provisions()[0]


def _as_of(records: list, path: str, on: date):
    matches = [
        record
        for record in records
        if record.path == path
        and record.effective_from <= on
        and (record.effective_to is None or on <= record.effective_to)
    ]
    assert len(matches) <= 1, f"overlapping versions for {path} on {on}"
    return matches[0] if matches else None


def test_section_17_5_a_text_as_of_either_side_of_2019_amendment(act_build) -> None:
    records, _ = act_build
    before = _as_of(records, "CGST/s17/ss5/cl(a)", date(2019, 1, 31))
    after = _as_of(records, "CGST/s17/ss5/cl(a)", date(2019, 2, 1))

    assert "motor vehicles and other conveyances" in before.text
    assert "approved seating capacity of not more than thirteen persons" in after.text
    assert "approved seating capacity" not in before.text
    # (aa) and (ab) were created by the same substitution, so they do not exist before it.
    for path in ("CGST/s17/ss5/cl(aa)", "CGST/s17/ss5/cl(ab)"):
        assert _as_of(records, path, date(2019, 1, 31)) is None
        assert _as_of(records, path, date(2019, 2, 1)) is not None


def test_act_version_chains_are_contiguous_and_cover_manifest(act_build) -> None:
    records, _ = act_build
    manifest = yaml.safe_load((ROOT / "eval" / "core_provisions.yaml").read_text(encoding="utf-8"))
    paths = {path for row in manifest["act_provisions"] for path in row["paths"]}
    assert paths == {record.path for record in records}
    for path in paths:
        chain = sorted((r for r in records if r.path == path), key=lambda r: r.effective_from)
        assert [r.id for r in chain] == [f"{path}@v{n}" for n in range(1, len(chain) + 1)]
        assert chain[-1].effective_to is None
        for earlier, later in zip(chain, chain[1:]):
            assert (later.effective_from - earlier.effective_to).days == 1
        assert all("[" not in r.text and "]" not in r.text for r in chain)


def test_section_16_footnotes_extract_marker_instrument_date_and_old_text(act_build) -> None:
    _, footnotes = act_build
    notes = {note["marker"]: note for note in footnotes["16"]}

    assert set(notes) == {str(n) for n in range(1, 9)}
    assert notes["1"]["operation"] == "inserted"
    assert notes["1"]["effective_from"] == "2022-01-01"
    assert notes["1"]["instrument"] == "s. 109 of The Finance Act, 2021 (No. 13 of 2021)"
    assert notes["6"]["operation"] == "substituted"
    assert notes["6"]["instrument"] == "s. 100 of The Finance Act 2022 (No. 6 of 2022)"
    assert notes["6"]["old_text"] == (
        "due date of furnishing of the return under section 39 for the month of September"
    )
    assert notes["7"]["old_text"] == "invoice relating to such"
    assert notes["5"]["old_text"] == "or section 43A"
    # "w.e.f. a date yet to be notified" and the order-only footnote carry no date.
    assert notes["4"]["effective_from"] is None
    assert notes["8"]["effective_from"] is None


def test_section_16_4_chain_matches_footnotes_and_2020_consolidation(act_build) -> None:
    records, _ = act_build
    v1 = _as_of(records, "CGST/s16/ss4", date(2020, 9, 30))
    v2 = _as_of(records, "CGST/s16/ss4", date(2021, 1, 1))
    v3 = _as_of(records, "CGST/s16/ss4", date(2022, 10, 1))

    assert "such invoice or invoice relating to such debit note" in v1.text
    assert "for the month of September following" in v2.text
    assert "invoice relating to such" not in v2.text
    assert "thirtieth day of November following" in v3.text
    # Independent check: the CBIC consolidation as on 2020-09-30 shows the v1 wording.
    consolidated = " ".join(
        _read_pdf_text(ROOT / "data" / "raw" / "cgst-act_2020-09-30.pdf").split()
    )
    assert v1.text.removeprefix("(4) ") in consolidated

    assert _as_of(records, "CGST/s16/ss4/proviso1", date(2018, 12, 30)) is None
    proviso = _as_of(records, "CGST/s16/ss4/proviso1", date(2018, 12, 31))
    assert proviso.amended_by.startswith("Order No. 02/2018-Central Tax (Central Goods and")
    assert proviso.source_url == "data/raw/order-02-2018-ct_2018-12-31.pdf"
    assert proviso.verified and proviso.verified_by == "manual"


def test_core_rules_keep_engine_provenance_and_manual_marks(core_records) -> None:
    rules = [record for record in core_records if record.doc_type == "rule"]
    assert all(record.verified for record in rules)
    for record in rules:
        rule = record.path.split("/")[1]
        expected = "manual" if rule in {"r42", "r43", "r86B", "r89"} else "engine"
        assert record.verified_by == expected, record.id


def test_rule_89_rule_10b_words_apply_from_the_38_2021_appointed_date(core_records) -> None:
    words = "may file, subject to the provisions of rule 10B, an application"
    for record in (r for r in core_records if r.path == "CGST/r89"):
        sub_rule_1 = record.text[: record.text.find("(1A)")]
        assert (words in sub_rule_1) == (record.effective_from >= date(2022, 1, 1)), record.id
    v2 = _as_of(core_records, "CGST/r89", date(2021, 12, 31))
    assert "(1A) Any person, claiming refund under section 77" in v2.text
    v3 = _as_of(core_records, "CGST/r89", date(2022, 1, 1))
    assert v3.amended_by.endswith("(commenced by Notification No. 38/2021-Central Tax)")
    assert v3.source_url == "data/raw/ct-38-2021_2021-12-21.pdf"


def test_rule_86b_clause_f_follows_clause_e_not_inside_it(core_records) -> None:
    v1 = _as_of(core_records, "CGST/r86B", date(2026, 1, 31))
    v2 = _as_of(core_records, "CGST/r86B", date(2026, 2, 1))
    clause_e = "(e) the registered person is – (i) Government Department;"
    assert clause_e in v1.text and clause_e in v2.text
    assert "rule 31D" not in v1.text
    assert re.search(
        r"\(iv\) a statutory body: \(f\) the registered person other than a manufacturer", v2.text
    )
    assert v2.text.index("(f) the registered person") < v2.text.index("Provided further")
    assert v2.verified_by == "manual"

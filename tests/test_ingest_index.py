from datetime import date

from ingest.index import build_provision_records, load_processed_records, write_processed_records
from retrieval.baseline import retrieve


def test_build_provision_records_creates_versioned_records_from_fixture() -> None:
    sample = """Section 16. Input tax credit
(1) Every registered person shall be entitled to take credit of input tax.
(a) The input tax credit shall include tax charged on inward supplies.
Provided that this credit shall not be available for certain blocked supplies.
"""

    records = build_provision_records(sample, act_id="CGST")

    assert any(record.path == "CGST/s16" for record in records)
    assert any(record.path == "CGST/s16/ss1" for record in records)
    assert any(record.path == "CGST/s16/ss1/cl(a)/proviso1" for record in records)
    assert all(record.effective_from == date(2017, 7, 1) for record in records)


def test_write_and_load_processed_records_round_trip(tmp_path) -> None:
    records = build_provision_records("Section 17. Blocked credits\n(1) Example blocked credit text.\n")
    output = tmp_path / "provisions.jsonl"
    write_processed_records(records, output)

    loaded = load_processed_records(output)
    assert len(loaded) == 2
    assert loaded[0].path == "CGST/s17"


def test_retrieval_works_against_processed_records() -> None:
    records = build_provision_records("Section 17. Blocked credits\n(1) Example blocked credit text.\n")
    ranked = retrieve("Section 17 blocked credits", records, limit=5)

    assert ranked[0].path == "CGST/s17/ss1"

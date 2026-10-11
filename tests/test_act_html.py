"""CBIC section pages as the current Act source: loading, the PDF diff, post-2022 versions."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from ingest.act_diff import run as run_diff
from ingest.act_html import latest_snapshot, load_pages
from ingest.act_post_pdf import PENDING_REASONS, build_post_pdf, check_pdf_chains_still_current
from ingest.core_provisions import ACT_PDF, _read_pdf_text, build_act_core_provisions
from ingest.index import load_processed_records
from retrieval.provisions import get_provision

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "processed" / "core_provisions.jsonl"


@pytest.fixture(scope="module")
def pages():
    return load_pages()


@pytest.fixture(scope="module")
def pdf_text():
    return _read_pdf_text(ACT_PDF)


@pytest.fixture(scope="module")
def post(pages, pdf_text):
    return build_post_pdf(pages, pdf_text)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_snapshot_pages_match_their_index_and_cover_the_act(pages) -> None:
    index = json.loads((latest_snapshot() / "index.json").read_text(encoding="utf-8"))
    assert len(pages) == index["count"] == 193  # load_pages also re-checks every sha256
    assert {"1", "16", "17", "174", "11A", "Schedule III"} <= set(pages)


def test_page_text_keeps_footnote_markers_and_one_line_per_block(pages) -> None:
    text = pages["16"].text()
    assert re.search(r"(?m)^\* ?Section 16\. Eligibility and conditions", text)
    assert "6[thirtieth day of November]" in _flat(text)
    assert re.search(r"(?m)^\*Enforced", text)


def test_pdf_built_core_chains_are_unchanged_on_the_cbic_pages(pages, pdf_text) -> None:
    records, _ = build_act_core_provisions(pdf_text)
    check_pdf_chains_still_current(pages, records)  # raises on any difference


def test_every_difference_from_the_2022_pdf_is_a_post_2022_footnote(pages, pdf_text) -> None:
    report = run_diff(pages, pdf_text)
    statuses = {s["number"]: s["status"] for s in report["sections"]}
    assert "needs_review" not in statuses.values(), [
        n for n, s in statuses.items() if s == "needs_review"
    ]
    # Sections the PDF parse does not yield: inserted after 2022, the Schedules (no
    # "Section N." header), and 168A (its PDF header is not matched).
    assert {n for n, s in statuses.items() if s == "missing_in_pdf"} == {
        "11A",
        "74A",
        "122A",
        "122B",
        "128A",
        "148A",
        "158A",
        "168A",
        "Schedule I",
        "Schedule II",
        "Schedule III",
    }
    core = {s["number"]: s for s in report["sections"] if s["number"] in {"9", "16", "17"}}
    assert {n: s["status"] for n, s in core.items()} == {
        "9": "explained",
        "16": "explained",
        "17": "explained",
    }


def test_post_2022_versions_follow_the_footnotes(post) -> None:
    records, _, _ = post
    by_id = {record.id: record for record in records}
    proviso2 = by_id["CGST/s16/ss2/proviso2@v1"], by_id["CGST/s16/ss2/proviso2@v2"]
    assert proviso2[0].effective_to == date(2023, 9, 30)
    assert proviso2[1].effective_from == date(2023, 10, 1)
    assert "added to his output tax liability, along with interest thereon" in proviso2[0].text
    assert "paid by him along with interest payable under section 50" in proviso2[1].text
    assert "s. 138 of The Finance Act 2023" in proviso2[1].amended_by
    assert "made by him of the amount" in by_id["CGST/s16/ss2/proviso3@v1"].text
    assert "made by him to the supplier of the amount" in by_id["CGST/s16/ss2/proviso3@v2"].text
    explanation = by_id["CGST/s17/ss3/explanation1@v1"], by_id["CGST/s17/ss3/explanation1@v2"]
    assert explanation[0].effective_from == date(2019, 2, 1)
    assert "except those specified in paragraph 5 of the said Schedule" in explanation[0].text
    assert "clause (a) of paragraph 8 of the said Schedule" in explanation[1].text
    assert by_id["CGST/s17/ss5/cl(fa)@v1"].effective_from == date(2023, 10, 1)
    assert "corporate social responsibility" in by_id["CGST/s17/ss5/cl(fa)@v1"].text
    assert by_id["CGST/s17/ss5/explanation1@v2"].text.startswith("Explanation 1.")
    assert by_id["CGST/s17/ss5/explanation2@v1"].effective_from == date(2025, 10, 1)
    # Pre-2022 versions cite the PDF they were checked against; later ones the CBIC page.
    for record in records:
        source = record.source_url
        if record.effective_from < date(2022, 9, 28):
            assert source == f"data/raw/{ACT_PDF.name}", record.id
        else:
            assert source.startswith("data/raw/cgst-act-sections_cbic-taxinfo_"), record.id


def test_undatable_amendments_are_pending_not_guessed(post) -> None:
    records, pending, _ = post
    assert {(row["section"], row["marker"]) for row in pending} == set(PENDING_REASONS)
    built = {record.path for record in records}
    for row in pending:
        for path in row["path"].split(", "):
            assert path not in built, path
    assert all(row["reason"] and row["needs"] for row in pending)


def test_written_core_file_serves_the_post_2022_versions() -> None:
    core = load_processed_records(CORE)
    assert "paid by him" in get_provision("CGST/s16/ss2/proviso2", "2023-10-01", records=core).text
    assert "added to his" in get_provision("CGST/s16/ss2/proviso2", "2023-09-30", records=core).text
    assert get_provision("CGST/s17/ss5/cl(fa)", "2023-09-30", records=core) is None
    assert get_provision("CGST/s17/ss5/cl(fa)", "2023-10-01", records=core) is not None
    # Knowledge date before the amendment came into force: the old text is what was known.
    served = get_provision(
        "CGST/s16/ss2/proviso2", "2024-01-01", knowledge_date="2023-06-01", records=core
    )
    assert "added to his output tax liability" in served.text

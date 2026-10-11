from datetime import date

from gst_copilot.models import ProvisionRecord
from retrieval.baseline import retrieve
from retrieval.provisions import get_provision


def _record(path: str, heading: str, text: str) -> ProvisionRecord:
    return ProvisionRecord(
        id=f"{path}@v1",
        doc_type="act",
        act_id="CGST2017",
        path=path,
        node_type="section",
        heading=heading,
        text=text,
        effective_from=date(2017, 7, 1),
        effective_to=None,
        in_force_since=date(2017, 7, 1),
    )


def test_exact_match_prefers_section_17_subsection_reference() -> None:
    records = [
        _record("CGST/s17", "Input tax credit", "General rule for input tax credit under the Act."),
        _record("CGST/s17/ss5", "Blocked credits", "Input tax credit is blocked for certain items under section 17(5)."),
        _record("CGST/s16", "Availability of credit", "Section 16 governs the availability of credit."),
    ]

    results = retrieve("Section 17(5) blocked credits", records, limit=3)

    assert results[0].path == "CGST/s17/ss5"
    assert results[0].matched_by == "exact_match"


def test_bm25_falls_back_for_relevant_topic_query() -> None:
    records = [
        _record("CGST/s17", "Input tax credit", "Rules for input tax credit and blocked credits in GST."),
        _record("CGST/s18", "Eligibility and conditions", "Other credit provisions."),
        _record("CGST/s16", "Availability of credit", "Section 16 explains availability of credit."),
    ]

    results = retrieve("blocked credits input tax credit", records, limit=2)

    assert results[0].path == "CGST/s17"
    assert results[0].matched_by == "bm25"


def test_get_provision_excludes_unverified_versions_by_default() -> None:
    unverified = _record(
        "CGST/r8",
        "Application for registration",
        "Unverified amended text.",
    ).model_copy(update={"verified": False})

    assert get_provision("CGST/r8", date(2022, 1, 1), records=[unverified]) is None
    included = get_provision(
        "CGST/r8",
        date(2022, 1, 1),
        records=[unverified],
        include_unverified=True,
    )
    assert included is not None
    assert included.verified is False

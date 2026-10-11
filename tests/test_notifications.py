from pathlib import Path
from types import SimpleNamespace

import pytest

import ingest.notifications as notifications
from ingest.notifications import parse_notification_text, read_pdf_text

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _target_string(entry: dict[str, object]) -> str:
    rule = str(entry["rule"])
    subrule = entry.get("subrule")
    clause = entry.get("clause")
    if subrule and clause:
        return f"{rule}({subrule})({clause})"
    if subrule:
        return f"{rule}({subrule})"
    if clause:
        return f"{rule}({clause})"
    return rule


@pytest.mark.parametrize(
    ("notification_files", "expected_rows"),
    [
        (
            [
                "ct-13-2021_2021-05-01.pdf",
                "ct-27-2021_2021-06-01.pdf",
                "ct-40-2021_2021-12-29.pdf",
                "ct-19-2022_2022-09-28.pdf",
                "ct-12-2024_2024-07-10.pdf",
                "ct-20-2024_2024-10-08.pdf",
            ],
            [
                ("13/2021", "36(4)", "2021-05-01"),
                ("27/2021", "36(4)", "2021-06-01"),
                ("40/2021", "36(4)", "2022-01-01"),
                ("19/2022", "36(2)", "2022-10-01"),
                ("19/2022", "36(4)(b)", "2022-10-01"),
                ("12/2024", "36(4)(a)", "2024-07-10"),
                ("20/2024", "36(3)", "2024-10-08"),
            ],
        ),
        (
            ["ct-20-2025_2025-12-31.pdf"],
            [("20/2025", "86B", "2026-02-01")],
        ),
    ],
)
def test_notification_timeline_snapshots(notification_files: list[str], expected_rows: list[tuple[str, str, str]]) -> None:
    rows: list[tuple[str, str, str]] = []
    for fname in notification_files:
        path = RAW_DIR / fname
        parsed = parse_notification_text(read_pdf_text(path), source_file=path.name)
        for entry in parsed["amendments"]:
            row = (parsed["notification_number"], _target_string(entry), entry["effective_date"])
            if row[1].startswith("36") or row[1].startswith("86B"):
                rows.append(row)

    assert rows == expected_rows


def test_rule36_substitution_keeps_nested_clauses_in_one_instruction() -> None:
    path = RAW_DIR / "ct-40-2021_2021-12-29.pdf"
    parsed = parse_notification_text(read_pdf_text(path), source_file=path.name)
    rule_36 = [
        entry
        for entry in parsed["amendments"]
        if entry["rule"] == "36" and entry["subrule"] == "4"
    ]

    assert len(rule_36) == 1
    assert "(a) the details of such invoices" in rule_36[0]["instruction"]
    assert "(b) the details of such invoices" in rule_36[0]["instruction"]
    assert "FORM GSTR-2B" in rule_36[0]["instruction"]


def test_notification_35_2021_keeps_each_rule_amendment_instruction_scoped() -> None:
    path = RAW_DIR / "ct-35-2021_2021-09-24.pdf"
    parsed = parse_notification_text(read_pdf_text(path), source_file=path.name)
    expected_starts = {
        "23": "(3) In rule 23",
        "45": "(4) In rule 45",
        "59": "(5) In rule 59",
        "89": "(6) In rule 89",
    }
    by_rule = {
        rule: [
            entry
            for entry in parsed["amendments"]
            if entry["rule"] == rule
        ]
        for rule in expected_starts
    }

    assert {rule: len(entries) for rule, entries in by_rule.items()} == {
        "23": 1,
        "45": 3,
        "59": 2,
        "89": 2,
    }
    for rule, entries in by_rule.items():
        for entry in entries:
            assert str(entry["instruction"]).startswith(expected_starts[rule])
            assert "3. For refund under rule 96" not in str(entry["instruction"])


def test_read_pdf_text_prefers_pdftotext_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.touch()
    calls: list[list[str]] = []
    monkeypatch.setattr(notifications, "_find_pdftotext", lambda: "pdftotext.exe")

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["check"] is True
        assert kwargs["encoding"] == "utf-8"
        return SimpleNamespace(stdout="Quoted \u2015 text \u2016")

    monkeypatch.setattr(notifications.subprocess, "run", run)

    assert read_pdf_text(pdf_path) == "Quoted \u2015 text \u2016"
    assert calls == [["pdftotext.exe", "-layout", "-enc", "UTF-8", str(pdf_path), "-"]]


def test_read_pdf_text_rejects_replacement_character_without_pdftotext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.touch()
    monkeypatch.setattr(notifications, "_find_pdftotext", lambda: None)
    monkeypatch.setattr(
        notifications,
        "PdfReader",
        lambda _path: SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda: "Unrecognized \ufffd glyph")]
        ),
    )

    with pytest.raises(RuntimeError, match="Poppler pdftotext is not installed"):
        read_pdf_text(pdf_path)


def test_read_pdf_text_rejects_replacement_character_from_pdftotext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.touch()
    monkeypatch.setattr(notifications, "_find_pdftotext", lambda: "pdftotext.exe")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="Unrecognized \ufffd glyph"),
    )

    with pytest.raises(ValueError, match="pdftotext produced replacement characters"):
        read_pdf_text(pdf_path)

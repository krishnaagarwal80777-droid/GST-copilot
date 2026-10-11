"""Index rows for retrieval: verified provision versions plus rate-notification schedule text.

Every row carries the bitemporal fields (effective_from, effective_to, in_force_since) and
act_id, path, node_type, doc_type, so the store can filter by reference date. A row's
``record_id`` is the provision version (``<path>@v<N>``); long texts (rate schedules) are split
into several rows sharing one record_id, and ranking deduplicates by record_id.

Sources:
- data/processed/core_provisions.jsonl: hand-verified Act and Rule versions (authoritative
  for the paths it covers).
- data/processed/provisions.jsonl: engine rule versions with verified=True, for rule paths
  not in the core build. Act records in this file come from the synthetic segmentation
  fixture (data/raw/cgst_sample_fixture.txt) and are never indexed.
- Rate notifications 1/2017 and 9/2025 (goods schedules) and 2/2017 and 10/2025 (goods
  exemptions), as issued. 9/2025 and 10/2025 supersede 1/2017 and 2/2017 from 2025-09-22.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from gst_copilot.models import ProvisionRecord
from ingest.index import load_processed_records

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CORE_FILE = ROOT / "data" / "processed" / "core_provisions.jsonl"
DISTRACTOR_FILE = ROOT / "data" / "processed" / "distractors.jsonl"
ENGINE_FILE = ROOT / "data" / "processed" / "provisions.jsonl"
ACT_PDF = RAW / "cgst-act_cbic-taxinfo_2022-09-28.pdf"
RULES_TITLE = "Central Goods and Services Tax Rules, 2017"
ACT_TITLE = "Central Goods and Services Tax Act, 2017"


@dataclass(frozen=True)
class IndexRow:
    id: str
    record_id: str
    path: str
    act_id: str
    doc_type: str
    node_type: str
    effective_from: date
    effective_to: date | None
    in_force_since: date
    text: str
    servable: bool = True


@dataclass(frozen=True)
class RateNotification:
    number: str  # "9/2025"
    filename: str
    effective_from: date
    effective_to: date | None
    schedule_rates: dict[str, str]  # roman numeral -> rate in the notification's own words


# Commencement and supersession dates are those stated in the notifications themselves.
RATE_NOTIFICATIONS = (
    RateNotification(
        "1/2017",
        "ctr-1-2017_2017-06-28.pdf",
        date(2017, 7, 1),
        date(2025, 9, 21),
        {"I": "2.5", "II": "6", "III": "9", "IV": "14", "V": "1.5", "VI": "0.125"},
    ),
    RateNotification(
        "9/2025",
        "ctr-09-2025_2025-09-17.pdf",
        date(2025, 9, 22),
        None,
        {"I": "2.5", "II": "9", "III": "20", "IV": "1.5", "V": "0.125", "VI": "0.75", "VII": "14"},
    ),
    RateNotification(
        "2/2017", "ctr-2-2017_2017-06-28.pdf", date(2017, 7, 1), date(2025, 9, 21), {}
    ),
    RateNotification("10/2025", "ctr-10-2025_2025-09-17.pdf", date(2025, 9, 22), None, {}),
)


def _flat(value: str) -> str:
    return " ".join(value.split())


def citation_label(path: str) -> str:
    """ "CGST/s17/ss5/cl(a)" -> "section 17(5)(a)"; "CGST/r36/ss4" -> "rule 36(4)"."""
    parts = path.split("/")[1:]
    head, rest = parts[0], parts[1:]
    label = ("section " if head.startswith("s") else "rule ") + head[1:]
    for part in rest:
        if part.startswith("ss"):
            label += f"({part[2:]})"
        elif part.startswith("cl("):
            label += part[2:]
        elif part.startswith("proviso"):
            label += f", proviso {part[7:]}"
        elif part.startswith("explanation"):
            label += f", Explanation {part[11:]}"
    return label


def _act_section_headings() -> dict[str, str]:
    from pypdf import PdfReader

    flat = _flat("\n".join(page.extract_text() or "" for page in PdfReader(str(ACT_PDF)).pages))
    headings: dict[str, str] = {}
    for match in re.finditer(r"\*?\s*Section\s+(\d+[A-Z]?)\.\s*(.+?)\s*[.]?-\s*\(1\)", flat):
        headings.setdefault(match.group(1), match.group(2).rstrip("."))
    return headings


def _rule_heading(text: str) -> str:
    match = re.match(r"\s*\d+[A-Z]*\.\s*(.+?)\s*\.?\s*[-–—]", text)
    return match.group(1).strip() if match else ""


def provision_rows() -> list[IndexRow]:
    core = [
        record
        for record in load_processed_records(CORE_FILE)
        if record.verified and record.text.strip()
    ]
    core_paths = {record.path for record in core}
    engine = [
        record
        for record in load_processed_records(ENGINE_FILE)
        if record.verified
        and record.doc_type == "rule"
        and record.path not in core_paths
        and record.text.strip()
    ]
    section_headings = _act_section_headings()
    rule_headings: dict[str, str] = {}
    all_records = load_processed_records(CORE_FILE) + load_processed_records(ENGINE_FILE)
    for record in all_records:
        if record.doc_type == "rule" and record.path.count("/") == 1:
            heading = record.heading or _rule_heading(record.text)
            if heading:
                rule_headings.setdefault(record.path, heading)

    distractors = load_processed_records(DISTRACTOR_FILE)
    if not distractors:
        raise FileNotFoundError(f"{DISTRACTOR_FILE} is missing; run `make ingest` first")
    rows: list[IndexRow] = []
    for record in core + engine + distractors:
        rows.append(_provision_row(record, section_headings, rule_headings))
    return rows


def _provision_row(
    record: ProvisionRecord, section_headings: dict[str, str], rule_headings: dict[str, str]
) -> IndexRow:
    top = "/".join(record.path.split("/")[:2])
    if record.doc_type == "act":
        source = ACT_TITLE
        heading = record.heading or section_headings.get(top.split("/s")[-1], "")
    else:
        source = RULES_TITLE
        heading = record.heading or rule_headings.get(top, "")
    prefix = f"{source}, {citation_label(record.path)}" + (f" - {heading}" if heading else "")
    return IndexRow(
        id=record.id,
        record_id=record.id,
        path=record.path,
        act_id=record.act_id,
        doc_type=record.doc_type,
        node_type=record.node_type,
        effective_from=record.effective_from,
        effective_to=record.effective_to,
        in_force_since=record.in_force_since,
        text=f"{prefix}: {_flat(record.text)}",
        servable=record.servable,
    )


def notification_text(filename: str) -> str:
    """Layout text of a raw notification PDF (pdftotext keeps schedule columns apart)."""
    from ingest.notifications import _find_pdftotext

    pdftotext = _find_pdftotext()
    if pdftotext is None:
        raise RuntimeError("pdftotext is required to read rate schedules")
    result = subprocess.run(
        [pdftotext, "-layout", "-enc", "UTF-8", str(RAW / filename), "-"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _schedule_sections(text: str) -> list[tuple[str, str]]:
    """Split a rate notification into (schedule numeral, text) by its "Schedule X – N%" headers."""
    headers = list(re.finditer(r"Schedule\s*[-–]?\s*(?P<num>[IVX]+)\s*[–-]\s*[\d.]+\s*%", text))
    sections = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        sections.append((header.group("num"), text[header.start() : end]))
    return sections


def _blocks(text: str, lines_per_block: int = 30, overlap: int = 5) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    step = lines_per_block - overlap
    return [
        "\n".join(lines[start : start + lines_per_block])
        for start in range(0, max(len(lines) - overlap, 1), step)
    ]


def rate_rows() -> list[IndexRow]:
    rows: list[IndexRow] = []
    for notification in RATE_NOTIFICATIONS:
        text = notification_text(notification.filename)
        label = f"Notification No. {notification.number}-Central Tax (Rate)"
        slug = notification.number.replace("/", "-")
        if notification.schedule_rates:
            sections = _schedule_sections(text)
            found = {num for num, _ in sections}
            if found != set(notification.schedule_rates):
                raise ValueError(
                    f"{label}: schedules {sorted(found)} in text, expected "
                    f"{sorted(notification.schedule_rates)}"
                )
            parts = [
                (
                    f"CTR/{slug}/sch{num}",
                    f"{label}, Schedule {num} - central tax rate "
                    f"{notification.schedule_rates[num]} per cent",
                    body,
                )
                for num, body in sections
            ]
        else:
            parts = [(f"CTR/{slug}", f"{label}, exemption schedule (nil central tax)", text)]
        for path, prefix, body in parts:
            record_id = f"{path}@v1"
            for index, block in enumerate(_blocks(body)):
                rows.append(
                    IndexRow(
                        id=f"{record_id}#{index}",
                        record_id=record_id,
                        path=path,
                        act_id="CTR",
                        doc_type="notification",
                        node_type="paragraph",
                        effective_from=notification.effective_from,
                        effective_to=notification.effective_to,
                        in_force_since=notification.effective_from,
                        text=f"{prefix}: {_flat(block)}",
                    )
                )
    return rows


def build_rows() -> list[IndexRow]:
    rows = provision_rows() + rate_rows()
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate index row ids")
    return rows

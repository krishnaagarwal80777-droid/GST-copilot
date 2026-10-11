"""Row-level parse of the goods rate schedules (Notifications 1/2017 and 9/2025-CT(Rate)).

Each row is (schedule, S. No., column (2) tariff entry, column (3) description, rate). Text
comes from ``pdftotext -raw`` (reading order), not ``-layout``: in 9/2025 the layout mode
interleaves the S. No., tariff and description columns. A row starts on a line ``<n>. ``
where ``n`` is the next serial number in its schedule, so numbered text inside a description
or in the Lists appended to Schedule I cannot start a row. Parsing of a schedule stops at the
first appended List or at the explanatory paragraphs that follow the last schedule.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from retrieval.corpus import RATE_NOTIFICATIONS, RAW, RateNotification

_SCHEDULE = re.compile(r"^Schedule\s*[-–]?\s*(?P<num>[IVX]+)\s*[–-]\s*[\d.]+\s*%")
_ROW = re.compile(r"^(?P<sno>\d+)\.(?:\s+(?P<rest>.*))?$")
# Column (2): tariff codes ("0101 21 00, 0101 29", "01012100,"), "(Except 050790)", "Any Chapter".
_TARIFF = re.compile(
    r"^(?:[Aa]ny\s+[Cc]hapter|(?:\d{1,8}(?:\s\d{2}){0,3}"
    r"(?:\s*[([](?:[Ee]xcept|other than)[^)\]]*[)\]])?"  # "(Except 050790)", "[except ...]"
    r"(?:\s*(?:,|or|and)\s*|\s+|$))+)"
)
_HEADER_WORDS = {
    "S.",
    "No.",
    "S. No.",
    "Chapter /",
    "Chapter/",
    "Heading /",
    "Sub-heading /",
    "Tariff item",
    "item",
    "(1) (2) (3)",
}
# Lists appended to Schedule I, and the notification-level explanation after the last schedule
# (row-level "Explanation.- For the purposes of this entry" stays part of its row).
_STOP = re.compile(
    r"^(?:List\s+\d|\(B\)\s|(?i:Explanation\.?\s*[-–—:]*\s*For the purposes? of this notification))"
)


@dataclass(frozen=True)
class RateRow:
    notification: str  # "9/2025"
    schedule: str  # roman numeral
    s_no: int
    tariff: str  # column (2) as printed, whitespace-normalised
    description: str  # column (3), whitespace-normalised
    rate: str  # central tax rate, in the notification's own words ("2.5")

    @property
    def path(self) -> str:
        """Index path of the schedule record that contains this row (retrieval.corpus)."""
        return f"CTR/{self.notification.replace('/', '-')}/sch{self.schedule}"

    @property
    def tariff_key(self) -> str:
        """Tariff entry with spaces inside codes removed: '0101 21 00' == '01012100'."""
        return re.sub(r"(?<=\d)\s+(?=\d)", "", self.tariff).replace(" ", "")


def raw_text(filename: str) -> str:
    from ingest.notifications import _find_pdftotext

    pdftotext = _find_pdftotext()
    if pdftotext is None:
        raise RuntimeError("pdftotext is required to read rate schedules")
    return subprocess.run(
        [pdftotext, "-raw", "-enc", "UTF-8", str(RAW / filename), "-"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def _is_noise(line: str) -> bool:
    if not line or (line.isdigit() and len(line) <= 3):  # blank or page number
        return True
    if line in _HEADER_WORDS or line.startswith(("Chapter /", "Chapter/", "S. No. Chapter")):
        return True
    return line.startswith(("Description of", "heading", "Heading /", "Sub-heading"))


def parse_rows(notification: RateNotification, text: str | None = None) -> list[RateRow]:
    if not notification.schedule_rates:
        return []
    text = raw_text(notification.filename) if text is None else text
    rows: list[RateRow] = []
    schedule: str | None = None
    expected = 1
    current: dict | None = None
    stopped = False

    def flush() -> None:
        nonlocal current
        if current is not None:
            rows.append(_finish(notification, current))
            current = None

    for raw in text.splitlines():
        line = " ".join(raw.split())
        header = _SCHEDULE.match(line)
        if header and header.group("num") in notification.schedule_rates:
            flush()
            schedule, expected, stopped = header.group("num"), 1, False
            continue
        if schedule is None or stopped or _is_noise(line):
            continue
        if _STOP.match(line):
            flush()
            stopped = True
            continue
        start = _ROW.match(line)
        if start and int(start.group("sno")) == expected:
            flush()
            current = {"schedule": schedule, "s_no": expected, "lines": [start.group("rest") or ""]}
            expected += 1
        elif current is not None:
            current["lines"].append(line)
    flush()
    return rows


def _finish(notification: RateNotification, current: dict) -> RateRow:
    body = " ".join(current["lines"]).strip()
    tariff_match = _TARIFF.match(body)
    tariff = tariff_match.group(0).strip(" ,") if tariff_match else ""
    description = body[tariff_match.end() :].strip() if tariff_match else body
    return RateRow(
        notification=notification.number,
        schedule=current["schedule"],
        s_no=current["s_no"],
        tariff=tariff,
        description=description,
        rate=notification.schedule_rates[current["schedule"]],
    )


def all_rows() -> dict[str, list[RateRow]]:
    return {n.number: parse_rows(n) for n in RATE_NOTIFICATIONS if n.schedule_rates}

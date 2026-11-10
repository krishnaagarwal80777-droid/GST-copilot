"""Knowledge horizons: the last date each source type in the corpus can speak for.

A question whose reference date is after the horizon of its source type cannot be answered
from the corpus, however closely a provision matches it, so generation.answer abstains before
the LLM is called (GenerationConfig.date_gate).

- Act: date of the newest Act source held (eval.harness.act_source_horizon).
- Rules: issue date of the newest notification in data/processed/amendments.jsonl.
- Rate notifications: the day before the earliest amendment in data/raw to an indexed rate
  notification still in force (retrieval.corpus.RATE_NOTIFICATIONS) whose changes are not
  applied to the index rows; with none, the commencement date of the newest ctr-*.pdf.

Keys are index doc_type values (retrieval.corpus), so a new source type is new data here.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AMENDMENTS_FILE = ROOT / "data" / "processed" / "amendments.jsonl"
SOURCE_LABELS: dict[str, str] = {
    "act": "the Act",
    "rule": "the Rules",
    "notification": "rate notifications",
}
# Preamble: "... makes the following further amendments in the notification of the Government
# of India, ..., No. 9/2025- Central Tax (Rate), ..."; closing note when the preamble is not in
# the text layer: "The principal notification number 13/2017 -Central Tax (Rate) ...".
_RATE_NOTIFICATION = (
    r"(?:No\.?|number)\s*(\d{1,3})\s*/\s*(\d{4})\s*[-–]\s*Central\s+Tax\s*\(\s*Rate\s*\)"
)
_AMENDS = re.compile(
    r"amendments?\b.{0,40}?\b(?:in|amend)\s+the\s+notification\b.{0,250}?" + _RATE_NOTIFICATION,
    re.IGNORECASE,
)
_PRINCIPAL = re.compile(r"principal\s+notification\s+" + _RATE_NOTIFICATION, re.IGNORECASE)
_PREAMBLE_CHARS = 1500
_FILE_NUMBER = re.compile(r"^ctr-(\d+)-(\d{4})_")


def act_horizon() -> date:
    from eval.harness import act_source_horizon

    return act_source_horizon()


def rules_horizon(path: Path = AMENDMENTS_FILE) -> date:
    """Issue date of the newest notification parsed into amendments.jsonl."""
    with path.open(encoding="utf-8") as handle:
        issued = [json.loads(line)["issue_date"] for line in handle if line.strip()]
    if not issued:
        raise ValueError(f"{path} has no notifications")
    return date.fromisoformat(max(issued))


def _read_rate_notification(path: Path) -> tuple[date, date | None, str]:
    from ingest.notifications import (
        _extract_issue_date,
        _extract_notification_commencement_date,
        _fallback_issue_date_from_filename,
        read_pdf_text,
    )

    text = read_pdf_text(path)
    ocr = path.with_suffix(".ocr.txt")  # OCR of a PDF with no text layer (data/raw/manifest.json)
    if not text.strip() and ocr.is_file():
        text = ocr.read_text(encoding="utf-8")
    issued = _extract_issue_date(text, _fallback_issue_date_from_filename(path.name))
    return issued, _extract_notification_commencement_date(text, issued), text


def rate_notification_dates(path: Path) -> tuple[date, date | None]:
    """(issue date, commencement date stated in the notification or None) of a raw
    rate-notification PDF."""
    issued, commencement, _ = _read_rate_notification(path)
    return issued, commencement


def _number(text: str) -> tuple[int, int]:
    """ "09/2025" -> (9, 2025)."""
    num, year = text.split("/")
    return int(num), int(year)


@dataclass(frozen=True)
class RateAmendment:
    number: str  # as in the file name: "19/2025", "01/2026"
    filename: str
    amends: tuple[int, int]  # (number, year) of the amended notification
    effective: date  # commencement stated in it, else its issue date ("immediate effect")


def amended_notification(text: str) -> tuple[int, int] | None:
    """(number, year) of the rate notification a raw notification amends, or None when it is
    not an amendment (or its text does not say)."""
    flat = " ".join(text.split())
    match = _AMENDS.search(flat[:_PREAMBLE_CHARS]) or _PRINCIPAL.search(flat)
    return (int(match.group(1)), int(match.group(2))) if match else None


def rate_amendments(raw_dir: Path = RAW) -> tuple[list[RateAmendment], list[str]]:
    """Amendments to rate notifications among data/raw/ctr-*.pdf, and the files that could not
    be read (no text layer), which are therefore not checked."""
    amendments: list[RateAmendment] = []
    unreadable: list[str] = []
    for path in sorted(raw_dir.glob("ctr-*.pdf")):
        issued, commencement, text = _read_rate_notification(path)
        if not text.strip():
            unreadable.append(path.name)
            continue
        target = amended_notification(text)
        number = _FILE_NUMBER.match(path.name)
        if target is None or number is None:
            continue
        amendments.append(
            RateAmendment(
                f"{number.group(1)}/{number.group(2)}", path.name, target, commencement or issued
            )
        )
    return amendments, unreadable


def rate_horizon_details(
    raw_dir: Path = RAW, notifications: Sequence | None = None
) -> tuple[date, str]:
    """(rates horizon, reason). Only amendments to indexed notifications still in force count:
    an amendment to a superseded notification (e.g. to 1/2017 before 9/2025 replaced it) is a
    gap inside past dates, not a limit on how far forward the index speaks."""
    if notifications is None:
        from retrieval.corpus import RATE_NOTIFICATIONS

        notifications = RATE_NOTIFICATIONS
    in_force = {
        _number(n.number): {_number(a) for a in n.applied_amendments}
        for n in notifications
        if n.effective_to is None
    }
    amendments, unreadable = rate_amendments(raw_dir)
    if unreadable:
        # TODO: no text layer, so whether these amend an indexed schedule is unknown; read
        # them (OCR or another source) so the horizon accounts for them.
        logger.warning("rate horizon: %s not readable, not checked", ", ".join(unreadable))
    pending = sorted(
        (
            a
            for a in amendments
            if a.amends in in_force and _number(a.number) not in in_force[a.amends]
        ),
        key=lambda a: (a.effective, a.filename),
    )
    if pending:
        names = ", ".join(a.number for a in pending)
        return pending[0].effective - timedelta(days=1), f"amendments {names} not applied to index"
    dated = [(rate_notification_dates(path), path) for path in sorted(raw_dir.glob("ctr-*.pdf"))]
    if not dated:
        raise FileNotFoundError(f"no rate notifications (ctr-*.pdf) in {raw_dir}")
    (_, commencement), latest = max(dated, key=lambda entry: entry[0][0])
    if commencement is None:
        raise ValueError(f"{latest.name}: no commencement date found")
    return commencement, f"all amendments applied; newest rate notification {latest.name}"


def rates_horizon(raw_dir: Path = RAW) -> date:
    horizon, reason = rate_horizon_details(raw_dir)
    logger.warning("rate notifications horizon %s: %s", horizon.isoformat(), reason)
    return horizon


@lru_cache(maxsize=1)
def knowledge_horizons() -> dict[str, date]:
    """doc_type -> last date the corpus covers for that source type."""
    return {"act": act_horizon(), "rule": rules_horizon(), "notification": rates_horizon()}

"""Knowledge horizons: the last date each source type in the corpus can speak for.

A question whose reference date is after the horizon of its source type cannot be answered
from the corpus, however closely a provision matches it, so generation.answer abstains before
the LLM is called (GenerationConfig.date_gate).

- Act: date of the newest Act source held (eval.harness.act_source_horizon).
- Rules: issue date of the newest notification in data/processed/amendments.jsonl.
- Rate notifications: commencement date stated in the newest ctr-*.pdf in data/raw.

Keys are index doc_type values (retrieval.corpus), so a new source type is new data here.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AMENDMENTS_FILE = ROOT / "data" / "processed" / "amendments.jsonl"
SOURCE_LABELS: dict[str, str] = {
    "act": "the Act",
    "rule": "the Rules",
    "notification": "rate notifications",
}


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


def rate_notification_dates(path: Path) -> tuple[date, date | None]:
    """(issue date, commencement date stated in the notification or None) of a raw
    rate-notification PDF."""
    from ingest.notifications import (
        _extract_issue_date,
        _extract_notification_commencement_date,
        _fallback_issue_date_from_filename,
        read_pdf_text,
    )

    text = read_pdf_text(path)
    issued = _extract_issue_date(text, _fallback_issue_date_from_filename(path.name))
    return issued, _extract_notification_commencement_date(text, issued)


def rates_horizon(raw_dir: Path = RAW) -> date:
    """Commencement date of the rate notification in data/raw with the latest issue date."""
    dated = [(rate_notification_dates(path), path) for path in sorted(raw_dir.glob("ctr-*.pdf"))]
    if not dated:
        raise FileNotFoundError(f"no rate notifications (ctr-*.pdf) in {raw_dir}")
    (_, commencement), latest = max(dated, key=lambda entry: entry[0][0])
    if commencement is None:
        raise ValueError(f"{latest.name}: no commencement date found")
    return commencement


@lru_cache(maxsize=1)
def knowledge_horizons() -> dict[str, date]:
    """doc_type -> last date the corpus covers for that source type."""
    return {"act": act_horizon(), "rule": rules_horizon(), "notification": rates_horizon()}

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from gst_copilot.models import ProvisionRecord
from ingest.index import load_processed_records


def get_provision(
    path: str,
    on: date | str,
    *,
    knowledge_date: date | str | None = None,
    records: Sequence[ProvisionRecord] | None = None,
    include_unverified: bool = False,
) -> ProvisionRecord | None:
    """Return the known version for a reference date.

    If a version effective on the reference date was not yet known, return the
    latest known version that had already begun by that date.
    """
    reference_date = _as_date(on, name="on")
    known_by = _as_date(knowledge_date, name="knowledge_date") if knowledge_date else date.today()
    provisions = records if records is not None else load_processed_records()
    path_records = [
        record
        for record in provisions
        if record.path == path
        and record.in_force_since <= known_by
        and (include_unverified or record.verified)
    ]
    matches = [
        record
        for record in path_records
        if record.effective_from <= reference_date
        and (record.effective_to is None or reference_date <= record.effective_to)
    ]
    if not matches:
        prior_as_known = [
            record for record in path_records if record.effective_from <= reference_date
        ]
        if prior_as_known:
            matches = [max(prior_as_known, key=lambda record: record.effective_from)]
    if len(matches) > 1:
        raise ValueError(
            f"Overlapping provision versions found for {path!r} on {reference_date.isoformat()}"
        )
    return matches[0] if matches else None


def _as_date(value: date | str, *, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}") from error

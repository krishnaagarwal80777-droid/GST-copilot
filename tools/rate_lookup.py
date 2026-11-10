"""Central tax rate of goods on a date, from the indexed rate-schedule rows.

    rate_lookup("ferro-alloys", on=date(2025, 12, 1))
    rate_lookup(hsn="7202", on=date(2025, 12, 1))

Rows are the schedule entries of the indexed rate notifications (retrieval.corpus.
RATE_NOTIFICATIONS, parsed by ingest.rate_schedule), each valid between its notification's
commencement and supersession dates. Only rows valid on ``on`` are searched.

Matching never guesses:
- ``hsn``: the rows whose tariff code is a prefix of the HSN (the most specific code wins);
  if none covers it, the rows whose code starts with the HSN. "Any Chapter" rows have no code
  and are never matched by HSN.
- ``query``: the rows whose description contains every content word of the query (lower
  case, simple plural folding). A row whose whole description equals the query wins.
More than one remaining row returns status "ambiguous" with every candidate and no rate.

After the rates horizon (generation.horizon) the lookup is refused with the date-gate
message. ``rate`` is the central tax rate as notified; ``combined_rate`` (CGST + SGST, or
IGST) is twice that, on the assumption that the State tax rate equals the central rate (the
State GST Acts and the IGST rate notifications are not in the index).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from functools import lru_cache

from tools.result import ToolResult

MAX_CANDIDATES = 15
STOPWORDS = frozenset(
    "a an and any are as at be by for from in including into is it of on or other than "
    "that the their this to under used with whether not such rate tax gst central what "
    "goods supply".split()
)


@dataclass(frozen=True)
class RateEntry:
    id: str  # index row id: "CTR/9-2025/schII@v1#261"
    notification: str  # "9/2025"
    schedule: str  # roman numeral; "" for an exemption notification
    s_no: int
    tariff: str
    description: str
    rate: str  # central tax rate as printed ("9"), or "nil" if exempt
    effective_from: date
    effective_to: date | None

    def valid_on(self, day: date) -> bool:
        return self.effective_from <= day and (
            self.effective_to is None or day <= self.effective_to
        )

    @property
    def codes(self) -> list[str]:
        key = re.sub(r"(?<=\d)\s+(?=\d)", "", self.tariff)
        return re.findall(r"\d{2,8}", key)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "notification": self.notification,
            "schedule": self.schedule or "exemption",
            "s_no": self.s_no,
            "tariff": self.tariff,
            "description": self.description[:200],
            "rate": self.rate,
        }


@lru_cache(maxsize=1)
def rate_entries() -> tuple[RateEntry, ...]:
    """Every indexed schedule row with its validity dates (parsed from data/raw)."""
    from ingest.rate_schedule import parse_rows
    from retrieval.corpus import RATE_NOTIFICATIONS

    entries = []
    for notification in RATE_NOTIFICATIONS:
        for row in parse_rows(notification):
            entries.append(
                RateEntry(
                    id=f"{row.path}@v1#{row.s_no}",
                    notification=row.notification,
                    schedule=row.schedule,
                    s_no=row.s_no,
                    tariff=row.tariff,
                    description=row.description,
                    rate=row.rate,
                    effective_from=notification.effective_from,
                    effective_to=notification.effective_to,
                )
            )
    return tuple(entries)


def words(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS or (len(word) < 3 and not word.isdigit()):
            continue
        if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def _by_hsn(hsn: str, rows: list[RateEntry]) -> tuple[list[RateEntry], str]:
    digits = re.sub(r"\D", "", hsn)
    if len(digits) < 2:
        raise ValueError(f"HSN must have at least 2 digits: {hsn!r}")
    covering = [
        (max(len(c) for c in r.codes if digits.startswith(c)), r)
        for r in rows
        if any(digits.startswith(c) for c in r.codes)
    ]
    if covering:
        best = max(length for length, _ in covering)
        return [r for length, r in covering if length == best], (
            f"HSN {digits}: rows whose tariff code is a prefix of it, most specific code "
            f"({best} digits) kept"
        )
    narrower = [r for r in rows if any(c.startswith(digits) for c in r.codes)]
    return narrower, f"HSN {digits}: no row covers it; rows whose tariff code starts with it"


def _by_query(query: str, rows: list[RateEntry]) -> tuple[list[RateEntry], str]:
    wanted = words(query)
    if not wanted:
        raise ValueError(f"query has no content words: {query!r}")
    matches = [r for r in rows if wanted <= words(r.description)]
    exact = [r for r in matches if words(r.description) == wanted]
    if len(exact) == 1:
        return (
            exact,
            f"description words {sorted(wanted)}: one row's description is exactly these words",
        )
    return (
        matches,
        f"description words {sorted(wanted)}: rows whose description contains all of them",
    )


def _combined(rate: str) -> str:
    if rate.strip().lower() == "nil":
        return "0"
    return f"{(Decimal(rate) * 2).normalize():f}"


def rate_lookup(
    query: str | None = None,
    *,
    hsn: str | None = None,
    on: date,
    entries: Sequence[RateEntry] | None = None,
    horizon: date | None = None,
) -> ToolResult:
    """Central tax rate of the goods described by ``query`` or classified under ``hsn``."""
    if (query is None) == (hsn is None):
        raise ValueError("give exactly one of query or hsn")
    if horizon is None:
        from generation.horizon import knowledge_horizons

        horizon = knowledge_horizons()["notification"]
    if on > horizon:
        from generation.answer import ABSTAIN_TEXT
        from generation.horizon import SOURCE_LABELS

        message = ABSTAIN_TEXT["horizon"].format(
            ref_date=on.isoformat(),
            horizon=horizon.isoformat(),
            source=SOURCE_LABELS["notification"],
        )
        return ToolResult(tool="rate_lookup", status="refused", notes=[message])

    pool = [e for e in (rate_entries() if entries is None else entries) if e.valid_on(on)]
    notifications = sorted({e.notification for e in pool})
    steps = [
        f"Searched {len(pool)} schedule rows in force on {on.isoformat()} "
        f"(Notifications {', '.join(notifications)}-Central Tax (Rate))."
    ]
    found, how = _by_hsn(hsn, pool) if hsn is not None else _by_query(query, pool)
    steps.append(f"Matched by {how}: {len(found)} row(s).")
    if not found:
        return ToolResult(tool="rate_lookup", status="not_found", steps=steps)
    if len(found) > 1:
        notes = [
            "More than one row matches, so no rate is given; narrow the description or "
            "give the HSN."
        ]
        if len(found) > MAX_CANDIDATES:
            notes.append(f"{len(found)} rows match; the first {MAX_CANDIDATES} are listed.")
        return ToolResult(
            tool="rate_lookup",
            status="ambiguous",
            steps=steps,
            candidates=[e.summary() for e in found[:MAX_CANDIDATES]],
            notes=notes,
        )
    entry = found[0]
    label = f"Notification No. {entry.notification}-Central Tax (Rate)"
    where = f"Schedule {entry.schedule}" if entry.schedule else "the exemption schedule"
    combined = _combined(entry.rate)
    steps.append(
        f"{label}, {where}, S. No. {entry.s_no} ({entry.tariff} {entry.description[:120]}): "
        f"central tax rate {entry.rate}{'' if entry.rate == 'nil' else ' per cent'}."
    )
    steps.append(f"Combined rate (CGST + SGST, or IGST) = 2 x {entry.rate} = {combined} per cent.")
    return ToolResult(
        tool="rate_lookup",
        status="ok",
        result={
            "rate": entry.rate,
            "combined_rate": combined,
            "schedule": entry.schedule or "exemption",
            "s_no": entry.s_no,
            "description": entry.description,
            "tariff": entry.tariff,
            "notification": entry.notification,
        },
        steps=steps,
        citations=[entry.id],
        notes=[
            "rate is the central tax (CGST) rate as notified. combined_rate assumes the State "
            "tax rate equals it; the State GST Acts and the IGST rate notifications are not in "
            "the index."
        ],
    )

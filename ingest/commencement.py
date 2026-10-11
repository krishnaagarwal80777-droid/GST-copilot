"""Resolve instructions whose commencement is deferred to a later notification.

The notification parser marks instructions "with effect from the date as may be notified" /
"from a date to be notified" (and whole notifications that "come into force on such date as the
Central Government may ... appoint") as ``commencement: "pending"`` with no ``effective_date``.
A commencement notification supplies the date, either in prose

    "... hereby notifies the 1st day of January, 2022, as the date from which the provisions of
    sub-rule (2), ..., clause (i) of sub-rule (6) and sub-rule (7) of rule 2 of the said rules,
    shall come into force."                                   (38/2021, for 35/2021)

or as a table of paragraphs and dates (09/2025, for 12/2024). Scope is matched by the
paragraph labels of the target notification's amending rules, never by the CGST rule number.
Until resolved, the version engines skip pending instructions, so nothing built from them is
ever served. Superseded notifications (51/2023 "in supersession of" 45/2023) never build.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

from pydantic import BaseModel, ConfigDict


class Appointment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notification_number: str
    source_file: str
    appointed_date: date
    target_notification: str
    # Paragraph labels of the target notification, e.g. ("6", "i") for clause (i) of sub-rule
    # (6) of rule 2 of 35/2021, or ("38", "ii") for clause (ii) of rule 38 of 12/2024. An empty
    # sub-label covers the whole paragraph.
    scope: tuple[tuple[str, str], ...]


class UnresolvedScopeError(ValueError):
    pass


_DATE = (
    r"\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December),?\s+\d{4}"
)
_PROSE = re.compile(
    r"hereby\s+(?:appoints|notifies)\s+the\s+(?P<date>.+?)\s*,?\s+as\s+the\s+date\s+"
    r"(?:on|from)\s+which\s+the\s+provisions?\s+of\s+(?P<scope>.+?)\s*,?\s+(?:shall|will)\s+"
    r"come\s+into\s+force",
    re.IGNORECASE | re.DOTALL,
)
_TABLE = re.compile(
    r"hereby\s+appoints\s+the\s+following\s+dates\s+as\s+mentioned\s+in\s+column\s*\(\d\)\s+of"
    r"\s+the\s+table\s+below\s*,?\s+on\s+which\s+the\s+provisions\s+of\s+rules\s+specified",
    re.IGNORECASE,
)
_NOTIFICATION_REF = re.compile(
    r"(?:notification\s+)?No\.?\s*(?P<number>\d{1,3}/\d{4})\s*[-–—]\s*Central\s+Tax",
    re.IGNORECASE,
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def is_commencement_text(text: str) -> bool:
    flat = _flat(text)
    return _PROSE.search(flat) is not None or _TABLE.search(flat) is not None


def _target_notification(flat: str, own_number: str, start: int) -> str | None:
    """The notification being commenced, named in the preamble or scope; never the notice itself."""
    for match in _NOTIFICATION_REF.finditer(flat[: start + 2000]):
        if match.group("number") != own_number:
            return match.group("number")
    return None


def commences_notification(text: str, own_number: str) -> bool:
    """True when the appointment commences a notification (rather than provisions of an Act)."""
    flat = _flat(text)
    match = _PROSE.search(flat) or _TABLE.search(flat)
    if match is None:
        return False
    return _target_notification(flat, own_number, match.start()) is not None


def _prose_scope_labels(scope: str) -> tuple[tuple[str, str], ...]:
    """'sub-rule (2), clause (i) of sub-rule (6) and sub-rule (7) of rule 2' -> sub-rule labels."""
    labels: list[tuple[str, str]] = []
    nested = (
        r"clauses?\s+(?P<subs>\([ivx]+\)(?:\s*(?:,|and)\s*\([ivx]+\))*)\s+of\s+"
        r"(?:sub-?\s*rule|paragraph|clause)\s+\((?P<para>\d+)\)"
    )
    for match in re.finditer(nested, scope, re.IGNORECASE):
        for sub in re.findall(r"\(([ivx]+)\)", match.group("subs")):
            labels.append((match.group("para"), sub.lower()))
    remainder = re.sub(nested, " ", scope, flags=re.IGNORECASE)
    for match in re.finditer(
        r"(?:sub-?\s*rules?|paragraphs?|clauses?)\s+(?P<paras>\(\d+\)(?:\s*(?:,|and)\s*\(\d+\))*)",
        remainder,
        re.IGNORECASE,
    ):
        labels.extend((para, "") for para in re.findall(r"\((\d+)\)", match.group("paras")))
    return tuple(dict.fromkeys(labels))


def _table_scope_labels(row: str) -> tuple[tuple[str, str], ...]:
    """'Rules 8, 37 and clause (ii) of rule 38' -> amending-rule (paragraph) labels."""
    labels: list[tuple[str, str]] = []
    nested = r"clauses?\s+\((?P<sub>[ivx]+)\)\s+of\s+rule\s+(?P<para>\d+)"
    for match in re.finditer(nested, row, re.IGNORECASE):
        labels.append((match.group("para"), match.group("sub").lower()))
    remainder = re.sub(nested, " ", row, flags=re.IGNORECASE)
    listed = re.search(r"\brules?\s+(?P<list>\d+(?:\s*(?:,|and)\s*\d+)*)", remainder, re.I)
    if listed:
        labels = [(para, "") for para in re.findall(r"\d+", listed.group("list"))] + labels
    return tuple(dict.fromkeys(labels))


def _table_rows(text: str) -> list[tuple[str, date]]:
    """Pair each table row's "Rules ..." cell with the dates in column order.

    Layout text can print a row's date beside the previous row, so rows and dates are matched
    by order; a count mismatch raises instead of guessing.
    """
    from ingest.notifications import parse_legal_date

    start = text.find("Table")
    body = text[start if start >= 0 else 0 :]
    end = re.search(r"\[\s*(?:F\.?\s*)?No\.?\s*CBIC", body, re.IGNORECASE)
    body = body[: end.start()] if end else body
    dates = [parse_legal_date(_flat(m.group())) for m in re.finditer(_DATE, body, re.IGNORECASE)]
    without_dates = re.sub(_DATE, "\n", body, flags=re.IGNORECASE)
    rows = [
        _flat(m.group("rules"))
        for m in re.finditer(
            r"(?m)^\s*\d+\s+(?P<rules>Rules?\s+[^\n]+)$", without_dates, re.IGNORECASE
        )
    ]
    if not rows or len(rows) != len(dates):
        raise UnresolvedScopeError(
            f"commencement table has {len(rows)} rule rows but {len(dates)} dates"
        )
    return list(zip(rows, dates, strict=True))


def parse_commencement_text(
    text: str, *, notification_number: str, source_file: str
) -> list[Appointment]:
    from ingest.notifications import parse_legal_date

    flat = _flat(text)
    table = _TABLE.search(flat)
    if table is not None:
        target = _target_notification(flat, notification_number, table.start())
        if target is None:
            return []
        appointments = []
        for row, appointed in _table_rows(text):
            labels = _table_scope_labels(row)
            if not labels:
                raise UnresolvedScopeError(f"{source_file}: could not read table row {row!r}")
            appointments.append(
                Appointment(
                    notification_number=notification_number,
                    source_file=source_file,
                    appointed_date=appointed,
                    target_notification=target,
                    scope=labels,
                )
            )
        return appointments

    appointments = []
    for match in _PROSE.finditer(flat):
        target = _target_notification(flat, notification_number, match.start())
        if target is None:
            # Commences provisions of an Act (e.g. a Finance Act), not a rule-amending
            # notification; Act versions come from the CBIC footnotes, not from here.
            continue
        labels = _prose_scope_labels(match.group("scope"))
        if not labels:
            raise UnresolvedScopeError(
                f"{source_file}: could not map commencement scope to paragraphs: "
                f"{match.group('scope')!r}"
            )
        appointments.append(
            Appointment(
                notification_number=notification_number,
                source_file=source_file,
                appointed_date=parse_legal_date(match.group("date")),
                target_notification=target,
                scope=labels,
            )
        )
    return appointments


def instruction_label(instruction: str) -> tuple[str, str]:
    """("6", "i") for "(6) In rule 89 of the said rules, - (i) in sub-rule (1), ..."."""
    flat = _flat(instruction)
    para = re.match(r"\(?(\d+)\)?\s*\.?\s", flat)
    sub = re.search(r"[-–—:]\s*\(([ivx]+)\)\s", flat)
    return (para.group(1) if para else "", sub.group(1).lower() if sub else "")


def resolve_pending(
    notifications: list[dict[str, object]],
    appointments: Iterable[Appointment],
) -> list[dict[str, object]]:
    """Give pending instructions the appointed date of the commencement notification covering them.

    Mutates and returns ``notifications`` (amendments.jsonl rows). Scope labels that match no
    rule instruction (e.g. paragraphs amending forms) are recorded on the commencing record as
    ``unmatched_scope``; an appointment that matches nothing at all raises.
    """
    by_number = {str(row.get("notification_number")): row for row in notifications}
    for appointment in appointments:
        target = by_number.get(appointment.target_notification)
        if target is None:
            raise UnresolvedScopeError(
                f"{appointment.source_file}: commences notification "
                f"{appointment.target_notification}, which is not in data/raw"
            )
        matched: set[tuple[str, str]] = set()
        for amendment in target.get("amendments", []):  # type: ignore[union-attr]
            if amendment.get("commencement") != "pending":
                continue
            if amendment.get("pending_reason") not in (None, "awaiting_notification"):
                continue
            para, sub = instruction_label(str(amendment.get("instruction", "")))
            label = (para, sub) if (para, sub) in appointment.scope else (para, "")
            if label in appointment.scope:
                amendment["effective_date"] = appointment.appointed_date.isoformat()
                amendment["commencement"] = "resolved"
                amendment["pending_reason"] = None
                amendment["commenced_by"] = appointment.notification_number
                matched.add(label)
        if not matched:
            raise UnresolvedScopeError(
                f"{appointment.source_file}: scope {appointment.scope} matched no pending "
                f"instruction of notification {appointment.target_notification}"
            )
        commencing = by_number.get(appointment.notification_number)
        for para, sub in appointment.scope:
            if (para, sub) not in matched and commencing is not None:
                commencing.setdefault("unmatched_scope", []).append(  # type: ignore[union-attr]
                    {
                        "target": appointment.target_notification,
                        "paragraph": para + (f"({sub})" if sub else ""),
                    }
                )
    return notifications


def _title_key(title: object) -> tuple[str, str] | None:
    """("third", "2023") for '... (Third Amendment) Rules, 2023' in either title spelling."""
    if not isinstance(title, str):
        return None
    match = re.search(
        r"\((?P<ordinal>[\w\s]*?)Amendment\)\s+Rules\s*,\s*(?P<year>\d{4})", title, re.IGNORECASE
    )
    return (match.group("ordinal").strip().lower(), match.group("year")) if match else None


def link_supersessions(notifications: list[dict[str, object]]) -> list[dict[str, object]]:
    """Mark notifications superseded by a later one ("in supersession of the ... Rules, YYYY").

    Every instruction of a superseded notification gets ``commencement: "superseded"`` so it
    never builds a version, even if a commencement notification later names it.
    """
    for row in notifications:
        key = _title_key(row.get("supersedes_title"))
        if key is None:
            continue
        superseded = [
            other
            for other in notifications
            if other is not row
            and _title_key(other.get("short_title")) == key
            and str(other["issue_date"]) <= str(row["issue_date"])
        ]
        if len(superseded) != 1:
            raise UnresolvedScopeError(
                f"{row['source_file']}: supersedes {row['supersedes_title']!r} but "
                f"{len(superseded)} notifications in data/raw carry that title"
            )
        old = superseded[0]
        old["superseded_by"] = row["notification_number"]
        for amendment in old.get("amendments", []):  # type: ignore[union-attr]
            amendment["commencement"] = "superseded"
            amendment["effective_date"] = None
            amendment["pending_reason"] = None
    return notifications

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from gst_copilot.models import ProvisionRecord
from ingest.index import DEFAULT_PROCESSED_FILE, load_processed_records, write_processed_records
from ingest.rule_versions import BASE_DATE, RULES_PDF, _read_pdf_text, build_rule_36_provisions

ROOT = Path(__file__).resolve().parents[1]
AMENDMENTS_PATH = ROOT / "data" / "processed" / "amendments.jsonl"
REPORT_PATH = ROOT / "data" / "processed" / "rule_version_report.json"
RULE_HEADING = re.compile(
    r"(?m)(?=^[ \t]*(?:\[\s*)?(?P<number>\d+[A-Z]?)(?:\.[ \t]*|(?<=[A-Z])[ \t]+)"
    r"(?P<heading>(?=[A-Za-z])(?:(?!\n[ \t]*\n)[\s\S]){3,300}?)"
    r"(?=[ \t]*\.\s*[-–—]))"
)


class AmendmentApplicationError(ValueError):
    """An instruction could not be applied uniquely to the available base text."""


def _normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\bFORM\s*GSTR\s*-\s*(\d[A-Z]?)", r"FORM GSTR-\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsub\s*-\s*(section|rule)\b", r"sub-\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bdet\s+ails\b", "details", value, flags=re.IGNORECASE)
    value = re.sub(r"\bun\s+der\b", "under", value, flags=re.IGNORECASE)
    return value


def _extract_rule_bases(text: str) -> dict[str, tuple[str, str]]:
    matches = [
        match
        for match in RULE_HEADING.finditer(text)
        if not re.match(
            r"(?:Substituted|Inserted|Omitted|Deleted)\s+vide\b",
            match.group("heading"),
            re.IGNORECASE,
        )
    ]
    unique_matches: list[re.Match[str]] = []
    seen: set[str] = set()
    duplicate_starts: dict[str, list[int]] = {}
    for match in matches:
        number = match.group("number")
        duplicate_starts.setdefault(number, []).append(match.start())
        if number not in seen:
            unique_matches.append(match)
            seen.add(number)

    bases: dict[str, tuple[str, str]] = {}
    for index, match in enumerate(unique_matches):
        end = (
            unique_matches[index + 1].start()
            if index + 1 < len(unique_matches)
            else len(text)
        )
        repeated = [
            position
            for position in duplicate_starts[match.group("number")]
            if position > match.start()
        ]
        if repeated:
            end = min(end, repeated[0])
        heading = _normalize_text(match.group("heading")).strip(" .-")
        body = text[match.start("number") : end]
        body = re.sub(r"(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$", "", body, flags=re.IGNORECASE)
        body = re.sub(
            r"(?m)^\s*\d+\s*(?:Inserted|Substituted|Subsitituted)\b.*$",
            "",
            body,
            flags=re.IGNORECASE,
        )
        body = _normalize_text(body)
        initial_subrules = [
            node_start
            for marker, node_start, _ in _node_spans(body, "sub-rule")
            if marker == "1"
        ]
        if len(initial_subrules) > 1:
            body = body[: initial_subrules[1]]
        bases[match.group("number")] = (heading, body)
    return bases


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Amendment ledger not found: {path}")
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            notification = json.loads(line)
            for amendment in notification.get("amendments", []):
                if amendment.get("commencement") not in (None, "fixed", "resolved"):
                    # Pending (no appointed or parseable date) or superseded: never builds a
                    # version (see ingest.commencement).
                    continue
                value = amendment.get("effective_date") or notification.get("effective_date")
                try:
                    effective_date = date.fromisoformat(value)
                    issue_date = date.fromisoformat(notification["issue_date"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid amendment date on amendments.jsonl line {line_number}"
                    ) from error
                events.append(
                    {
                        "notification_number": notification.get("notification_number"),
                        "issue_date": issue_date,
                        "effective_date": effective_date,
                        "rule": str(amendment.get("rule")),
                        "subrule": amendment.get("subrule"),
                        "instruction": str(amendment.get("instruction", "")),
                        "operation": amendment.get("operation") or "amend",
                    }
                )
    return sorted(
        events,
        key=lambda event: (
            event["effective_date"],
            event["rule"],
            event["issue_date"],
            str(event["notification_number"]),
        ),
    )


def _read_pending(path: Path) -> list[dict[str, Any]]:
    """Instructions awaiting a commencement notification; reported, never applied."""
    pending: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            notification = json.loads(line)
            for amendment in notification.get("amendments", []):
                if amendment.get("commencement") in ("pending", "superseded"):
                    pending.append(
                        {
                            "notification_number": notification.get("notification_number"),
                            "status": amendment.get("commencement"),
                            "reason": amendment.get("pending_reason")
                            or (
                                f"superseded by {notification.get('superseded_by')}"
                                if amendment.get("commencement") == "superseded"
                                else None
                            ),
                            "rule": str(amendment.get("rule")),
                            "instruction": re.sub(
                                r"\s+", " ", str(amendment.get("instruction", ""))
                            )[:500],
                        }
                    )
    return pending


def _normalize_quote_glyphs(value: str) -> str:
    return value.translate(
        str.maketrans(
            {
                "\u2015": '"',
                "\u2016": '"',
                "\u2017": '"',
                "\u2018": '"',
                "\u2019": '"',
                "\u201c": '"',
                "\u201d": '"',
            }
        )
    )


def _quoted_phrases(instruction: str) -> list[str]:
    instruction = _normalize_quote_glyphs(instruction)
    quote_positions = [match.start() for match in re.finditer('"', instruction)]
    phrases = [
        instruction[start + 1 : end]
        for start, end in zip(quote_positions[::2], quote_positions[1::2])
    ]
    return [
        _normalize_text(phrase.strip(" ,;:"))
        for phrase in phrases
        if _normalize_text(phrase.strip(" ,;:"))
    ]


def _replace_unique(text: str, old: str, new: str, *, operation: str) -> str:
    old = _normalize_text(old)
    if not old:
        raise AmendmentApplicationError(f"ambiguous anchor: empty anchor for {operation}")
    pattern = re.compile(re.escape(old).replace(r"\ ", r"\s+"), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        raise AmendmentApplicationError(f"anchor text not found: {old[:90]}")
    if len(matches) != 1:
        raise AmendmentApplicationError(f"ambiguous anchor ({len(matches)} matches): {old[:90]}")
    match = matches[0]
    return text[: match.start()] + new + text[match.end() :]


def _node_span(
    text: str,
    kind: str,
    marker: str | None = None,
    *,
    start: int = 0,
    end: int | None = None,
) -> tuple[int, int]:
    end = len(text) if end is None else end
    spans = _node_spans(text, kind, start=start, end=end)
    if marker is not None:
        spans = [
            span
            for node_marker, node_start, node_end in spans
            if node_marker is not None and node_marker.casefold() == marker.casefold()
            for span in [(node_marker, node_start, node_end)]
        ]
    if len(spans) != 1:
        detail = "not found" if not spans else f"ambiguous ({len(spans)} matches)"
        raise AmendmentApplicationError(
            f"anchor text {detail}: {kind}{f' ({marker})' if marker else ''}"
        )
    return spans[0][1], spans[0][2]


def _node_spans(
    text: str,
    kind: str,
    *,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[str | None, int, int]]:
    end = len(text) if end is None else end
    segment = text[start:end]
    if kind == "sub-rule":
        pattern = re.compile(r"\((?P<marker>\d+[A-Z]?)\)\s*(?=[A-Z(])")
    elif kind == "clause":
        pattern = re.compile(
            r"(?P<boundary>[:;,\u2013\u2014-]|\b(?:or|and)\b|\[)"
            r"\s*(?:or\s+)?\[?\s*\((?P<marker>[ivx]{1,4}a?|[a-z]{1,2})\)\s+",
        )
    elif kind == "proviso":
        pattern = re.compile(
            r"\bProvided(?:\s+(?:further|also|additionally))?\s+that\b",
            re.IGNORECASE,
        )
    else:
        raise AmendmentApplicationError(f"unsupported legal node: {kind}")

    matches = list(pattern.finditer(segment))
    if kind == "clause":
        return _clause_spans(segment, matches, start)
    if kind == "sub-rule":
        matches = [
            match
            for match in matches
            if not re.search(
                r"(?:sub\s*-\s*(?:rule|section)|rule|section)\s*$",
                segment[max(0, match.start() - 28) : match.start()],
                re.IGNORECASE,
            )
        ]
    return [
        (
            match.groupdict().get("marker"),
            start + (match.start() if kind == "proviso" else match.start("marker") - 1),
            start
            + (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(segment)
            ),
        )
        for index, match in enumerate(matches)
    ]


_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv"]


def _clause_successors(marker: str) -> set[str]:
    """Markers that can follow clause (marker) at the same level.

    (e) -> (f) or an inserted (ea); (ba) -> (bb) or (c); roman items (iv) -> (v).
    """
    marker = marker.casefold()
    successors: set[str] = set()
    if len(marker) == 1 and marker < "z":
        successors |= {chr(ord(marker) + 1), marker + "a"}
    elif len(marker) == 2 and marker[1] < "z":
        successors |= {marker[0] + chr(ord(marker[1]) + 1)}
        if marker[0] < "z":
            successors.add(chr(ord(marker[0]) + 1))
    roman = marker[:-1] if marker.endswith("a") and marker[:-1] in _ROMAN else marker
    if roman in _ROMAN[:-1]:
        successors.add(_ROMAN[_ROMAN.index(roman) + 1])
        if roman == marker:
            successors.add(marker + "a")
    return successors


def _clause_spans(
    segment: str, matches: list[re.Match[str]], offset: int
) -> list[tuple[str | None, int, int]]:
    """Clause spans that run to the next sibling, so nested items stay inside their clause.

    Markers like (i), (ii) inside clause (e) also match the clause pattern; ending clause (e)
    at the next match of any kind would cut it off before its own items. A clause ends at the
    next marker of its own sequence, else at the next proviso or Explanation, else at the end
    of the scope.
    """
    spans: list[tuple[str | None, int, int]] = []
    for index, match in enumerate(matches):
        marker = match.group("marker")
        node_start = match.start("marker") - 1
        successors = _clause_successors(marker)
        # End just before the sibling's "(" (or its opening "["), so the separator that closes
        # this clause ("; ", "; or ") stays with it.
        end = next(
            (
                len(segment[: later.start("marker") - 1].rstrip().rstrip("[").rstrip()) + 1
                for later in matches[index + 1 :]
                if later.group("marker") in successors
            ),
            None,
        )
        if end is None:
            boundary = re.search(
                r"\bProvided(?:\s+(?:further|also|additionally))?\s+that\b|\bExplanation\b",
                segment[match.end() :],
            )
            end = match.end() + boundary.start() if boundary else len(segment)
        spans.append((marker, offset + node_start, offset + end))
    return spans


def _instruction_scope(
    text: str, instruction: str, subrule: str | None
) -> tuple[int, int]:
    scope_start, scope_end = 0, len(text)
    explicit_subrule = re.search(
        r"(?:in\s+)?sub\s*-\s*rule\s*\((?P<marker>[^)]+)\)",
        instruction,
        re.IGNORECASE,
    )
    subrule_marker = subrule or (
        explicit_subrule.group("marker") if explicit_subrule else None
    )
    if subrule_marker:
        scope_start, scope_end = _node_span(
            text, "sub-rule", subrule_marker.strip()
        )

    explicit_clause = re.search(
        r"(?:in\s+)?clause\s*\((?P<marker>[a-z]{1,2})\)",
        instruction,
        re.IGNORECASE,
    )
    if explicit_clause:
        scope_start, scope_end = _node_span(
            text,
            "clause",
            explicit_clause.group("marker"),
            start=scope_start,
            end=scope_end,
        )

    if re.search(r"\bproviso\b", instruction, re.IGNORECASE):
        provisos = list(
            re.finditer(
                r"\bProvided(?:\s+(?:further|also|additionally))?\s+that\b",
                text[scope_start:scope_end],
                re.IGNORECASE,
            )
        )
        ordinal = re.search(
            r"\b(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+proviso\b",
            instruction,
            re.IGNORECASE,
        )
        if ordinal:
            order = {
                "first": 1,
                "second": 2,
                "third": 3,
                "fourth": 4,
                "fifth": 5,
            }
            token = ordinal.group(1).lower()
            position = order.get(token)
            if position is None:
                position = int(re.match(r"\d+", token).group())  # type: ignore[union-attr]
            if position > len(provisos):
                raise AmendmentApplicationError("anchor text not found: requested proviso")
            match = provisos[position - 1]
            scope_start += match.start()
            scope_end = (
                scope_start
                + provisos[position].start()
                - match.start()
                if position < len(provisos)
                else scope_end
            )
        elif len(provisos) == 1:
            scope_start += provisos[0].start()
        elif len(provisos) > 1:
            raise AmendmentApplicationError(
                f"ambiguous anchor: proviso ({len(provisos)} matches)"
            )
    return scope_start, scope_end


def _insertion_anchor(instruction: str) -> tuple[str, str] | None:
    match = re.search(
        r"after\s+(?:the\s+)?(?:words(?:,\s*letters\s+and\s+figures)?|word)"
        r'\s*[,:]?\s*"([^"]+)"',
        instruction,
        re.IGNORECASE,
    )
    if match is not None:
        phrases = _quoted_phrases(instruction)
        if len(phrases) >= 2:
            return match.group(1), phrases[-1]
        return None
    match = re.search(
        r"after\s+(?:sub\s*[- ]\s*rule|clause)\s*\((?P<marker>[^)]+)\)",
        instruction,
        re.IGNORECASE,
    )
    if match is not None:
        marker = match.group("marker")
        return f"({marker})", ""
    return None


def _inserted_text(instruction: str) -> str:
    instruction = _normalize_quote_glyphs(instruction)
    match = re.search(r"\bnamely\s*[:\-–—]?\s*", instruction, re.IGNORECASE)
    if match is None:
        raise AmendmentApplicationError("ambiguous anchor: inserted text not found")
    remainder = instruction[match.end() :].lstrip(" \t\r\n:;-–—")
    if '"' not in remainder[:5]:
        raise AmendmentApplicationError("ambiguous anchor: inserted text is not quoted")
    quote_at = remainder.index('"')
    close_at = remainder.find('"', quote_at + 1)
    inserted_end = close_at if close_at >= 0 else len(remainder)
    inserted = remainder[quote_at + 1 : inserted_end].strip()
    if close_at < 0:
        inserted = re.split(
            r"\s+(?:\[F\.\s*No\.|Note:|(?:\([A-Z][^)]+\)\s*)?(?:Under Secretary|Director)\b)",
            inserted,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].rstrip(" \t\r\n.;")
    if not inserted:
        raise AmendmentApplicationError("ambiguous anchor: inserted text is empty")
    return inserted


def _parent_subrule_scope(
    text: str, instruction: str, subrule: str | None
) -> tuple[int, int]:
    explicit = re.search(
        r"\bin\s+(?:the\s+)?sub\s*-\s*rule\s*\((?P<marker>[^)]+)\)",
        instruction,
        re.IGNORECASE,
    )
    marker = subrule or (explicit.group("marker") if explicit else None)
    if marker is None:
        return 0, len(text)
    return _node_span(text, "sub-rule", marker.strip())


def _structural_target_span(
    text: str,
    kind: str,
    marker: str | None,
    instruction: str,
    subrule: str | None,
) -> tuple[int, int]:
    parent_start, parent_end = _parent_subrule_scope(text, instruction, subrule)
    if kind == "sub-rule":
        siblings = _node_spans(text, kind)
    else:
        siblings = _node_spans(text, kind, start=parent_start, end=parent_end)

    if kind == "proviso":
        ordinal = re.search(
            r"\b(first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+proviso\b",
            instruction,
            re.IGNORECASE,
        )
        if ordinal:
            order = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
            token = ordinal.group(1).lower()
            position = order.get(token)
            if position is None:
                position = int(re.match(r"\d+", token).group())  # type: ignore[union-attr]
            siblings = [siblings[position - 1]] if position <= len(siblings) else []
    elif marker is not None:
        siblings = [
            node
            for node in siblings
            if node[0] is not None and node[0].casefold() == marker.casefold()
        ]

    if len(siblings) != 1:
        raise AmendmentApplicationError(
            f"ambiguous anchor: {kind}{f' ({marker})' if marker else ''}"
        )
    return siblings[0][1], siblings[0][2]


def _clause_omission_marker(instruction: str) -> str | None:
    omitted = re.search(
        r"\((?P<marker>[a-z]{1,2})\)\s+shall be omitted\b",
        instruction,
        re.IGNORECASE,
    )
    rule_targets = list(
        re.finditer(r"\bin\s+rule\s+\d+[A-Z]?\b", instruction, re.IGNORECASE)
    )
    if omitted is None or not rule_targets or rule_targets[-1].start() > omitted.start():
        return None
    return omitted.group("marker")


def _apply_structural_instruction(
    text: str, instruction: str, subrule: str | None
) -> str:
    lowered = instruction.lower()
    if "shall be substituted" in lowered:
        target = re.search(
            r"\bfor\s+(?:the\s+)?(?P<kind>sub\s*-\s*rule|clause|proviso)"
            r"(?:\s*\((?P<marker>[^)]+)\))?",
            instruction,
            re.IGNORECASE,
        )
        if target is None:
            raise AmendmentApplicationError("ambiguous anchor: substitution target not found")
        kind = re.sub(r"\s*-\s*", "-", target.group("kind").lower())
        marker = target.group("marker") or (subrule if kind == "sub-rule" else None)
        start, end = _structural_target_span(
            text, kind, marker, instruction, marker if kind == "sub-rule" else subrule
        )
        return text[:start] + _inserted_text(instruction) + text[end:]

    if "shall be omitted" in lowered:
        target = re.search(
            r"\b(?:the\s+)?(?P<kind>sub\s*-\s*rule|clause|proviso)"
            r"(?:\s*\((?P<marker>[^)]+)\))?\s+shall be omitted",
            instruction,
            re.IGNORECASE,
        )
        if target is None:
            marker = _clause_omission_marker(instruction)
            if marker is not None:
                kind = "clause"
            else:
                kind = marker = None
        else:
            kind = re.sub(r"\s*-\s*", "-", target.group("kind").lower())
            marker = target.group("marker") or (subrule if kind == "sub-rule" else None)
        if kind is None:
            raise AmendmentApplicationError("ambiguous anchor: omission target not found")
        start, end = _structural_target_span(
            text, kind, marker, instruction, marker if kind == "sub-rule" else subrule
        )
        return text[:start] + text[end:]

    if "shall be inserted" in lowered:
        target = re.search(
            r"\b(?P<relation>after|before)\s+(?:the\s+)?"
            r"(?:(?P<ordinal>first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+)?"
            r"(?P<kind>sub\s*-\s*rule|clause|proviso)"
            r"(?:\s*\((?P<marker>[^)]+)\))?",
            instruction,
            re.IGNORECASE,
        )
        inserted_kind = re.search(
            r"\bfollowing\s+(?:sub\s*-\s*rule|clause|proviso|explanation)\b",
            instruction,
            re.IGNORECASE,
        )
        if target is None or inserted_kind is None:
            raise AmendmentApplicationError(
                "ambiguous anchor: structural insertion point not found"
            )
        kind = re.sub(r"\s*-\s*", "-", target.group("kind").lower())
        parent_start, parent_end = _parent_subrule_scope(text, instruction, subrule)
        scope_start, scope_end = (
            (0, len(text)) if kind == "sub-rule" else (parent_start, parent_end)
        )
        siblings = _node_spans(text, kind, start=scope_start, end=scope_end)
        ordinal = target.group("ordinal")
        marker = target.group("marker")
        if kind == "proviso" and ordinal:
            order = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
            position = order.get(ordinal.lower())
            if position is None:
                position = int(re.match(r"\d+", ordinal).group())  # type: ignore[union-attr]
            siblings = [siblings[position - 1]] if position <= len(siblings) else []
        elif marker is not None:
            siblings = [
                node
                for node in siblings
                if node[0] is not None and node[0].casefold() == marker.casefold()
            ]
        if len(siblings) != 1:
            raise AmendmentApplicationError(
                f"ambiguous anchor: insertion target {kind}{f' ({marker})' if marker else ''}"
            )
        inserted = _inserted_text(instruction)
        target_start, target_end = siblings[0][1], siblings[0][2]
        if target.group("relation").lower() == "before":
            return text[:target_start] + inserted + " " + text[target_start:]
        before = text[:target_end].rstrip()
        return before + " " + inserted + " " + text[target_end:].lstrip()

    raise AmendmentApplicationError("ambiguous: unsupported structural operation")


def _apply_generic_instruction(
    text: str,
    instruction: str,
    subrule: str | None = None,
    *,
    structural_operations: bool = True,
) -> str:
    instruction = _normalize_quote_glyphs(instruction)
    instruction = re.sub(r"\s+", " ", instruction).strip()
    lowered = instruction.lower()
    phrases = _quoted_phrases(instruction)
    has_node_operation = (
        re.search(
            r"\bfollowing\s+(?:sub\s*-\s*rule|clause|proviso|explanation)\b",
            instruction,
            re.IGNORECASE,
        )
        is not None
        or re.search(
            r"\bfor\s+(?:the\s+)?(?:sub\s*-\s*rule|clause|proviso)\b"
            r".{0,80}\bfollowing\b",
            instruction,
            re.IGNORECASE,
        )
        is not None
        or re.search(
            r"\b(?:the\s+)?(?:sub\s*-\s*rule|clause|proviso)\b.{0,60}\bshall be omitted\b",
            instruction,
            re.IGNORECASE,
        )
        is not None
        or re.search(
            r"\bclause\s*\([a-z]{1,2}\)\s+shall be omitted\b",
            instruction,
            re.IGNORECASE,
        )
        is not None
        or _clause_omission_marker(instruction) is not None
    )
    if structural_operations and has_node_operation and any(
        operation in lowered
        for operation in ("shall be substituted", "shall be inserted", "shall be omitted")
    ):
        return _apply_structural_instruction(text, instruction, subrule)

    scope_start, scope_end = _instruction_scope(text, instruction, subrule)
    scoped_text = text[scope_start:scope_end]

    if "shall be substituted" in lowered:
        if re.search(
            r"\bfor\s+(?:the\s+)?(?:figures,\s*letters\s+and\s+words|words)\b",
            instruction,
            re.IGNORECASE,
        ) or "for and words" in lowered:
            if len(phrases) < 2:
                raise AmendmentApplicationError(
                    "ambiguous anchor: quoted substitution pair not found"
                )
            updated = _replace_unique(
                scoped_text, phrases[0], phrases[1], operation="substitution"
            )
            return text[:scope_start] + updated + text[scope_end:]
        raise AmendmentApplicationError(
            "ambiguous: full sub-rule substitution needs structural matching"
        )

    if "shall be omitted" in lowered:
        if not phrases:
            raise AmendmentApplicationError("ambiguous anchor: quoted omission text not found")
        updated = _replace_unique(
            scoped_text, phrases[0], "", operation="omission"
        )
        return text[:scope_start] + updated + text[scope_end:]

    if "shall be inserted" in lowered:
        anchor = _insertion_anchor(instruction)
        if anchor is None:
            raise AmendmentApplicationError("ambiguous anchor: insertion point could not be parsed")
        anchor_text, inserted_text = anchor
        if not inserted_text:
            raise AmendmentApplicationError(
                "ambiguous: insertion requires legal-structure matching"
            )
        updated = _replace_unique(
            scoped_text,
            anchor_text,
            f"{anchor_text} {inserted_text}",
            operation="insertion",
        )
        return text[:scope_start] + updated + text[scope_end:]

    raise AmendmentApplicationError("ambiguous: unsupported amendment operation")


def _source_observations(text: str) -> list[dict[str, Any]]:
    reader = PdfReader(str(RULES_PDF))
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        match = re.search(r"the\s+such\s+condition", page_text, re.IGNORECASE)
        if match is not None:
            printed_page_match = re.search(r"Page\s+(\d+)\s+of\s+\d+", page_text)
            return [
                {
                    "phrase": "the such condition",
                    "status": "present_in_pdf_text_layer",
                    "source_file": RULES_PDF.name,
                    "pdf_page": page_index,
                    "printed_page": (
                        int(printed_page_match.group(1)) if printed_page_match else None
                    ),
                    "action": "preserved verbatim; not treated as an extraction error",
                }
            ]
    return []


def _base_record(
    number: str,
    heading: str,
    text: str,
    *,
    effective_from: date = BASE_DATE,
    in_force_since: date = BASE_DATE,
    amended_by: str | None = None,
) -> ProvisionRecord:
    path = f"CGST/r{number}"
    return ProvisionRecord(
        id=f"{path}@v1",
        doc_type="rule",
        act_id="CGST2017",
        path=path,
        node_type="rule",
        heading=heading,
        text=text,
        context_prefix=heading,
        effective_from=effective_from,
        effective_to=None,
        in_force_since=in_force_since,
        amended_by=amended_by,
    )


def run_all_rule_versions(
    *,
    raw_pdf: str | Path = RULES_PDF,
    amendments_path: str | Path = AMENDMENTS_PATH,
    output_path: str | Path = DEFAULT_PROCESSED_FILE,
    report_path: str | Path = REPORT_PATH,
    structural_operations: bool = True,
    classify_cascades: bool = True,
) -> dict[str, Any]:
    pdf_path = Path(raw_pdf)
    source_text = _read_pdf_text(pdf_path)
    bases = _extract_rule_bases(source_text)
    events = _read_events(Path(amendments_path))
    versions: dict[str, list[ProvisionRecord]] = {}
    for number, (heading, text) in bases.items():
        base_effective_from = date(2021, 5, 1) if number == "36" else BASE_DATE
        base_amended_by = (
            "Notification No. 13/2021-Central Tax" if number == "36" else None
        )
        base_in_force_since = date(2021, 5, 1) if number == "36" else BASE_DATE
        versions[number] = [
            _base_record(
                number,
                heading,
                text,
                effective_from=base_effective_from,
                in_force_since=base_in_force_since,
                amended_by=base_amended_by,
            )
        ]

    applied: list[dict[str, Any]] = []
    root_failures: list[dict[str, Any]] = []
    cascade_failures: list[dict[str, Any]] = []
    target_rule_not_in_base: list[dict[str, Any]] = []
    new_rules_inserted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    rule36_records = build_rule_36_provisions(raw_pdf, amendments_path)
    for event in events:
        row = {
            "notification_number": event["notification_number"],
            "rule": event["rule"],
            "subrule": event["subrule"],
            "effective_date": event["effective_date"].isoformat(),
            "issue_date": event["issue_date"].isoformat(),
            "instruction": re.sub(r"\s+", " ", event["instruction"])[:500],
        }
        if event["effective_date"] <= BASE_DATE:
            row["status"] = "covered_by_base_snapshot"
            skipped.append(row)
            continue
        if event["rule"] == "36":
            row["status"] = "applied_cleanly"
            applied.append(row)
            continue
        if event["operation"] == "insert_rule":
            # "After rule 10A ..., the following rule shall be inserted": the rule is new, so
            # it is absent from the base snapshot by design, not a failure.
            row["status"] = "new_rule_inserted"
            new_rules_inserted.append(row)
            continue
        path_versions = versions.get(event["rule"])
        if path_versions is None:
            row["status"] = "target_rule_not_in_base"
            row["reason"] = (
                f"target rule not in base snapshot: Rule {event['rule']} may "
                "have been inserted later"
            )
            target_rule_not_in_base.append(row)
            continue

    failed_rules: set[str] = set()
    for (_, rule_number), grouped_events in groupby(
        (
            event
            for event in events
            if event["effective_date"] > BASE_DATE and event["rule"] != "36"
        ),
        key=lambda event: (event["effective_date"], event["rule"]),
    ):
        same_effective_changes = list(grouped_events)
        path_versions = versions.get(rule_number)
        if path_versions is None:
            continue
        current = path_versions[-1]
        updated_text = current.text
        clean_rows: list[dict[str, Any]] = []
        for event in same_effective_changes:
            row = {
                "notification_number": event["notification_number"],
                "rule": event["rule"],
                "subrule": event["subrule"],
                "effective_date": event["effective_date"].isoformat(),
                "issue_date": event["issue_date"].isoformat(),
                "instruction": re.sub(r"\s+", " ", event["instruction"])[:500],
            }
            if classify_cascades and rule_number in failed_rules:
                row["status"] = "cascade_failure"
                row["reason"] = (
                    "cascade failure: skipped because an earlier instruction for this "
                    f"rule failed ({rule_number})"
                )
                cascade_failures.append(row)
                continue
            try:
                updated_text = _apply_generic_instruction(
                    updated_text,
                    event["instruction"],
                    event["subrule"],
                    structural_operations=structural_operations,
                )
            except AmendmentApplicationError as error:
                row["status"] = "root_failure"
                row["reason"] = str(error)
                root_failures.append(row)
                failed_rules.add(rule_number)
            else:
                row["status"] = "applied_cleanly"
                clean_rows.append(row)
        if not clean_rows:
            continue
        effective_date = same_effective_changes[0]["effective_date"]
        issue_date = max(
            date.fromisoformat(row["issue_date"]) for row in clean_rows
        )
        path_versions[-1] = current.model_copy(
            update={"effective_to": effective_date - timedelta(days=1)}
        )
        references = dict.fromkeys(
            f"Notification No. {row['notification_number']}-Central Tax"
            for row in clean_rows
        )
        path_versions.append(
            current.model_copy(
                update={
                    "id": f"{current.path}@v{len(path_versions) + 1}",
                    "text": updated_text,
                    "effective_from": effective_date,
                    "effective_to": None,
                    "in_force_since": issue_date,
                    "amended_by": "; ".join(references),
                }
            )
        )
        applied.extend(clean_rows)

    for rule_number in failed_rules:
        versions[rule_number] = [
            record.model_copy(update={"verified": False})
            for record in versions[rule_number]
        ]

    rule36_by_path = {record.path for record in rule36_records}
    generated = [record for group in versions.values() for record in group]
    generated = [
        record
        for record in generated
        if record.path not in rule36_by_path and not record.path.startswith("CGST/r36/")
    ]
    generated.extend(rule36_records)

    output = Path(output_path)
    existing = load_processed_records(output)
    preserve = [
        record
        for record in existing
        if not record.path.startswith("CGST/r")
    ]
    write_processed_records(preserve + generated, output)

    by_reason = Counter(row["reason"].split(":", 1)[0] for row in root_failures)
    top_root_failures = sorted(
        root_failures,
        key=lambda row: (
            row["effective_date"],
            row["notification_number"],
            row["rule"],
            str(row["subrule"]),
        ),
    )[:10]
    pending = _read_pending(Path(amendments_path))
    report: dict[str, Any] = {
        "source_file": pdf_path.name,
        "amendments_file": Path(amendments_path).name,
        "base_snapshot_date": BASE_DATE.isoformat(),
        "rules_in_base_pdf": len(bases),
        "ledger_instruction_count": len(events),
        "instructions_applied_cleanly": len(applied),
        "instructions_flagged": len(root_failures) + len(cascade_failures),
        "instructions_root_failures": len(root_failures),
        "instructions_cascade_failures": len(cascade_failures),
        "instructions_target_rule_not_in_base": len(target_rule_not_in_base),
        "instructions_new_rule_inserted": len(new_rules_inserted),
        "instructions_covered_by_base_snapshot": len(skipped),
        "instructions_pending_commencement": len(
            [row for row in pending if row["status"] == "pending"]
        ),
        "instructions_unparsed_date": len([row for row in pending if row["reason"] == "unparsed_date"]),
        "instructions_superseded": len([row for row in pending if row["status"] == "superseded"]),
        "pending_commencement": pending,
        "flag_reasons": dict(by_reason),
        "top_10_root_failures": top_root_failures,
        "top_10_flagged_cases": sorted(
            root_failures + cascade_failures,
            key=lambda row: (
                row["effective_date"],
                row["notification_number"],
                row["rule"],
                str(row["subrule"]),
            ),
        )[:10],
        "source_text_observations": _source_observations(source_text),
        "applied_cases": applied,
        "root_failures": root_failures,
        "cascade_failures": cascade_failures,
        "target_rule_not_in_base": target_rule_not_in_base,
        "new_rules_inserted": new_rules_inserted,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run_all_rule_versions()
    print(
        "Processed "
        f"{report['ledger_instruction_count']} instructions: "
        f"{report['instructions_applied_cleanly']} applied cleanly, "
        f"{report['instructions_root_failures']} root failures, "
        f"{report['instructions_cascade_failures']} cascade failures, "
        f"{report['instructions_target_rule_not_in_base']} target rules not in base, "
        f"{report['instructions_covered_by_base_snapshot']} covered by base snapshot."
    )
    print(f"Report: {REPORT_PATH}")
    for index, case in enumerate(report["top_10_root_failures"], start=1):
        print(
            f"{index}. Notification {case['notification_number']}, Rule {case['rule']}, "
            f"{case['effective_date']}: {case['reason']}"
        )


if __name__ == "__main__":
    main()

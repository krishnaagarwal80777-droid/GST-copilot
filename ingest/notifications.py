from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "amendments.jsonl"

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


# "with effect from the date as may be notified" (and close variants) defers commencement to a
# later notification. Such instructions have no effective date until it is appointed.
PENDING_COMMENCEMENT = re.compile(
    r"with\s+effect\s+from\s+(?:"
    r"(?:the|such)\s+date\s+as\s+(?:may\s+be|the\s+\w+(?:\s+\w+)?\s+may)"
    r"\s*,?\s*(?:by\s+notification\s*,?\s*)?(?:notified|notify|appointed|appoint)"
    r"|(?:a\s+|the\s+|such\s+)?date\s+to\s+be\s+notified)",
    re.IGNORECASE,
)
# Whole notification deferred: "They shall come into force on such date as the Central
# Government may, by notification in the Official Gazette, appoint."
NOTIFICATION_PENDING = re.compile(
    r"come\s+into\s+force\s+on\s+such\s+date\s+as\s+the\s+Central\s+Government\s+may\s*,?"
    r"(?:\s*by\s+notification[^,]*,)?\s*appoint",
    re.IGNORECASE,
)
_SHORT_TITLE = re.compile(r"(?:may\s+be\s+called|supersession\s+of)\s+the\s+(?P<title>[^.;]+?Rules\s*,\s*\d{4})", re.I)
_QUOTED = re.compile(
    "[“―‗\"](?:.|\n)*?[”‖‘\"]"
)


def is_pending_commencement(instruction: str) -> bool:
    """True when the instruction's own words (not quoted inserted text) defer commencement."""
    return PENDING_COMMENCEMENT.search(_QUOTED.sub(" ", instruction)) is not None



def _own_words(instruction: str) -> str:
    """The instruction with quoted inserted/substituted text removed."""
    return _QUOTED.sub(" ", instruction)


_NEW_RULE = re.compile(
    r"after\s+rule\s+(?P<after>\d+[A-Z]*)\b[^“―‗\"]*?the\s+following\s+rules?\s+"
    r"shall\s+be\s+inserted\s*,?\s*namely\s*:?\s*[-–—]*\s*[“―‗\"]\s*"
    r"(?P<rule>\d+[A-Z]+)\s*\.",
    re.IGNORECASE,
)


def new_rule_insertion(instruction: str) -> tuple[str, str] | None:
    """("10A", "10B") for 'After rule 10A ..., the following rule shall be inserted ... "10B. ...'."""
    match = _NEW_RULE.search(" ".join(instruction.split()))
    return (match.group("after"), match.group("rule")) if match else None


def instruction_commencement(
    instruction: str,
    *,
    issue_date: date,
    default: date | None,
    default_reason: str = "unparsed_date",
) -> tuple[date | None, str, str | None]:
    """(effective_date, commencement, pending_reason) from the instruction's own words.

    A stated date that cannot be parsed is never replaced by the publication date: the
    instruction is pending with reason "unparsed_date" and listed in the version report.
    """
    own = " ".join(_own_words(instruction).split())
    if PENDING_COMMENCEMENT.search(own):
        return None, "pending", "awaiting_notification"
    for pattern in (
        r"with\s+effect\s+from\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+come\s+into\s+force\s+(?:from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+(?:be\s+)?deemed\s+to\s+have\s+come\s+into\s+force\s+(?:from|on)\s+"
        r"(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
    ):
        match = re.search(pattern, own, re.IGNORECASE)
        if match:
            if match.group("date").lower().startswith("date of their publication"):
                return issue_date, "fixed", None
            parsed = _parse_date_expression(match.group("date"))
            if parsed is None:
                return None, "pending", "unparsed_date"
            return parsed, "fixed", None
    if "date of their publication" in own.lower():
        return issue_date, "fixed", None
    if default is None:
        return None, "pending", default_reason
    return default, "fixed", None


def _parse_day(value: str) -> int:
    digits = re.search(r"(\d{1,2})", value)
    if digits is None:
        raise ValueError(f"Could not parse day from {value!r}")
    return int(digits.group(1))


def parse_legal_date(value: str) -> date:
    """Parse legal-date strings like '31st March, 2023' or '1st day of April, 2026'."""
    cleaned = value.strip().replace("—", " ").replace("–", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    match = re.search(
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+day\s+of)?\s+(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})",
        cleaned,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Unsupported legal date string: {value!r}")
    day = _parse_day(match.group("day"))
    month = MONTHS[match.group("month").capitalize()]
    year = int(match.group("year"))
    return date(year, month, day)


def _extract_notification_number(text: str) -> str:
    patterns = [
        r"Notification\s+No\.\s*([0-9]+/[0-9]{4})",
        r"Notification\s+No\.\s*([0-9]+[-/][A-Za-z0-9]+)",
        r"No\.\s*\[?([0-9]+/[0-9]{4})\]?",
        r"No\.\s*([0-9]+/[0-9]{4})\s*[-\u2013]\s*Central\s+Tax",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "")
    raise ValueError("Could not identify notification number")


def _extract_issue_date(text: str, fallback: date | None = None) -> date:
    patterns = [
        r"New\s+Delhi,\s+dated\s+the\s+(?P<date>[A-Za-z0-9,\s]+)",
        r"New\s+Delhi,\s+the\s+(?P<date>[A-Za-z0-9,\s]+)",
        r"dated\s+the\s+(?P<date>[A-Za-z0-9,\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return parse_legal_date(match.group("date"))
            except ValueError:
                continue
    if fallback is not None:
        return fallback
    raise ValueError("Could not identify issue date")


def _fallback_issue_date_from_filename(source_file: str) -> date | None:
    match = re.search(r"_(\d{4})-(\d{2})-(\d{2})\.pdf$", source_file)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _parse_date_expression(value: str, *, fallback: date | None = None) -> date | None:
    value = value.strip().replace("—", " ").replace("–", " ")
    value = re.sub(r"\s+", " ", value)
    try:
        return parse_legal_date(value)
    except ValueError:
        if fallback is not None:
            return fallback
        return None


def _extract_notification_commencement_date(text: str, issue_date: date) -> date | None:
    # Layout text breaks phrases like "date of their\npublication" across lines.
    text = " ".join(text.split())
    patterns = [
        r"shall\s+(?:be\s+)?deemed\s+to\s+have\s+come\s+into\s+force\s+(?:from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+come\s+into\s+force\s+(?:with\s+effect\s+from|from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"come\s+into\s+force\s+(?:with\s+effect\s+from|from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            parsed = _parse_date_expression(match.group("date"))
            if parsed is not None:
                return parsed
    if "date of their publication" in text.lower():
        return issue_date
    return None


def _extract_item_effective_date(item_text: str, *, issue_date: date, default: date | None = None) -> date | None:
    patterns = [
        r"with\s+effect\s+from\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+come\s+into\s+force\s+(?:with\s+effect\s+from|from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+(?:be\s+)?deemed\s+to\s+have\s+come\s+into\s+force\s+(?:from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"date\s+of\s+their\s+publication\s+in\s+the\s+Official\s+Gazette",
    ]
    for pattern in patterns:
        match = re.search(pattern, item_text, re.IGNORECASE)
        if match:
            if "date of their publication" in match.group(0).lower():
                return issue_date
            parsed = _parse_date_expression(match.group("date"), fallback=issue_date)
            if parsed is not None:
                return parsed
    if default is not None:
        return default
    return None


def _classify_notification(text: str) -> str:
    lowered = re.sub(r"\s+", " ", text.lower())
    lowered = lowered.replace("am end", "amend")
    if re.search(r"rules\s+further\s+to\s+amend|further\s+to\s+amend\s+the\s+central\s+goods\s+and\s+services\s+tax\s+rules", lowered):
        return "amendment-rules"
    if re.search(r"appoints?|empowers?|constitutes?|notifies?", lowered) and "come into force" in lowered:
        return "commencement"
    if "shall come into force" in lowered or "deemed to have come into force" in lowered:
        return "commencement"
    return "other"


def _extract_rule_mentions(text: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"\brule\s+(\d+[A-Z]?)\b", text, re.IGNORECASE):
        ref = match.group(1)
        if ref and ref not in refs:
            refs.append(ref)
    for match in re.finditer(r"\bsub\s*[- ]\s*rule\s*\((\d+[A-Z]?)\)\s+of\s+rule\s+(\d+[A-Z]?)\b", text, re.IGNORECASE):
        ref = match.group(2)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _extract_target_rules(item_text: str) -> list[tuple[str, str | None, str | None]]:
    targets: list[tuple[str, str | None, str | None]] = []

    def add_target(rule: str, subrule: str | None = None, clause: str | None = None) -> None:
        normalized_rule = rule.strip()
        normalized_subrule = subrule.strip() if subrule else None
        normalized_clause = clause.strip() if clause else None
        if any(existing[0] == normalized_rule and existing[1] == normalized_subrule and existing[2] == normalized_clause for existing in targets):
            return
        targets.append((normalized_rule, normalized_subrule, normalized_clause))

    # Add a broad rule target for "in rule N" phrases even when the sentence has a long preamble
    # before the actual amendment instruction. The later filtering removes the broader form when a
    # more specific sub-rule/clause target exists in the same instruction.
    for match in re.finditer(r"\bin\s+rule\s+(?P<rule>\d+[A-Z]?)\b", item_text, re.IGNORECASE):
        add_target(match.group("rule"))

    for match in re.finditer(
        r"\bin\s+rule\s+(?P<rule>\d+[A-Z]?)\b(?:[\s\S]{0,200}?)\bin\s+sub\s*[- ]\s*rule\s*\((?P<subrule>[^)]+)\)(?:[\s\S]{0,200}?)\bin\s+clause\s*\((?P<clause>[^)]+)\)",
        item_text,
        re.IGNORECASE | re.DOTALL,
    ):
        add_target(match.group("rule"), match.group("subrule"), match.group("clause"))

    for match in re.finditer(
        r"\bin\s+rule\s+(?P<rule>\d+[A-Z]?)\b(?:[\s\S]{0,200}?)\b(?:for|in)\s+sub\s*[- ]\s*rule\s*\((?P<subrule>[^)]+)\)",
        item_text,
        re.IGNORECASE | re.DOTALL,
    ):
        add_target(match.group("rule"), match.group("subrule"), None)

    for match in re.finditer(
        r"\bin\s+rule\s+(?P<rule>\d+[A-Z]?)\b(?:[\s\S]{0,200}?)\bin\s+clause\s*\((?P<clause>[^)]+)\)",
        item_text,
        re.IGNORECASE | re.DOTALL,
    ):
        add_target(match.group("rule"), None, match.group("clause"))

    for match in re.finditer(
        r"\b(?:in|for)\s+sub\s*[- ]\s*rule\s*\((?P<subrule>[^)]+)\)\s*(?:of\s+)?rule\s+(?P<rule>\d+[A-Z]?)\b",
        item_text,
        re.IGNORECASE,
    ):
        add_target(match.group("rule"), match.group("subrule"), None)

    for match in re.finditer(
        r"\b(?:in|for)\s+clause\s*\((?P<clause>[^)]+)\)\s*(?:of\s+)?rule\s+(?P<rule>\d+[A-Z]?)\b",
        item_text,
        re.IGNORECASE,
    ):
        add_target(match.group("rule"), None, match.group("clause"))

    filtered: list[tuple[str, str | None, str | None]] = []
    for rule, subrule, clause in targets:
        if subrule is not None and clause is not None:
            filtered.append((rule, subrule, clause))
            continue
        if clause is not None and any(other[0] == rule and other[1] is not None and other[2] == clause for other in targets):
            continue
        if subrule is not None and any(other[0] == rule and other[1] == subrule and other[2] is not None for other in targets):
            continue
        if subrule is None and clause is None and any(
            other[0] == rule and (other[1] is not None or other[2] is not None) for other in targets
        ):
            continue
        filtered.append((rule, subrule, clause))
    return filtered


def _is_amendment_item(item_text: str) -> bool:
    lower = re.sub(r"\s+", " ", item_text.lower())
    if re.search(r"after\s+rule\s+\d+[A-Z]?\s*,\s*the\s+following\s+rule\s+shall\s+be\s+inserted", lower):
        return True
    if re.search(r"shall\s+be\s+(?:substituted|omitted|inserted)", lower):
        return bool(
            re.search(
                r"(?:\brule\s+\d+[A-Za-z]?\b|\bsub\s*[- ]\s*rule\s*\(\d+[A-Za-z]?\)\b|\bclause\s*\([A-Za-z0-9]+\)\b|\bin\s+rule\s+\d+[A-Za-z]?\b|\bafter\s+clause\s*\([A-Za-z0-9]+\)\b)",
                lower,
                re.IGNORECASE,
            )
        )
    if "for the following" in lower and "rule" in lower:
        return True
    return False


def _split_lettered_items(item_text: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9])\((?:[a-z]|[ivxlcdm]+)\)\s+",
            item_text,
            re.IGNORECASE,
        )
    )
    roman_matches = [
        match
        for match in matches
        if re.fullmatch(r"[ivxlcdm]+", match.group(0)[1 : match.group(0).find(")")], re.IGNORECASE)
    ]
    has_roman_list = any(
        len(match.group(0)[1 : match.group(0).find(")")]) > 1 for match in roman_matches
    )
    selected_matches = roman_matches if has_roman_list else [
        match
        for match in matches
        if re.fullmatch(r"[a-hj-uw-z]", match.group(0)[1 : match.group(0).find(")")], re.IGNORECASE)
    ]
    if len(selected_matches) < 2:
        return [item_text]

    prefix = item_text[: selected_matches[0].start()].strip()
    items: list[str] = []
    for index, match in enumerate(selected_matches):
        start = match.start()
        end = (
            selected_matches[index + 1].start()
            if index + 1 < len(selected_matches)
            else len(item_text)
        )
        segment = item_text[start:end].strip()
        if segment:
            combined = f"{prefix} {segment}".strip()
            items.append(combined)
    return items


def split_numbered_items(text: str) -> list[str]:
    # Paragraph numbers are 1-3 digits; a year ("... 1st day of November, 2024.") is not one.
    matches = list(re.finditer(r"(?m)(?<![\w,])(\d{1,3})\.\s+", text))
    if not matches:
        return [text]
    items: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        if not item:
            continue
        rule_instruction_starts = list(
            re.finditer(
                r"(?<![A-Za-z0-9])\(\d+\)\s*(?=(?:in|after)\s+rule\s+\d+[A-Z]?\b)",
                item,
                re.IGNORECASE,
            )
        )
        rule_items = [item]
        if rule_instruction_starts:
            rule_items = []
            for index, match in enumerate(rule_instruction_starts):
                end = (
                    rule_instruction_starts[index + 1].start()
                    if index + 1 < len(rule_instruction_starts)
                    else len(item)
                )
                rule_items.append(item[match.start() : end].strip())
        for rule_item in rule_items:
            items.extend(_split_lettered_items(rule_item))
    return items


def _extract_inline_insertions(raw_text: str) -> list[str]:
    insertions: list[str] = []
    for match in re.finditer(
        r"after\s+rule\s+\d+[A-Z]?\s*,\s*the\s+following\s+rule\s+shall\s+be\s+inserted.*?(?:\?\?\s*)?(\d+[A-Z]?)\.?\s*\.",
        raw_text,
        re.IGNORECASE | re.DOTALL,
    ):
        inserted_text = match.group(0).strip()
        if inserted_text and inserted_text not in insertions:
            insertions.append(inserted_text)
    return insertions


def _extract_instruction_effective_date(item_text: str, *, issue_date: date, default: date | None = None) -> date | None:
    phrase_patterns = [
        r"with\s+effect\s+from\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+come\s+into\s+force\s+(?:from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
        r"shall\s+(?:be\s+)?deemed\s+to\s+have\s+come\s+into\s+force\s+(?:from|on)\s+(?:the\s+)?(?P<date>[A-Za-z0-9,\s]+)",
    ]
    for pattern in phrase_patterns:
        match = re.search(pattern, item_text, re.IGNORECASE)
        if match:
            parsed = _parse_date_expression(match.group("date"), fallback=issue_date)
            if parsed is not None:
                return parsed
    if "date of their publication" in item_text.lower():
        return issue_date
    if default is not None:
        return default
    return None


def parse_notification_text(raw_text: str, *, source_file: str) -> dict[str, object]:
    notification_number = _extract_notification_number(raw_text)
    issue_date = _extract_issue_date(raw_text, fallback=_fallback_issue_date_from_filename(source_file))
    flat_text = " ".join(raw_text.split())
    notification_pending = NOTIFICATION_PENDING.search(flat_text) is not None
    notification_start = (
        None if notification_pending else _extract_notification_commencement_date(raw_text, issue_date)
    )
    default_reason = "awaiting_notification" if notification_pending else "unparsed_date"
    titles = [" ".join(m.group("title").split()) for m in _SHORT_TITLE.finditer(flat_text[:3000])]
    supersedes = re.search(r"supersession\s+of\s+the\s+(?P<title>[^.;]+?Rules\s*,\s*\d{4})", flat_text[:3000], re.I)
    items = split_numbered_items(raw_text)
    references: set[str] = set()
    amendments: list[dict[str, object]] = []
    seen_amendments: set[tuple[str, str | None, str | None, str]] = set()

    inline_insertions = _extract_inline_insertions(raw_text)
    for insertion in inline_insertions:
        match = re.search(r"(\d+[A-Z]?)\.?\s*\.", insertion)
        if match:
            rule_number = match.group(1)
            if rule_number:
                references.add(rule_number)
                instruction = insertion.strip()
                target_key = (rule_number, None, None, instruction)
                if target_key not in seen_amendments:
                    seen_amendments.add(target_key)
                    effective, commencement, reason = instruction_commencement(
                        instruction,
                        issue_date=issue_date,
                        default=notification_start,
                        default_reason=default_reason,
                    )
                    amendments.append(
                        {
                            "rule": rule_number,
                            "subrule": None,
                            "clause": None,
                            "effective_date": effective.isoformat() if effective else None,
                            "commencement": commencement,
                            "pending_reason": reason,
                            "operation": "insert_rule",
                            "issue_date": issue_date.isoformat(),
                            "instruction": instruction,
                        }
                    )

    for item in items:
        item_lower = item.lower()
        item_refs = _extract_rule_mentions(item)
        for ref in item_refs:
            references.add(ref)

        if not _is_amendment_item(item):
            continue
        inserted_rule = new_rule_insertion(item)
        operation = "insert_rule" if inserted_rule else "amend"
        # Quoted inserted text can mention other rules (e.g. "... provided in rule 10B"); only
        # the instruction's own words name its target.
        if inserted_rule:
            targets = [(inserted_rule[1], None, None)]
        else:
            targets = _extract_target_rules(_own_words(item))
        if not targets:
            continue

        item_effective, commencement, reason = instruction_commencement(
            item, issue_date=issue_date, default=notification_start, default_reason=default_reason
        )
        for rule_number, subrule, clause in targets:
            target_key = (rule_number, subrule, clause, item.strip())
            if target_key in seen_amendments:
                continue
            seen_amendments.add(target_key)
            amendments.append(
                {
                    "rule": rule_number,
                    "subrule": subrule,
                    "clause": clause,
                    "effective_date": item_effective.isoformat() if item_effective else None,
                    "commencement": commencement,
                    "pending_reason": reason,
                    "operation": operation,
                    "after_rule": inserted_rule[0] if inserted_rule else None,
                    "issue_date": issue_date.isoformat(),
                    "instruction": item.strip(),
                }
            )

    rule_refs = sorted({item["rule"] for item in amendments})
    parsed = {
        "notification_number": notification_number,
        "issue_date": issue_date.isoformat(),
        "source_file": source_file,
        "type": _classify_notification(raw_text),
        "commencement": "pending" if notification_pending else "fixed",
        "short_title": next((t for t in titles if not supersedes or t != " ".join(supersedes.group("title").split())), None),
        "supersedes_title": " ".join(supersedes.group("title").split()) if supersedes else None,
        "effective_date": notification_start.isoformat() if notification_start else None,
        "rule_refs": rule_refs,
        "references": sorted(references),
        "amendments": amendments,
        "instruction": " | ".join(item for item in items if re.search(r"rule\s+\d+[A-Z]?\b", item, re.IGNORECASE)),
    }
    return parsed


def _find_pdftotext() -> str | None:
    pdftotext = os.environ.get("PDFTOTEXT") or shutil.which("pdftotext")
    if pdftotext is None:
        conda_pdftotext = Path(sys.prefix) / "Library" / "bin" / "pdftotext.exe"
        if conda_pdftotext.is_file():
            pdftotext = str(conda_pdftotext)
    return pdftotext


def read_pdf_text(path: Path) -> str:
    pdftotext = _find_pdftotext()
    if pdftotext is not None:
        result = subprocess.run(
            # Without -enc, pdftotext on Windows writes the console code page (cp1252).
            [pdftotext, "-layout", "-enc", "UTF-8", str(path), "-"],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if "\ufffd" in result.stdout:
            raise ValueError(f"pdftotext produced replacement characters for {path}")
        return result.stdout

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        content = page.extract_text() or ""
        pages.append(content)
    text = "\n".join(pages)
    if "\ufffd" in text:
        raise RuntimeError(
            f"PDF text extraction produced replacement characters for {path}, "
            "but Poppler pdftotext is not installed or not on PATH"
        )
    return text


def iter_notifications(raw_dir: Path = RAW_DIR) -> Iterable[dict[str, object]]:
    for pdf_path in sorted(raw_dir.glob("ct-*.pdf")):
        text = read_pdf_text(pdf_path)
        yield parse_notification_text(text, source_file=pdf_path.name)


def resolve_commencements(records: list[dict[str, object]], raw_dir: Path = RAW_DIR) -> list[dict[str, object]]:
    """Apply every commencement notification in raw_dir to the pending instructions it covers."""
    from ingest.commencement import (
        commences_notification,
        is_commencement_text,
        link_supersessions,
        parse_commencement_text,
        resolve_pending,
    )

    link_supersessions(records)

    appointments = []
    for record in records:
        text = read_pdf_text(raw_dir / str(record["source_file"]))
        if is_commencement_text(text):
            record["type"] = "commencement"
            number = str(record["notification_number"])
            record["commences"] = "notification" if commences_notification(text, number) else "act"
            appointments.extend(
                parse_commencement_text(
                    text,
                    notification_number=str(record["notification_number"]),
                    source_file=str(record["source_file"]),
                )
            )
    return resolve_pending(records, appointments)


def write_amendments(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH) -> list[dict[str, object]]:
    records = resolve_commencements(list(iter_notifications(raw_dir)), raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    records = write_amendments()
    print(f"Wrote {len(records)} notification records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

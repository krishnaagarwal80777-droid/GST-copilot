from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

from gst_copilot.models import NodeType, ProvisionRecord
from ingest.index import DEFAULT_PROCESSED_FILE, load_processed_records, write_processed_records

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "eval" / "core_provisions.yaml"
ACT_PDF = ROOT / "data" / "raw" / "cgst-act_cbic-taxinfo_2022-09-28.pdf"
RULES_BASE_SOURCE = "data/raw/cgst-rules-part-a_2021-05-18.pdf"
OUTPUT_PATH = ROOT / "data" / "processed" / "core_provisions.jsonl"
FOOTNOTES_PATH = ROOT / "data" / "processed" / "core_provision_footnotes.json"
# Post-2022 Act versions come from the CBIC section pages (ingest.act_post_pdf); their
# footnotes and the amendments that could not be dated go here.
POST_PDF_PATH = ROOT / "data" / "processed" / "act_post_pdf.json"
ACT_BASE_DATE = date(2017, 7, 1)
AMENDMENT_EFFECTIVE_DATE = date(2019, 2, 1)


def _read_pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _section_parts(text: str, number: str) -> tuple[str, str]:
    match = re.search(
        rf"(?im)^\s*\*?\s*Section\s+{re.escape(number)}\.\s*(?P<title>[^-\n]+).*?$",
        text,
    )
    if match is None:
        raise ValueError(f"Could not find CGST Act section {number} in the source PDF")
    next_section = re.search(
        r"(?im)^\s*\*?\s*Section\s+\d+[A-Z]?\.",
        text[match.end() :],
    )
    end = match.end() + next_section.start() if next_section else len(text)
    section = text[match.start() : end]
    enforced = re.search(r"(?im)^\s*\*?\s*Enforced\b", section)
    body_end = enforced.start() if enforced else len(section)
    return section[:body_end], section[body_end:]


def _footnote_entries(section: str) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    # The CBIC PDF prefixes some footnote markers with a zero-width space.
    section = section.replace("​", "")
    matches = list(re.finditer(r"(?m)^\s*(?P<marker>\d+)\s*\.\s+", section))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        text = re.sub(r"\s+", " ", section[match.end() : end]).strip()
        effective = _extract_effective_date(text)
        old_text = _extract_old_text(text)
        operation = re.match(r"(Substituted|Inserted|Omitted)\b", text, re.IGNORECASE)
        notification = re.search(
            r"(?:Notification|Order)\s+No\.\s*\d+/\d{4}\s*-\s*(?:C\.?\s*T\.?|Central Tax)",
            text,
            re.IGNORECASE,
        )
        notes.append(
            {
                "marker": match.group("marker"),
                "text": text,
                "operation": operation.group(1).lower() if operation else None,
                "instrument": _extract_instrument(text),
                "notification": re.sub(r"\s+", " ", notification.group()) if notification else None,
                "effective_from": effective.isoformat() if effective else None,
                "old_text": old_text,
            }
        )
    return notes


def _extract_effective_date(text: str) -> date | None:
    match = re.search(
        r"w\.?\s*e\.?\s*f\.?\s*(?:the\s*)?"
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)"
        r",?\s*(?P<year>\d{4})",
        text,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(r"w\.?\s*e\.?\s*f\.?\s*(?P<date>\d{1,2}[./]\d{1,2}[./]\d{4})", text, re.I)
        if match is None:
            return None
        day, month, year = re.split(r"[./]", match.group("date"))
        return date(int(year), int(month), int(day))
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return date(
        int(match.group("year")),
        months[match.group("month").lower()],
        int(match.group("day")),
    )


def _extract_instrument(text: str) -> str | None:
    match = re.search(
        r"\bby\s+(?P<instrument>s\.?\s*\d+.*?)(?=(?:\s*[-–—]\s*Brought into force)|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        # Drop a trailing "for '<old text>'" when the footnote names the instrument first.
        instrument = re.split(r"\s+for\s+['\"‘’“”]", match.group("instrument"), maxsplit=1)[0]
        return instrument.strip(" .")
    match = re.search(
        r"\b(vide\s+(?:Notification|Order)\s+No\.\s*[^)]+?)(?=\s+dated\b|\)|$)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip(" .") if match else None


def _extract_old_text(text: str) -> str | None:
    match = re.search(
        r"Substituted(?:\s+\([^)]*\))?\s+for\s+['\"\u2018\u2019\u201c\u201d](?P<old>.*?)['\"\u2018\u2019\u201c\u201d]\s+(?:by|w\.e\.f\.)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"Substituted\s+for\s+['\"\u2018\u2019\u201c\u201d](?P<old>.*?)['\"\u2018\u2019\u201c\u201d]",
            text,
            re.IGNORECASE,
        )
    if match is None:
        # "Substituted (w.e.f. ...) by <instrument> for ''<old text>"."
        match = re.search(
            r"Substituted\b.*?\bby\s+s\.\s*\d+.*?\bfor\s+['\"‘’“”]+"
            r"(?P<old>.*?)['\"‘’“”]+\s*\.?\s*$",
            text,
            re.IGNORECASE,
        )
    if match is None:
        match = re.search(
            r"^Omitted\s+['\"‘’“”](?P<old>.*?)['\"‘’“”]",
            text,
            re.IGNORECASE,
        )
    return match.group("old").strip() if match else None


def _clause(section_body: str, marker: str, *, end_marker: str | None = None) -> str:
    start = re.search(rf"(?m)^\s*\(?{re.escape(marker)}\)\s*", section_body)
    if start is None:
        raise ValueError(f"Could not find clause ({marker}) in the source section")
    end = len(section_body)
    if end_marker is not None:
        next_node = re.search(
            rf"(?m)^\s*\(?{re.escape(end_marker)}\)\s*",
            section_body[start.end() :],
        )
        if next_node is None:
            raise ValueError(f"Could not find end marker ({end_marker}) in the source section")
        end = start.end() + next_node.start()
    return re.sub(r"\s+", " ", section_body[start.start() : end]).strip()


def _record(
    *,
    path: str,
    node_type: NodeType,
    text: str,
    version: int,
    effective_from: date,
    effective_to: date | None,
    amended_by: str | None,
    source_url: str,
    verified: bool = True,
) -> ProvisionRecord:
    return ProvisionRecord(
        id=f"{path}@v{version}",
        doc_type="act",
        act_id="CGST2017",
        path=path,
        node_type=node_type,
        text=text,
        effective_from=effective_from,
        effective_to=effective_to,
        in_force_since=effective_from,
        amended_by=amended_by,
        source_url=source_url,
        verified=verified,
        verified_by="manual" if verified else None,
    )


def _strip_markers(text: str) -> str:
    """Remove CBIC footnote markers like ``6 [`` / ``]`` / ``7 [****]`` from current text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+\s*\[\s*\*+\s*\]", "", text)
    text = re.sub(r"\d+\s*\[\s*", "", text)
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"\s+([,;:.])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _find(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not find the current text of {label}")
    return match.group(0)


def _note(notes: list[dict[str, Any]], marker: str, label: str) -> dict[str, Any]:
    note = next((note for note in notes if note["marker"] == marker), None)
    if note is None:
        raise ValueError(f"Footnote {marker} for {label} not found")
    return note


def _apply_core_rule_edits(records: list[ProvisionRecord]) -> list[ProvisionRecord]:
    by_path: dict[str, list[ProvisionRecord]] = {}
    for record in records:
        by_path.setdefault(record.path, []).append(record)

    updated: list[ProvisionRecord] = []
    for path in ("CGST/r42", "CGST/r43"):
        versions = sorted(by_path.get(path, []), key=lambda record: record.effective_from)
        if len(versions) != 1:
            raise ValueError(f"Expected one source version for manual review of {path}")
        base = versions[0].model_copy(update={"verified": True, "verified_by": "manual"})
        text = base.text
        if path == "CGST/r42":
            target = re.compile(r"at\s+the\s+invoice\s+level\s+in\s+FORM\s+GSTR-2(?:\s+and)?", re.I)
            text, count = target.subn("", text, count=1)
            notification = "Notification No. 19/2022-Central Tax"
            in_force_since = date(2022, 9, 28)
        else:
            target = re.compile(r"FORM\s+GSTR-2\s*\[\s*and\s+FORM\s+GSTR-3B\s*\]", re.I)
            text, count = target.subn("FORM GSTR-3B", text)
            notification = "Notification No. 19/2022-Central Tax"
            in_force_since = date(2022, 9, 28)
        expected = 1 if path == "CGST/r42" else 2
        if count != expected:
            raise ValueError(
                f"Manual amendment for {path} expected {expected} targets, found {count}"
            )
        effective_from = date(2022, 10, 1)
        if effective_from <= base.effective_from:
            raise ValueError(f"Manual amendment date precedes base version for {path}")
        base = base.model_copy(update={"effective_to": effective_from - timedelta(days=1)})
        amended = base.model_copy(
            update={
                "id": f"{path}@v2",
                "text": text,
                "effective_from": effective_from,
                "effective_to": None,
                "in_force_since": in_force_since,
                "amended_by": notification,
                "verified": True,
                "verified_by": "manual",
                "source_url": "data/raw/ct-19-2022_2022-09-28.pdf",
            }
        )
        updated.extend((base, amended))

    other = [record for record in records if record.path not in {"CGST/r42", "CGST/r43"}]
    return other + updated


def _rule_amendments(
    notification_path: Path,
    number: str,
    *,
    rule: str = "89",
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    with notification_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            notification = json.loads(line)
            if notification.get("notification_number") != number:
                continue
            for amendment in notification.get("amendments", []):
                if str(amendment.get("rule")) == rule:
                    found.append(amendment)
    return found


def _quoted_texts(instruction: str) -> list[str]:
    normalized = instruction.translate(
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
    return re.findall(r'"([^"]*)"', normalized, re.DOTALL)


def _replace_source_text(text: str, old: str, new: str, *, label: str) -> str:
    normalized_old = re.sub(r"\s+", " ", old).strip()
    normalized_old = re.sub(r"\s*-\s*", "-", normalized_old)
    source_pattern = re.escape(normalized_old).replace(r"\ ", r"\s+")
    source_pattern = source_pattern.replace(r"\-", r"\s*-\s*")
    pattern = re.compile(source_pattern, re.IGNORECASE)
    updated, count = pattern.subn(new, text, count=1)
    if count != 1:
        raise ValueError(f"Manual edit {label}: expected one source anchor, found {count}")
    return updated


def _event_for(
    events: list[dict[str, Any]],
    predicate: Any,
    *,
    label: str,
) -> dict[str, Any]:
    matches = [event for event in events if predicate(str(event.get("instruction", "")))]
    matches = list({str(event.get("instruction", "")): event for event in matches}.values())
    if len(matches) != 1:
        raise ValueError(f"Expected one source instruction for {label}, found {len(matches)}")
    return matches[0]


def _manual_rule43_versions(records: list[ProvisionRecord]) -> list[ProvisionRecord]:
    versions = sorted(
        (record for record in records if record.path == "CGST/r43"),
        key=lambda record: record.effective_from,
    )
    if len(versions) != 2 or versions[-1].amended_by != "Notification No. 19/2022-Central Tax":
        raise ValueError("Expected Rule 43 source and Notification 19/2022 versions")

    events = _rule_amendments(
        ROOT / "data" / "processed" / "amendments.jsonl",
        "38/2023",
        rule="43",
    )
    _event_for(
        events,
        lambda value: "(c) shall be omitted" in value.lower(),
        label="Rule 43/38/2023(a)",
    )
    _event_for(
        events,
        lambda value: "after explanation 2" in value.lower() and "explanation 3" in value.lower(),
        label="Rule 43/38/2023(b)",
    )
    source_text = versions[-1].text
    explanation1 = re.search(
        r"Explanation\s*1\s*:\s*-?\s*For the purposes of rule 42 and this rule,"
        r"[\s\S]*?(?=Explanation\s*2\s*:)",
        source_text,
        re.IGNORECASE,
    )
    if explanation1 is None:
        raise ValueError("Manual edit Rule 43/38/2023(a): Explanation 1 not found")
    clause_c = re.compile(r"\s*and\s*\(c\)[\s\S]*?(?=\])", re.IGNORECASE)
    updated_explanation, count = clause_c.subn("", explanation1.group(), count=1)
    if count != 1:
        raise ValueError("Manual edit Rule 43/38/2023(a): Explanation 1 clause (c) not found")
    text = (
        source_text[: explanation1.start()]
        + updated_explanation
        + source_text[explanation1.end() :]
    )

    notification_pdf = ROOT / "data" / "raw" / "ct-38-2023_2023-08-04.pdf"
    notification_text = _read_pdf_text(notification_pdf)
    inserted_match = re.search(
        r"Explanation\s*3\s*:\s*-?\s*[\s\S]*?incoming passengers\.",
        notification_text,
        re.IGNORECASE,
    )
    if inserted_match is None:
        raise ValueError("Could not extract complete Rule 43 Explanation 3 from source PDF")
    inserted = re.sub(r"\s+", " ", inserted_match.group()).strip()
    explanation2 = re.search(
        r"Explanation\s*2\s*:[\s\S]*?integrated tax\.",
        text,
        re.IGNORECASE,
    )
    if explanation2 is None:
        raise ValueError("Manual edit Rule 43/38/2023(b): Explanation 2 end not found")

    result = [
        versions[0],
        versions[1].model_copy(update={"effective_to": date(2023, 8, 3)}),
    ]
    omitted = versions[1].model_copy(
        update={
            "id": "CGST/r43@v3",
            "text": text,
            "effective_from": date(2023, 8, 4),
            "effective_to": date(2023, 9, 30),
            "in_force_since": date(2023, 8, 4),
            "amended_by": "Notification No. 38/2023-Central Tax",
            "verified": True,
            "verified_by": "manual",
            "source_url": "data/raw/ct-38-2023_2023-08-04.pdf",
        }
    )
    inserted_version = omitted.model_copy(
        update={
            "id": "CGST/r43@v4",
            "text": text[: explanation2.end()] + " " + inserted + text[explanation2.end() :],
            "effective_from": date(2023, 10, 1),
            "effective_to": None,
            "in_force_since": date(2023, 10, 1),
        }
    )
    result.extend((omitted, inserted_version))
    return [record for record in records if record.path != "CGST/r43"] + result


def _manual_rule89_edits(text: str, notification_path: Path, notification: str) -> str:
    from ingest.all_rule_versions import _inserted_text

    events = _rule_amendments(notification_path, notification)
    if notification == "14/2022":
        source = _read_pdf_text(ROOT / "data" / "raw" / "ct-14-2022_2022-07-05.pdf")
        source_start = source.lower().find("in rule 89")
        source_end = source.find("\n9.", source_start)
        if source_start < 0 or source_end < 0:
            raise ValueError("Could not isolate Rule 89 instructions in Notification 14/2022")
        source = source[source_start:source_end]
        inserted_explanation_match = re.search(
            r"\u2017Explanation\.[\s\S]*?Economic Zone Rules, 2006\.\u2018",
            source,
            re.IGNORECASE,
        )
        clause_ba_start = re.search(r"[\u2015\u2017]?\(ba\)", source, re.IGNORECASE)
        clause_ba_end = (
            source.find("\u2016", clause_ba_start.end()) if clause_ba_start is not None else -1
        )
        clause_ba_text = (
            source[clause_ba_start.start() : clause_ba_end]
            if clause_ba_start is not None and clause_ba_end >= 0
            else None
        )
        inserted_explanation4_match = re.search(
            r"\u2015Explanation\.\s*[–-]\s*For the purposes of this sub\s*-rule,"
            r"[\s\S]*?whichever is less\.\u2016",
            source,
            re.IGNORECASE,
        )
        formula_match = re.search(
            r"\u2015\{tax payable on such inverted rated supply[\s\S]*?\}\.\u2016",
            source,
            re.IGNORECASE,
        )
        missing_source_text = [
            name
            for name, match in (
                ("sub-rule (1) Explanation", inserted_explanation_match),
                ("sub-rule (2) clause (ba)", clause_ba_text),
                ("sub-rule (4) Explanation", inserted_explanation4_match),
                ("sub-rule (5) formula", formula_match),
            )
            if match is None
        ]
        if missing_source_text:
            raise ValueError(
                "Could not extract Rule 89 Notification 14/2022 source text: "
                + ", ".join(missing_source_text)
            )

        def normalized_source(match: re.Match[str]) -> str:
            return re.sub(r"\s+", " ", match.group()).strip()

        quote_glyphs = '\u2015\u2016\u2017\u2018\u201c\u201d"'
        inserted_explanation = normalized_source(inserted_explanation_match).strip(quote_glyphs)
        inserted_clause = re.sub(r"\s+", " ", clause_ba_text).strip(quote_glyphs)
        inserted_explanation4 = normalized_source(inserted_explanation4_match).strip(quote_glyphs)
        new_formula = normalized_source(formula_match).strip(quote_glyphs)
        fourth_proviso = re.search(
            r"Provided that[\s\S]*?(?=Provided further that)", text, re.IGNORECASE
        )
        if fourth_proviso is None:
            raise ValueError("Manual edit Rule 89/14/2022(a): fourth proviso not found")
        text = (
            text[: fourth_proviso.end()] + " " + inserted_explanation + text[fourth_proviso.end() :]
        )

        application_offset = text.find("The application under sub-rule (1)")
        subrule2 = re.search(r"\(2\)[\s\S]*?(?=\(3\))", text)
        if application_offset < 0 or subrule2 is None:
            raise ValueError("Manual edit Rule 89/14/2022(b)(ii): sub-rule (2) not found")
        clause_b = re.search(r"\(b\)[\s\S]*?(?=\(c\))", text[application_offset:])
        if clause_b is None:
            raise ValueError("Manual edit Rule 89/14/2022(b)(ii): clause (b) not found")
        words_update = re.compile(r"on account of export of goods", re.IGNORECASE)
        scoped_clause_b = text[
            application_offset + clause_b.start() : application_offset + clause_b.end()
        ]
        scoped_clause_b, count = words_update.subn(
            "on account of export of goods, other than electricity",
            scoped_clause_b,
            count=1,
        )
        if count != 1:
            raise ValueError("Manual edit Rule 89/14/2022(b)(i): clause (b) anchor not found")
        text = (
            text[: application_offset + clause_b.start()]
            + scoped_clause_b
            + text[application_offset + clause_b.end() :]
        )
        clause_b_end = application_offset + clause_b.end()
        # The clause insertion follows clause (b), prior to clause (c).
        clause_b_end += len("on account of export of goods, other than electricity") - len(
            "on account of export of goods"
        )
        text = text[:clause_b_end] + " " + inserted_clause + text[clause_b_end:]

        subrule5_start = re.search(
            r"\(5\)\s*In the case of refund on account of inverted duty structure",
            text,
            re.IGNORECASE,
        )
        if subrule5_start is None:
            raise ValueError("Manual edit Rule 89/14/2022(c): sub-rule (4) not found")
        text = (
            text[: subrule5_start.start()]
            + " "
            + inserted_explanation4
            + text[subrule5_start.start() :]
        )

        old_formula = "tax payable on such inverted rated supply of goods and services"
        subrule5 = re.search(r"\(5\)[\s\S]*$", text, re.IGNORECASE)
        if subrule5 is None:
            raise ValueError("Manual edit Rule 89/14/2022(d): sub-rule (5) not found")
        scoped_subrule5 = text[subrule5.start() : subrule5.end()]
        scoped_subrule5 = _replace_source_text(
            scoped_subrule5,
            old_formula,
            new_formula,
            label="Rule 89/14/2022(d)",
        )
        text = text[: subrule5.start()] + scoped_subrule5 + text[subrule5.end() :]

        return text

    if notification == "19/2022":
        insert_event = _event_for(
            events,
            lambda value: "after the words" in value and "claiming refund" in value,
            label="Rule 89/19/2022(a)",
        )
        quoted = _quoted_texts(insert_event["instruction"])
        if len(quoted) < 2:
            raise ValueError("Could not read Notification 19/2022 Rule 89(1)(a) quoted wording")
        text = _replace_source_text(
            text,
            quoted[0],
            f"{quoted[0]} {quoted[1]}",
            label="Rule 89/19/2022(a)",
        )
        proviso = re.compile(
            r"Provided that any claim for refund relating to balance in the electronic cash ledger"
            r"[\s\S]*?as the case may be\s*:",
            re.IGNORECASE,
        )
        text, count = proviso.subn("", text, count=1)
        if count != 1:
            raise ValueError("Manual edit Rule 89/19/2022(b): first proviso anchor not found")
        for predicate, label in (
            (
                lambda value: (
                    len(_quoted_texts(value)) == 2
                    and _quoted_texts(value)[0].strip() == "Provided further that"
                ),
                "Rule 89/19/2022(c)",
            ),
            (
                lambda value: (
                    len(_quoted_texts(value)) == 2
                    and _quoted_texts(value)[0].strip() == "Provided also that"
                ),
                "Rule 89/19/2022(d)",
            ),
        ):
            event = _event_for(events, predicate, label=label)
            quoted = _quoted_texts(event["instruction"])
            if len(quoted) != 2:
                raise ValueError(f"Could not parse quoted words for {label}")
            text = _replace_source_text(text, quoted[0], quoted[1], label=label)
        return text

    if notification == "26/2022":
        insert_event = _event_for(
            events, lambda value: "following clauses" in value, label="Rule 89/26/2022(i)"
        )
        inserted = re.sub(r"\s+", " ", _inserted_text(insert_event["instruction"])).strip()
        clause_k = re.search(r"\(k\)", text, re.IGNORECASE)
        clause_k_end = re.search(r";", text[clause_k.end() :]) if clause_k is not None else None
        if clause_k is None or clause_k_end is None:
            raise ValueError("Manual edit Rule 89/26/2022(i): clause (k) end not found")
        insert_at = clause_k.end() + clause_k_end.end()
        text = text[:insert_at] + " " + inserted + text[insert_at:]
        proviso_event = _event_for(
            events, lambda value: "following proviso" in value, label="Rule 89/26/2022(ii)"
        )
        quoted = _quoted_texts(proviso_event["instruction"])
        inserted_proviso = next(
            (item for item in quoted if item.strip().startswith("Provided further")), None
        )
        if inserted_proviso is None:
            raise ValueError("Manual edit Rule 89/26/2022(ii): inserted proviso not found")
        subrule2 = re.search(
            r"\(2\)\s*The application under sub-rule \(1\)[\s\S]*?"
            r"(?=\(3\)\s*Where the application relates)",
            text,
        )
        if subrule2 is None:
            raise ValueError("Manual edit Rule 89/26/2022(ii): sub-rule (2) not found")
        clause_m = list(re.finditer(r"\(m\)", text[subrule2.start() : subrule2.end()], re.I))
        if len(clause_m) != 1:
            raise ValueError("Manual edit Rule 89/26/2022(ii): clause (m) ambiguous")
        explanation = re.search(
            r"\bExplanation\s*[.\u2013-]", text[clause_m[0].end() + subrule2.start() :]
        )
        if explanation is None:
            raise ValueError("Manual edit Rule 89/26/2022(ii): clause (m) end not found")
        insert_at = clause_m[0].end() + subrule2.start() + explanation.start()
        text = text[:insert_at] + " " + inserted_proviso + text[insert_at:]
        return text

    if notification == "38/2023":
        first = _event_for(
            events, lambda value: "third proviso" in value, label="Rule 89/38/2023(a)"
        )
        quoted = _quoted_texts(first["instruction"])
        if len(quoted) != 2:
            raise ValueError("Could not parse Rule 89/38/2023(a) substituted wording")
        text = _replace_source_text(text, quoted[0], quoted[1], label="Rule 89/38/2023(a)")
        second = _event_for(
            events, lambda value: "payment of tax" in value, label="Rule 89/38/2023(b)"
        )
        quoted = _quoted_texts(second["instruction"])
        if len(quoted) != 2:
            raise ValueError("Could not parse Rule 89/38/2023(b) inserted wording")
        clause_k = re.search(r"\(k\)[\s\S]*?(?=\(l\))", text, re.I)
        if clause_k is None:
            raise ValueError("Manual edit Rule 89/38/2023(b): clause (k) not found")
        start = clause_k.start()
        end = clause_k.end()
        text = (
            text[:start]
            + _replace_source_text(
                text[start:end],
                quoted[0],
                f"{quoted[0]} {quoted[1]}",
                label="Rule 89/38/2023(b)",
            )
            + text[end:]
        )
        return text

    if notification == "12/2024":
        subrule_event = _event_for(
            events, lambda value: "following sub-rule" in value, label="Rule 89/12/2024(i)"
        )
        subrule = re.sub(r"\s+", " ", _inserted_text(subrule_event["instruction"])).strip()
        end_subrule1a = re.search(r"\(1A\)[\s\S]*?(?=\(2\))", text, re.I)
        if end_subrule1a is None:
            raise ValueError("Manual edit Rule 89/12/2024(i): sub-rule (1A) not found")
        insert_at = end_subrule1a.end()
        text = text[:insert_at] + " " + subrule + text[insert_at:]
        clause_event = _event_for(
            events, lambda value: "after clause (ba)" in value, label="Rule 89/12/2024(ii)"
        )
        clause = re.sub(r"\s+", " ", _inserted_text(clause_event["instruction"])).strip()
        subrule2 = re.search(r"\(2\)[\s\S]*?(?=\(3\))", text, re.I)
        if subrule2 is None:
            raise ValueError("Manual edit Rule 89/12/2024(ii): sub-rule (2) not found")
        next_clause = re.search(r"\(c\)", text[subrule2.start() : subrule2.end()], re.I)
        if next_clause is None:
            raise ValueError("Manual edit Rule 89/12/2024(ii): clause (c) not found")
        insert_at = subrule2.start() + next_clause.start()
        text = text[:insert_at] + clause + " " + text[insert_at:]
        return text

    if notification == "20/2024":
        clause_b = _event_for(
            events, lambda value: "in clause (B)" in value, label="Rule 89/20/2024(a)(i)"
        )
        old = _quoted_texts(clause_b["instruction"])[0]
        text = _replace_source_text(text, old, "", label="Rule 89/20/2024(a)(i)")
        clause_c = _event_for(
            events, lambda value: "in clause (C)" in value, label="Rule 89/20/2024(a)(ii)"
        )
        old = _quoted_texts(clause_c["instruction"])[0]
        text = _replace_source_text(text, old, "", label="Rule 89/20/2024(a)(ii)")
        clause_e = _event_for(
            events, lambda value: "in clause (E)" in value, label="Rule 89/20/2024(a)(iii)"
        )
        quoted = _quoted_texts(clause_e["instruction"])
        if len(quoted) != 3:
            raise ValueError("Could not parse Rule 89/20/2024(a)(iii) substituted wording")
        subrule4 = re.search(r"\(4\)[\s\S]*?(?=\(4A\)|\(5\))", text, re.I)
        if subrule4 is None:
            raise ValueError("Manual edit Rule 89/20/2024(a)(iii): sub-rule (4) not found")
        start = subrule4.start()
        end = subrule4.end()
        text = (
            text[:start]
            + _replace_source_text(
                text[start:end],
                "excluding",
                quoted[2],
                label="Rule 89/20/2024(a)(iii)",
            )
            + text[end:]
        )
        clause_a = _event_for(
            events, lambda value: "sub-rule (5)" in value, label="Rule 89/20/2024(c)"
        )
        old = _quoted_texts(clause_a["instruction"])[0]
        text = _replace_source_text(text, old, "", label="Rule 89/20/2024(c)")
        omitted, count = re.subn(r"\[?\(4A\)[\s\S]*?(?=\[?\(5\))", "", text, count=1, flags=re.I)
        if count != 1:
            raise ValueError("Manual edit Rule 89/20/2024(b): sub-rules (4A) and (4B) not found")
        return omitted

    raise ValueError(f"No manual Rule 89 application is defined for Notification {notification}")


def _rule89_rule10b_step(notification_path: Path) -> dict[str, Any] | None:
    """Notification 35/2021 para 2(6)(i), once a commencement notification has appointed its date.

    The instruction inserts ", subject to the provisions of rule 10B," after "may file" in
    sub-rule (1) "with effect from the date as may be notified", so it is pending in the ledger
    until ``ingest.commencement`` resolves it. Returns None while it is still pending.
    """
    from ingest.commencement import instruction_label

    rows = [json.loads(line) for line in notification_path.open(encoding="utf-8") if line.strip()]
    by_number = {row.get("notification_number"): row for row in rows}
    matches = [
        amendment
        for amendment in by_number.get("35/2021", {}).get("amendments", [])
        if str(amendment.get("rule")) == "89"
        and instruction_label(str(amendment.get("instruction", ""))) == ("6", "i")
    ]
    if len(matches) != 1:
        raise ValueError("Expected one Rule 89 instruction for Notification 35/2021 para 2(6)(i)")
    amendment = matches[0]
    if amendment.get("commencement") == "pending":
        return None
    if amendment.get("commencement") != "resolved":
        raise ValueError("Notification 35/2021 para 2(6)(i) must be pending or resolved")
    commencing = by_number[amendment["commenced_by"]]
    quoted = _quoted_texts(str(amendment["instruction"]))
    if len(quoted) != 2 or quoted[0] != "may file":
        raise ValueError("Could not read the anchor and words of Notification 35/2021 para 2(6)(i)")
    return {
        "effective": date.fromisoformat(amendment["effective_date"]),
        "issue_date": date.fromisoformat(commencing["issue_date"]),
        "anchor": quoted[0],
        "words": quoted[1],
        "amended_by": (
            "Notification No. 35/2021-Central Tax (commenced by Notification No. "
            f"{amendment['commenced_by']}-Central Tax)"
        ),
        "source_url": f"data/raw/{commencing['source_file']}",
    }


def _manual_rule89_versions(
    records: list[ProvisionRecord],
    notification_path: Path = ROOT / "data" / "processed" / "amendments.jsonl",
) -> list[ProvisionRecord]:
    versions = sorted(
        (record for record in records if record.path == "CGST/r89"),
        key=lambda record: record.effective_from,
    )
    # Only the base text and the 35/2021 sub-rule (1A) version are taken from the engine; every
    # later version is rebuilt by hand below (the engine fails at 14/2022).
    if len(versions) < 2 or versions[1].effective_from != date(2021, 9, 24):
        raise ValueError(
            "Expected engine Rule 89 base and 2021-09-24 versions before manual review"
        )
    manualized = [
        record.model_copy(update={"verified": True, "verified_by": "manual"})
        for record in versions[:2]
    ]
    if "rule 10B" in manualized[1].text.split("(1A)")[0]:
        raise ValueError("Rule 89 v2 must not contain the pending rule 10B words in sub-rule (1)")
    steps: list[dict[str, Any]] = [
        {
            "notification": notification,
            "effective": effective,
            "issue_date": issue_date,
            "amended_by": f"Notification No. {notification}-Central Tax",
            "source_url": f"data/raw/ct-{notification.replace('/', '-')}.pdf",
        }
        for notification, effective, issue_date in (
            ("14/2022", date(2022, 7, 5), date(2022, 7, 5)),
            ("19/2022", date(2022, 10, 1), date(2022, 9, 28)),
            ("26/2022", date(2022, 12, 26), date(2022, 12, 26)),
            ("38/2023", date(2023, 8, 4), date(2023, 8, 4)),
            ("12/2024", date(2024, 7, 10), date(2024, 7, 10)),
            ("20/2024", date(2024, 10, 8), date(2024, 10, 8)),
        )
    ]
    rule10b = _rule89_rule10b_step(notification_path)
    if rule10b is not None:
        steps.append(rule10b)
    steps.sort(key=lambda step: step["effective"])
    if len({step["effective"] for step in steps}) != len(steps):
        raise ValueError("Two Rule 89 manual steps share an effective date; merge them by hand")

    current = manualized[1]
    result = manualized[:2]
    for step in steps:
        if step["effective"] <= current.effective_from:
            raise ValueError(f"Rule 89 step {step['amended_by']} precedes the version it amends")
        if "notification" in step:
            updated_text = _manual_rule89_edits(
                current.text, notification_path, step["notification"]
            )
        else:
            updated_text = _replace_source_text(
                current.text,
                step["anchor"],
                step["anchor"] + step["words"],
                label="Rule 89/35/2021 para 2(6)(i)",
            )
        result[-1] = result[-1].model_copy(
            update={"effective_to": step["effective"] - timedelta(days=1)}
        )
        current = current.model_copy(
            update={
                "id": f"CGST/r89@v{len(result) + 1}",
                "text": updated_text,
                "effective_from": step["effective"],
                "effective_to": None,
                "in_force_since": step["issue_date"],
                "amended_by": step["amended_by"],
                "verified": True,
                "verified_by": "manual",
                "source_url": step["source_url"],
            }
        )
        result.append(current)
    return [record for record in records if record.path != "CGST/r89"] + result


def _manual_rule86b_versions(
    records: list[ProvisionRecord],
    notification_path: Path = ROOT / "data" / "processed" / "amendments.jsonl",
) -> list[ProvisionRecord]:
    """Re-apply Notification 20/2025 para 3 to Rule 86B by hand.

    The instruction inserts clause (f) "in the first proviso, after clause (e)". Clause (e) ends
    at its item (iv), before the second proviso. The engine once put (f) inside clause (e); it
    now inserts siblings correctly, and this hand application cross-checks it.
    """
    versions = sorted(
        (record for record in records if record.path == "CGST/r86B"),
        key=lambda record: record.effective_from,
    )
    if len(versions) != 2 or versions[1].amended_by != "Notification No. 20/2025-Central Tax":
        raise ValueError("Expected Rule 86B base and Notification 20/2025 versions")
    event = _event_for(
        _rule_amendments(notification_path, "20/2025", rule="86B"),
        lambda value: "after clause (e)" in value,
        label="Rule 86B/20/2025 para 3",
    )
    inserted = _quoted_texts(str(event["instruction"]))
    if len(inserted) != 1 or not inserted[0].startswith("(f) "):
        raise ValueError("Could not read clause (f) from Notification 20/2025 para 3")
    base = versions[0]
    end_of_e = re.search(r"\(iv\)\s*a statutory body\s*:", base.text)
    if end_of_e is None or base.text.count("(iv) a statutory body") != 1:
        raise ValueError("Rule 86B: end of clause (e) not found")
    clause_f = " ".join(inserted[0].split())
    text = base.text[: end_of_e.end()] + " " + clause_f + base.text[end_of_e.end() :]
    if " ".join(text.split()) != " ".join(versions[1].text.split()):
        raise ValueError("Rule 86B: engine and hand application of 20/2025 para 3 disagree")
    result = [
        base.model_copy(update={"verified": True, "verified_by": "manual"}),
        versions[1].model_copy(update={"text": text, "verified": True, "verified_by": "manual"}),
    ]
    return [record for record in records if record.path != "CGST/r86B"] + result


ORDER_02_2018_PDF = ROOT / "data" / "raw" / "order-02-2018-ct_2018-12-31.pdf"


def _order_02_2018(proviso_text: str) -> dict[str, Any]:
    """Check the 16(4) proviso against Order No. 02/2018-Central Tax and return its provenance.

    The order (in the removal-of-difficulty annex of the consolidated Act PDF) is dated
    "New Delhi, the 31st December, 2018"; para 1 gives only the short title and there is no
    separate commencement clause, so the insertion operates from the order's date.
    """
    from ingest.notifications import parse_legal_date, read_pdf_text

    flat = " ".join(read_pdf_text(ORDER_02_2018_PDF).split())
    header = re.search(
        r"Order No\.\s*02/2018\s*-\s*Central Tax\s+New Delhi,\s*the\s+"
        r"(?P<date>\d{1,2}(?:st|nd|rd|th)?\s+\w+,\s*\d{4})",
        flat,
    )
    if header is None:
        raise ValueError("Order No. 02/2018-Central Tax not found in its source PDF")
    end = flat.find("Order No.03/2018", header.end())
    order = flat[header.start() : end if end > 0 else header.end() + 6000]
    if re.search(r"come\s+into\s+force|with\s+effect\s+from", order, re.IGNORECASE):
        raise ValueError("Order No. 02/2018 has a commencement clause; read it before dating")
    title = re.search(r"may be called the (?P<title>.+?Order, 2018)", order)
    inserted = re.search(
        r"In sub-section \(4\) of section 16 of the said Act, the following proviso shall be "
        r"inserted, namely: ?- ?\S?(?P<text>Provided that.*?March, 2019\.)",
        order,
    )
    if title is None or inserted is None:
        raise ValueError("Could not read the 16(4) proviso insertion in Order No. 02/2018")

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    if norm(inserted.group("text")) != norm(proviso_text):
        raise ValueError("Order No. 02/2018 proviso text differs from the Act's 16(4) proviso")
    return {
        "effective_from": parse_legal_date(header.group("date")),
        "amended_by": f"Order No. 02/2018-Central Tax ({title.group('title')})",
        "source_url": f"data/raw/{ORDER_02_2018_PDF.name}",
    }


def build_act_core_provisions(
    text: str,
    *,
    source_name: str = ACT_PDF.name,
) -> tuple[list[ProvisionRecord], dict[str, list[dict[str, Any]]]]:
    extracted: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for number in ("9", "16", "17"):
        body, footnote_text = _section_parts(text, number)
        extracted[number] = (body, _footnote_entries(footnote_text))

    section9 = extracted["9"][0]
    section9_3 = _clause(section9, "3", end_marker="4")
    section9_4_current = _strip_markers(_clause(section9, "4", end_marker="5"))
    note9 = next(note for note in extracted["9"][1] if note["marker"] == "1")
    old9_4 = str(note9["old_text"])

    section16 = extracted["16"][0]
    # Footnote 2 records the exact replaced text for the Explanation under 16(2)(b).
    note16 = next(note for note in extracted["16"][1] if note["marker"] == "2")
    explanation_start = re.search(r"Explanation\s*\.?\s*[-–—]", section16, re.IGNORECASE)
    explanation_end = (
        re.search(r"\]\s*\d+\s*\[\s*\(ba\)", section16[explanation_start.start() :], re.I)
        if explanation_start
        else None
    )
    if explanation_start is None or explanation_end is None:
        raise ValueError("Could not find the current Explanation under section 16(2)(b)")
    explanation_text = section16[
        explanation_start.start() : explanation_start.start() + explanation_end.start() + 1
    ]
    current16_explanation = _strip_markers(explanation_text)
    old16_explanation = str(note16["old_text"])

    section17 = extracted["17"][0]
    note17a = next(note for note in extracted["17"][1] if note["marker"] == "2")
    current17a_match = re.search(
        r"\(a\)\s*motor vehicles for transportation of persons[\s\S]*?(?=\(aa\))",
        section17,
        re.IGNORECASE,
    )
    if current17a_match is None:
        raise ValueError("Could not find the current text of section 17(5)(a)")
    current17a = _strip_markers(current17a_match.group(0))
    old17a = str(note17a["old_text"])
    note17b = next(note for note in extracted["17"][1] if note["marker"] == "3")
    current17b_match = re.search(
        r"\(b\)\s*\d+\s*\[[\s\S]*?(?=\(c\))",
        section17,
        re.IGNORECASE,
    )
    if current17b_match is None:
        raise ValueError("Could not find the current text of section 17(5)(b)")
    current17b = _strip_markers(current17b_match.group(0))
    old17b = str(note17b["old_text"])
    # Footnote 2's bracket spans the new (a), (aa) and (ab): the 2018 substitution of the old
    # clause (a) created (aa) and (ab), so they have no version before 2019-02-01.
    flat17 = re.sub(r"\s+", " ", section17)
    current17aa = _strip_markers(
        _find(r"\(aa\)\s*vessels and aircraft.*?(?=\(ab\))", flat17, "section 17(5)(aa)")
    )
    current17ab = _strip_markers(
        _find(r"\(ab\)\s*services of general insurance.*?(?=\(b\))", flat17, "section 17(5)(ab)")
    )

    notes16 = extracted["16"][1]
    flat16 = re.sub(r"\s+", " ", section16.replace("​", ""))
    note16aa = _note(notes16, "1", "section 16(2)(aa)")
    current16aa = _strip_markers(
        _find(r"\[\s*\(aa\).*?section 37\s*;\s*\]", flat16, "section 16(2)(aa)")
    )
    note16ba = _note(notes16, "3", "section 16(2)(ba)")
    current16ba = _strip_markers(
        _find(r"\[\s*\(ba\).*?restricted\s*;\s*\]", flat16, "section 16(2)(ba)")
    )
    # Section 16(4): footnote 7 (FA 2020) omitted "invoice relating to such" w.e.f. 2021-01-01;
    # footnote 6 (FA 2022) substituted "thirtieth day of November" w.e.f. 2022-10-01.
    note16_omit = _note(notes16, "7", "section 16(4)")
    note16_nov = _note(notes16, "6", "section 16(4)")
    current16_4 = _strip_markers(
        _find(r"\(4\)\s*A registered person.*?whichever is earlier\.", flat16, "section 16(4)")
    )
    if "thirtieth day of November" not in current16_4 or "invoice or debit note pertains" not in (
        current16_4
    ):
        raise ValueError("Section 16(4) current text does not show the FA 2020/2022 amendments")
    sept16_4 = current16_4.replace("thirtieth day of November", str(note16_nov["old_text"]))
    original16_4 = sept16_4.replace(
        "such invoice or debit note", f"such invoice or {note16_omit['old_text']} debit note"
    )
    note16_proviso = _note(notes16, "8", "section 16(4) proviso")
    if not str(note16_proviso["notification"]).startswith("Order No. 02/2018"):
        raise ValueError("Section 16(4) footnote 8 no longer names Order No. 02/2018")
    current16_4_proviso = _strip_markers(
        _find(
            r"Provided that the registered person shall be entitled.*?March, 2019\.",
            flat16,
            "section 16(4) proviso",
        )
    )

    order16 = _order_02_2018(current16_4_proviso)
    source_url = f"data/raw/{source_name}"
    records = [
        _record(
            path="CGST/s9/ss3",
            node_type="subsection",
            text=section9_3,
            version=1,
            effective_from=ACT_BASE_DATE,
            effective_to=None,
            amended_by=None,
            source_url=source_url,
        ),
        _record(
            path="CGST/s9/ss4",
            node_type="subsection",
            text=old9_4,
            version=1,
            effective_from=ACT_BASE_DATE,
            effective_to=AMENDMENT_EFFECTIVE_DATE - timedelta(days=1),
            amended_by=None,
            source_url=source_url,
        ),
        _record(
            path="CGST/s9/ss4",
            node_type="subsection",
            text=section9_4_current,
            version=2,
            effective_from=AMENDMENT_EFFECTIVE_DATE,
            effective_to=None,
            amended_by="Section 4, Central Goods and Services Tax (Amendment) Act, 2018",
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss2/cl(b)/explanation1",
            node_type="explanation",
            text=old16_explanation,
            version=1,
            effective_from=ACT_BASE_DATE,
            effective_to=AMENDMENT_EFFECTIVE_DATE - timedelta(days=1),
            amended_by=None,
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss2/cl(b)/explanation1",
            node_type="explanation",
            text=current16_explanation,
            version=2,
            effective_from=AMENDMENT_EFFECTIVE_DATE,
            effective_to=None,
            amended_by="Section 8, Central Goods and Services Tax (Amendment) Act, 2018",
            source_url=source_url,
        ),
        _record(
            path="CGST/s17/ss5/cl(a)",
            node_type="clause",
            text=old17a,
            version=1,
            effective_from=ACT_BASE_DATE,
            effective_to=AMENDMENT_EFFECTIVE_DATE - timedelta(days=1),
            amended_by=None,
            source_url=source_url,
        ),
        _record(
            path="CGST/s17/ss5/cl(b)",
            node_type="clause",
            text=old17b,
            version=1,
            effective_from=ACT_BASE_DATE,
            effective_to=AMENDMENT_EFFECTIVE_DATE - timedelta(days=1),
            amended_by=None,
            source_url=source_url,
        ),
        _record(
            path="CGST/s17/ss5/cl(b)",
            node_type="clause",
            text=current17b,
            version=2,
            effective_from=AMENDMENT_EFFECTIVE_DATE,
            effective_to=None,
            amended_by="Section 9, Central Goods and Services Tax (Amendment) Act, 2018",
            source_url=source_url,
        ),
        _record(
            path="CGST/s17/ss5/cl(a)",
            node_type="clause",
            text=current17a,
            version=2,
            effective_from=AMENDMENT_EFFECTIVE_DATE,
            effective_to=None,
            amended_by="Section 9, Central Goods and Services Tax (Amendment) Act, 2018",
            source_url=source_url,
        ),
        _record(
            path="CGST/s17/ss5/cl(aa)",
            node_type="clause",
            text=current17aa,
            version=1,
            effective_from=AMENDMENT_EFFECTIVE_DATE,
            effective_to=None,
            amended_by="Section 9, Central Goods and Services Tax (Amendment) Act, 2018",
            source_url=source_url,
        ),
        _record(
            path="CGST/s17/ss5/cl(ab)",
            node_type="clause",
            text=current17ab,
            version=1,
            effective_from=AMENDMENT_EFFECTIVE_DATE,
            effective_to=None,
            amended_by="Section 9, Central Goods and Services Tax (Amendment) Act, 2018",
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss2/cl(aa)",
            node_type="clause",
            text=current16aa,
            version=1,
            effective_from=date.fromisoformat(note16aa["effective_from"]),
            effective_to=None,
            amended_by=f"Section 109, Finance Act, 2021 ({note16aa['notification']})",
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss2/cl(ba)",
            node_type="clause",
            text=current16ba,
            version=1,
            effective_from=date.fromisoformat(note16ba["effective_from"]),
            effective_to=None,
            amended_by=f"Section 100, Finance Act, 2022 ({note16ba['notification']})",
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss4",
            node_type="subsection",
            text=original16_4,
            version=1,
            effective_from=ACT_BASE_DATE,
            effective_to=date.fromisoformat(note16_omit["effective_from"]) - timedelta(days=1),
            amended_by=None,
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss4",
            node_type="subsection",
            text=sept16_4,
            version=2,
            effective_from=date.fromisoformat(note16_omit["effective_from"]),
            effective_to=date.fromisoformat(note16_nov["effective_from"]) - timedelta(days=1),
            amended_by=f"Section 120, Finance Act, 2020 ({note16_omit['notification']})",
            source_url=source_url,
        ),
        _record(
            path="CGST/s16/ss4",
            node_type="subsection",
            text=current16_4,
            version=3,
            effective_from=date.fromisoformat(note16_nov["effective_from"]),
            effective_to=None,
            amended_by=f"Section 100, Finance Act, 2022 ({note16_nov['notification']})",
            source_url=source_url,
        ),
        # Footnote 8 names Order No. 02/2018-Central Tax without a w.e.f. date; the order itself
        # supplies the date and the inserted text (checked against the Act's proviso).
        _record(
            path="CGST/s16/ss4/proviso1",
            node_type="proviso",
            text=current16_4_proviso,
            version=1,
            effective_from=order16["effective_from"],
            effective_to=None,
            amended_by=order16["amended_by"],
            source_url=order16["source_url"],
        ),
    ]
    footnotes = {number: notes for number, (_, notes) in extracted.items()}
    return records, footnotes


def build_core_provisions() -> tuple[list[ProvisionRecord], dict[str, list[dict[str, Any]]]]:
    records, footnotes, _ = build_core_provisions_with_post_pdf()
    return records, footnotes


def build_core_provisions_with_post_pdf() -> tuple[
    list[ProvisionRecord], dict[str, list[dict[str, Any]]], dict[str, Any]
]:
    """Core records, the PDF footnotes, and the post-2022 report (HTML footnotes, pending)."""
    from ingest.act_html import latest_snapshot, load_pages, snapshot_date
    from ingest.act_post_pdf import build_post_pdf, check_pdf_chains_still_current

    pdf_text = _read_pdf_text(ACT_PDF)
    records, footnotes = build_act_core_provisions(pdf_text)
    snapshot = latest_snapshot()
    pages = load_pages(snapshot)
    check_pdf_chains_still_current(pages, records)
    post_records, pending, html_footnotes = build_post_pdf(pages, pdf_text)
    records = records + post_records
    post_pdf = {
        "html_snapshot": snapshot.relative_to(ROOT).as_posix(),
        "html_snapshot_date": snapshot_date(snapshot),
        "records": [record.id for record in post_records],
        "pending": pending,
        "footnotes": html_footnotes,
    }
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    act_paths = {path for row in manifest["act_provisions"] for path in row["paths"]}
    if act_paths != {record.path for record in records}:
        raise ValueError("Act records built from footnotes do not match core_provisions.yaml")
    rule_paths = {path for rule in manifest["rules"] for path in rule["paths"]}
    current = load_processed_records(DEFAULT_PROCESSED_FILE)
    rules = [record for record in current if record.path in rule_paths]
    # Versions the engine applied cleanly keep that provenance; only the flagged ones
    # (Rules 42, 43, 89) are re-done by hand below and marked "manual".
    for record in rules:
        if record.verified and record.verified_by is None:
            record.verified_by = "engine"
    rules = _apply_core_rule_edits(rules)
    rules = _manual_rule43_versions(rules)
    rules = _manual_rule89_versions(rules)
    rules = _manual_rule86b_versions(rules)
    rules = _attach_rule_sources(
        rules,
        ROOT / "data" / "processed" / "amendments.jsonl",
    )
    return rules + records, footnotes, post_pdf


def _attach_rule_sources(
    records: list[ProvisionRecord],
    notification_path: Path,
) -> list[ProvisionRecord]:
    source_by_notification: dict[str, str] = {}
    with notification_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            notification = json.loads(line)
            source = notification.get("source_file")
            number = notification.get("notification_number")
            if source and number:
                source_by_notification[str(number)] = f"data/raw/{source}"

    sourced: list[ProvisionRecord] = []
    for record in records:
        if record.doc_type != "rule":
            sourced.append(record)
            continue
        if record.source_url is not None and (ROOT / record.source_url).is_file():
            sourced.append(record)
            continue
        if record.path in {"CGST/r36", "CGST/r36/ss4"} and record.effective_from == date(
            2021, 5, 1
        ):
            source_url = RULES_BASE_SOURCE
        elif record.amended_by:
            match = re.search(r"Notification No\.\s*(\d+/\d+)-Central Tax", record.amended_by)
            if match is None or match.group(1) not in source_by_notification:
                raise ValueError(f"Could not resolve source PDF for {record.amended_by}")
            source_url = source_by_notification[match.group(1)]
        else:
            source_url = RULES_BASE_SOURCE
        sourced.append(record.model_copy(update={"source_url": source_url}))
    return sourced


def write_core_provisions() -> list[ProvisionRecord]:
    records, footnotes, post_pdf = build_core_provisions_with_post_pdf()
    write_processed_records(records, OUTPUT_PATH)
    FOOTNOTES_PATH.write_text(
        json.dumps(footnotes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    POST_PDF_PATH.write_text(
        json.dumps(post_pdf, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def version_tables(records: list[ProvisionRecord]) -> str:
    """One Markdown table per manifest provision: version, dates, source, verified."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    groups = [(f"Rule {row['rule']}", row["paths"]) for row in manifest["rules"]] + [
        (row["citation"], row["paths"]) for row in manifest["act_provisions"]
    ]
    out: list[str] = []
    for title, paths in groups:
        out += [
            f"### {title}",
            "",
            "| Version | From | To | Amended by | Source | Verified |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        rows = sorted(
            (record for record in records if record.path in paths),
            key=lambda record: (paths.index(record.path), record.effective_from),
        )
        for record in rows:
            verified = f"yes ({record.verified_by})" if record.verified else "NO"
            out.append(
                f"| {record.id} | {record.effective_from} | {record.effective_to or 'current'} "
                f"| {record.amended_by or 'base text'} | {record.source_url or '-'} | {verified} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    print(version_tables(write_core_provisions()))


if __name__ == "__main__":
    main()

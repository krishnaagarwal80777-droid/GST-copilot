from __future__ import annotations

import json
import re
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from gst_copilot.models import ProvisionRecord
from ingest.index import DEFAULT_PROCESSED_FILE, load_processed_records, write_processed_records

ROOT = Path(__file__).resolve().parents[1]
RULES_PDF = ROOT / "data" / "raw" / "cgst-rules-part-a_2021-05-18.pdf"
AMENDMENTS_PATH = ROOT / "data" / "processed" / "amendments.jsonl"
BASE_DATE = date(2021, 5, 18)
RULE_36_TITLE = "Documentary requirements and conditions for claiming input tax credit"


def _read_pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _extract_rule_36(text: str) -> tuple[str, dict[str, str]]:
    start = re.search(
        r"(?im)^\s*36\.\s*Documentary requirements and conditions for claiming input tax credit",
        text,
    )
    if start is None:
        raise ValueError("Could not find Rule 36 in the consolidated CGST Rules PDF")
    end = re.search(r"(?m)^\s*37\.", text[start.end() :])
    section_end = start.end() + end.start() if end is not None else len(text)
    section = text[start.start() : section_end]

    cleaned_lines: list[str] = []
    for line in section.splitlines():
        if re.match(r"\s*Page\s+\d+\s+of\s+\d+\s*$", line, re.IGNORECASE):
            continue
        if re.match(r"\s*\d+\s*(?:Inserted|Substituted|Subsitituted)\b", line, re.IGNORECASE):
            continue
        cleaned_lines.append(re.sub(r"\](\d+)(?=[.,;\s]|$)", "]", line))
    section = "\n".join(cleaned_lines)

    heading_match = re.search(
        r"(?is)^\s*36\.\s*(?P<heading>.*?)\s*-\s*\(1\)\s*",
        section,
    )
    if heading_match is None:
        raise ValueError("Could not split Rule 36 heading from its sub-rules")
    heading = re.sub(r"\s+", " ", heading_match.group("heading")).strip(" .-")

    first_start = heading_match.end()
    later_subrules = list(
        re.finditer(r"(?m)^\s*\[?\((?P<number>[2-4])\)\s*", section[first_start:])
    )
    starts = [(1, first_start)]
    starts.extend(
        (int(match.group("number")), first_start + match.start())
        for match in later_subrules
    )
    if [number for number, _ in starts] != [1, 2, 3, 4]:
        raise ValueError(
            "Expected Rule 36 sub-rules (1) through (4); found "
            f"{[number for number, _ in starts]}"
        )

    subrules: dict[str, str] = {}
    for index, (number, subrule_start) in enumerate(starts):
        subrule_end = starts[index + 1][1] if index + 1 < len(starts) else len(section)
        content = section[subrule_start:subrule_end].strip()
        content = re.sub(r"^\[?\(\d+\)\s*", "", content)
        content = re.sub(r"\[([^\[\]]+)\]", r"\1", content, flags=re.DOTALL)
        content = content.replace("[", "").replace("]", "")
        content = re.sub(r"\s+", " ", content).strip()
        content = re.sub(
            r"\bFORM\s*GSTR\s*-\s*(\d[A-Z]?)",
            r"FORM GSTR-\1",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r"\bFORM\s*GSTR-(\d[A-Z]?)\s*by\s*such\b",
            r"FORM GSTR-\1 by such",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\bsub\s*-\s*(section|rule)\b", r"sub-\1", content, flags=re.IGNORECASE)
        content = re.sub(r"\bdet\s+ails\b", "details", content, flags=re.IGNORECASE)
        content = re.sub(r"\bun\s+der\b", "under", content, flags=re.IGNORECASE)
        subrules[str(number)] = f"({number}) {content}"
    return heading, subrules


def _parse_date(value: Any, *, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"Missing {label} in Rule 36 amendment data")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid {label} date in Rule 36 amendment data: {value!r}") from error


def _load_amendment_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Amendment ledger not found: {path}")
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            issue_date = _parse_date(record.get("issue_date"), label="issue")
            for amendment in record.get("amendments", []):
                if str(amendment.get("rule")) != "36":
                    continue
                if amendment.get("commencement") not in (None, "fixed", "resolved"):
                    continue
                subrule = amendment.get("subrule")
                if subrule is None:
                    match = re.search(
                        r"\b(?:in|for)\s+sub\s*[- ]\s*rule\s*\((?P<subrule>\d+)\)"
                        r"\s*(?:of\s+)?rule\s+36\b"
                        r"|\bin\s+rule\s+36\b[\s\S]{0,300}?"
                        r"\b(?:in|for)\s+sub\s*[- ]\s*rule\s*\((?P<subrule2>\d+)\)",
                        amendment.get("instruction", ""),
                        re.IGNORECASE,
                    )
                    if match is not None:
                        subrule = match.group("subrule") or match.group("subrule2")
                if str(subrule) not in {"2", "3", "4"}:
                    raise ValueError(
                        "Unsupported or missing Rule 36 sub-rule target on "
                        f"amendments.jsonl line {line_number}"
                    )
                effective_value = amendment.get("effective_date") or record.get("effective_date")
                effective_date = _parse_date(effective_value, label="effective")
                if effective_date <= BASE_DATE:
                    continue
                events.append(
                    {
                        "notification_number": record.get("notification_number"),
                        "issue_date": issue_date,
                        "effective_date": effective_date,
                        "subrule": str(subrule),
                        "instruction": amendment.get("instruction", ""),
                    }
                )
    return sorted(
        events,
        key=lambda event: (
            event["effective_date"],
            event["subrule"],
            event["issue_date"],
            str(event["notification_number"]),
        ),
    )


def _plain_instruction(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = text.replace("cu mulatively", "cumulatively")
    text = re.sub(
        r"\bFORM\s*GSTR\s*-\s*(\d[A-Z]?)",
        r"FORM GSTR-\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bsub\s*-\s*(section|rule)\b", r"sub-\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdet\s+ails\b", "details", text, flags=re.IGNORECASE)
    text = re.sub(r"\bun\s+der\b", "under", text, flags=re.IGNORECASE)
    return text


def _replace_second_proviso(current: str, instruction: str) -> str:
    replacement = re.search(r"Provided further that[\s\S]*?above\.", instruction, re.IGNORECASE)
    if replacement is None:
        raise ValueError("Could not extract replacement proviso from the notification instruction")
    replacement_text = _plain_instruction(replacement.group(0))
    current_pattern = re.compile(
        r"Provided further that[\s\S]*?April\s+and\s+May,\s*2021[\s\S]*?above\.",
        re.IGNORECASE,
    )
    updated, count = current_pattern.subn(replacement_text, current, count=1)
    if count != 1:
        raise ValueError("Could not find the April-May 2021 proviso in Rule 36(4)")
    return updated


def _extract_subrule_replacement(instruction: str) -> str:
    replacement = re.search(
        r"(?is)namely[\s\S]{0,40}?(?P<text>\(4\).*?rule\s+60\.)",
        instruction,
    )
    if replacement is None:
        raise ValueError("Could not extract the substituted text for Rule 36(4)")
    return _plain_instruction(replacement.group("text"))


def _apply_instruction(subrule_text: str, subrule: str, instruction: str) -> str:
    lowered = instruction.lower()
    if subrule == "4" and "second proviso" in lowered and "substituted" in lowered:
        return _replace_second_proviso(subrule_text, instruction)

    if subrule == "4" and "following sub-rule shall be substituted" in lowered:
        return _extract_subrule_replacement(instruction)

    if subrule == "4" and "clause (b)" in lowered and "shall be inserted" in lowered:
        clause_start = re.search(r"\(\s*b\s*\)", subrule_text, re.IGNORECASE)
        if clause_start is None:
            raise ValueError("Could not locate clause (b) in Rule 36(4)")
        prefix, clause = subrule_text[: clause_start.end()], subrule_text[clause_start.end() :]
        updated, count = re.subn(
            r"\bthe details of\b",
            "the details of input tax credit in respect of",
            clause,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise ValueError("Could not apply the clause (b) insertion to Rule 36(4)")
        return prefix + updated

    if subrule == "4" and "clause (a)" in lowered and "shall be inserted" in lowered:
        phrase = re.search(
            r"as amended in FORM\s*GSTR\s*-\s*1A if any",
            instruction,
            re.IGNORECASE,
        )
        if phrase is None:
            raise ValueError("Could not extract the inserted phrase for Rule 36(4)(a)")
        clause_start = re.search(r"\(\s*a\s*\)", subrule_text, re.IGNORECASE)
        clause_end = re.search(r"\(\s*b\s*\)", subrule_text, re.IGNORECASE)
        if clause_start is None or clause_end is None:
            raise ValueError("Could not locate clause (a) in Rule 36(4)")
        prefix = subrule_text[: clause_start.end()]
        clause = subrule_text[clause_start.end() : clause_end.start()]
        updated, count = re.subn(
            r"FORM\s*GSTR\s*-\s*1\b",
            lambda match: f"{match.group(0)}, {phrase.group(0)},",
            clause,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise ValueError("Could not apply the clause (a) insertion to Rule 36(4)")
        return prefix + updated + subrule_text[clause_end.start() :]

    if subrule == "2" and "shall be omitted" in lowered:
        updated, count = re.subn(
            r",?\s*and\s+the\s+relevant\s+information,\s*as\s+contained\s+"
            r"in\s+the\s+said\s+document,\s*is\s+furnished\s+in\s*"
            r"FORM\s*GSTR\s*-\s*2\s*by\s*such\s+person",
            "",
            subrule_text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise ValueError("Could not apply the omission to Rule 36(2)")
        return updated

    if subrule == "3" and "suppression of facts" in lowered and "shall be inserted" in lowered:
        updated, count = re.subn(
            r"\bsuppression of facts\b",
            "suppression of facts under section 74",
            subrule_text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise ValueError("Could not apply the insertion to Rule 36(3)")
        return updated

    raise ValueError(f"Unsupported Rule 36 amendment instruction: {instruction[:180]!r}")


def build_rule_36_provisions(
    raw_pdf: str | Path = RULES_PDF,
    amendments_path: str | Path = AMENDMENTS_PATH,
) -> list[ProvisionRecord]:
    heading, subrules = _extract_rule_36(_read_pdf_text(Path(raw_pdf)))
    versions: dict[str, list[ProvisionRecord]] = {}

    root_record = ProvisionRecord(
        id="CGST/r36@v1",
        doc_type="rule",
        act_id="CGST2017",
        path="CGST/r36",
        node_type="rule",
        heading=heading,
        text="",
        context_prefix=heading,
        effective_from=date(2021, 5, 1),
        effective_to=None,
        in_force_since=date(2021, 5, 1),
        amended_by="Notification No. 13/2021-Central Tax",
    )
    for number, text in subrules.items():
        path = f"CGST/r36/ss{number}"
        record = ProvisionRecord(
            id=f"{path}@v1",
            doc_type="rule",
            act_id="CGST2017",
            path=path,
            node_type="subsection",
            heading=None,
            text=text,
            context_prefix=heading,
            effective_from=BASE_DATE,
            effective_to=None,
            in_force_since=BASE_DATE,
            amended_by=None,
        )
        versions[path] = [record]

    events = _load_amendment_events(Path(amendments_path))
    for (_, subrule), grouped_events in groupby(
        events,
        key=lambda event: (event["effective_date"], event["subrule"]),
    ):
        same_effective_change = list(grouped_events)
        path = f"CGST/r36/ss{subrule}"
        current = versions[path][-1]
        updated_text = current.text
        for event in same_effective_change:
            updated_text = _apply_instruction(updated_text, subrule, str(event["instruction"]))
        if updated_text == current.text:
            notifications = ", ".join(
                str(event["notification_number"]) for event in same_effective_change
            )
            raise ValueError(f"Rule 36 instructions did not change {path}: {notifications}")
        effective_date = same_effective_change[0]["effective_date"]
        issue_date = max(event["issue_date"] for event in same_effective_change)
        closed = current.model_copy(update={"effective_to": effective_date - timedelta(days=1)})
        versions[path][-1] = closed
        references = dict.fromkeys(
            f"Notification No. {event['notification_number']}-Central Tax"
            for event in same_effective_change
        )
        reference = "; ".join(references)
        next_record = ProvisionRecord(
            id=f"{path}@v{len(versions[path]) + 1}",
            doc_type="rule",
            act_id="CGST2017",
            path=path,
            node_type="subsection",
            heading=None,
            text=updated_text,
            context_prefix=heading,
            effective_from=effective_date,
            effective_to=None,
            in_force_since=issue_date,
            amended_by=reference,
        )
        versions[path].append(next_record)

    for path_versions in versions.values():
        base = path_versions[0]
        path_versions[0] = base.model_copy(
            update={
                "in_force_since": base.effective_from,
            }
        )
    base_rule4 = versions["CGST/r36/ss4"][0]
    versions["CGST/r36/ss4"][0] = base_rule4.model_copy(
        update={
            "effective_from": date(2021, 5, 1),
            "in_force_since": date(2021, 5, 1),
            "amended_by": "Notification No. 13/2021-Central Tax",
        }
    )
    records = [root_record]
    records.extend(record for path_versions in versions.values() for record in path_versions)
    return sorted(records, key=lambda record: (record.path, record.effective_from, record.id))


def write_rule_36_provisions(
    output_path: str | Path = DEFAULT_PROCESSED_FILE,
    raw_pdf: str | Path = RULES_PDF,
    amendments_path: str | Path = AMENDMENTS_PATH,
) -> list[ProvisionRecord]:
    output = Path(output_path)
    existing = load_processed_records(output)
    preserved = [
        record
        for record in existing
        if record.path != "CGST/r36" and not record.path.startswith("CGST/r36/")
    ]
    generated = build_rule_36_provisions(raw_pdf, amendments_path)
    write_processed_records(preserved + generated, output)
    return generated


def main() -> None:
    generated = write_rule_36_provisions()
    print(f"Wrote {len(generated)} Rule 36 provision versions to {DEFAULT_PROCESSED_FILE}")


if __name__ == "__main__":
    main()

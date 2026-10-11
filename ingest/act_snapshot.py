"""Snapshot segmentation of the full CGST Act and Rules, used only as retrieval distractors.

The hand-verified, versioned records cover a few provisions. For a fair retrieval comparison
the index must also hold the rest of the Act and Rules, so the ranker has to find the gold
version among realistic neighbours. These records are the text of one snapshot each
(cgst-act_cbic-taxinfo_2022-09-28.pdf; the Rules base snapshot of 2021-05-18), are not
versioned or reviewed, and carry ``servable=False``: retrievable for scoring, never citable.

Segmentation: Act sections split at sub-section markers "(1)", "(2)", ... (so section 2 yields
one record per definition); where verified records exist below a sub-section, that
sub-section is split at clause markers instead. Rules: one record per rule. Any node whose
path already has a verified record gets no distractor copy.

Bitemporal fields: snapshot text is not dated per provision, so distractors are given
effective_from 2017-07-01 with no end; they compete on every reference date, as noise should.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from gst_copilot.models import ProvisionRecord
from ingest.all_rule_versions import RULES_PDF, _clause_successors, _extract_rule_bases
from ingest.core_provisions import ACT_PDF, _read_pdf_text, _strip_markers
from ingest.index import load_processed_records, write_processed_records

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "distractors.jsonl"
VERIFIED_SOURCES = (
    ROOT / "data" / "processed" / "core_provisions.jsonl",
    ROOT / "data" / "processed" / "provisions.jsonl",
)
SNAPSHOT_START = date(2017, 7, 1)
_HEADER = re.compile(
    r"(?m)^[ \t]*(?:\*[ \t]*\n?[ \t]*)?(?:\d+[ \t]*\n?[ \t]*\[[ \t]*)?"
    r"Section[ \t]*\n?[ \t]*(?P<num>\d+)(?P<suffix>[A-Z]*)\.[ \t]*\n?(?P<title>[^\n]*)"
)


def _key(number: str, suffix: str) -> tuple[int, str]:
    return int(number), suffix


def act_sections(text: str) -> list[tuple[str, str, str]]:
    """(section number, heading, body) in Act order; footnote echoes and omitted sections skipped."""
    accepted = []
    last: tuple[int, str] = (0, "")
    for match in _HEADER.finditer(text):
        key = _key(match.group("num"), match.group("suffix"))
        if key <= last:
            continue  # a footnote quoting an earlier section's old text
        accepted.append(match)
        last = key
    sections = []
    for index, match in enumerate(accepted):
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        chunk = text[match.end() : end]
        enforced = re.search(r"(?im)^\s*\*?\s*Enforced\b", chunk)
        body = chunk[: enforced.start()] if enforced else chunk
        title = " ".join(match.group("title").split()).rstrip(".-– ")
        number = match.group("num") + match.group("suffix")
        cleaned = _strip_markers(body)
        if re.fullmatch(r"[\s*.\-–]*", cleaned):
            continue  # omitted section ("[***]")
        sections.append((number, title, body))
    return sections


def _split_subsections(body: str) -> list[tuple[str | None, str]]:
    """Split at line-start "(N)" markers in order; text before the first goes with no marker."""
    marks = []
    expected = 1
    for match in re.finditer(r"(?m)^[ \t]*(?:\d+[ \t]*\[[ \t]*)?\((\d+[A-Z]?)\)", body):
        number = int(re.match(r"\d+", match.group(1)).group())
        if number in (expected, expected - 1) and (not marks or match.group(1) != marks[-1][0]):
            marks.append((match.group(1), match.start()))
            expected = number + 1
    if not marks:
        return [(None, body)]
    parts = [(None, body[: marks[0][1]])] if body[: marks[0][1]].strip() else []
    for index, (marker, start) in enumerate(marks):
        end = marks[index + 1][1] if index + 1 < len(marks) else len(body)
        parts.append((marker, body[start:end]))
    return parts


def _split_clauses(text: str) -> list[tuple[str | None, str]]:
    """Split a sub-section at its lettered clauses, following (a) -> (aa)/(b) -> ... in order."""
    flat = " ".join(text.split())
    positions: list[tuple[str, int]] = []
    current = None
    cursor = 0
    while True:
        wanted = {"a"} if current is None else _clause_successors(current)
        found = None
        for match in re.finditer(r"(?:^|[\s;:\-–\[])\(([a-z]{1,2})\)\s", flat[cursor:]):
            if match.group(1) in wanted:
                found = match
                break
        if found is None:
            break
        start = cursor + found.start(1) - 1
        positions.append((found.group(1), start))
        current = found.group(1)
        cursor = start + 3
    if not positions:
        return [(None, flat)]
    parts = [(None, flat[: positions[0][1]])]
    for index, (marker, start) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(flat)
        parts.append((marker, flat[start:end]))
    return parts


def _distractor(path: str, node_type: str, heading: str, text: str, source: str) -> ProvisionRecord:
    return ProvisionRecord(
        id=f"{path}@snapshot",
        doc_type="act" if path.startswith("CGST/s") else "rule",
        act_id="CGST2017",
        path=path,
        node_type=node_type,
        heading=heading or None,
        text=text,
        effective_from=SNAPSHOT_START,
        effective_to=None,
        in_force_since=SNAPSHOT_START,
        source_url=source,
        verified=False,
        servable=False,
    )


def act_distractors(verified_paths: set[str]) -> list[ProvisionRecord]:
    source = f"data/raw/{ACT_PDF.name}"
    records = []
    for number, heading, body in act_sections(_read_pdf_text(ACT_PDF)):
        section = f"CGST/s{number}"
        for marker, part in _split_subsections(body):
            path = section if marker is None else f"{section}/ss{marker}"
            text = _strip_markers(part)
            if not text or path in verified_paths:
                continue
            if any(other.startswith(path + "/") for other in verified_paths):
                for clause, piece in _split_clauses(part):
                    clause_path = path if clause is None else f"{path}/cl({clause})"
                    piece = _strip_markers(piece)
                    if piece and clause_path not in verified_paths:
                        node = "subsection" if clause is None else "clause"
                        records.append(_distractor(clause_path, node, heading, piece, source))
                continue
            node = "section" if marker is None else "subsection"
            records.append(_distractor(path, node, heading, text, source))
    return records


def rule_distractors(verified_paths: set[str]) -> list[ProvisionRecord]:
    source = f"data/raw/{RULES_PDF.name}"
    records = []
    for number, (heading, text) in _extract_rule_bases(_read_pdf_text(RULES_PDF)).items():
        path = f"CGST/r{number}"
        if path in verified_paths or not text.strip():
            continue
        records.append(_distractor(path, "rule", heading, " ".join(text.split()), source))
    return records


def build_distractors() -> list[ProvisionRecord]:
    verified_paths = {
        record.path
        for source in VERIFIED_SOURCES
        for record in load_processed_records(source)
        if record.verified
    }
    records = act_distractors(verified_paths) + rule_distractors(verified_paths)
    ids = [record.id for record in records]
    duplicates = {record_id for record_id in ids if ids.count(record_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate distractor paths: {sorted(duplicates)[:10]}")
    return records


def main() -> None:
    records = build_distractors()
    write_processed_records(records, OUTPUT)
    acts = sum(record.doc_type == "act" for record in records)
    print(f"wrote {len(records)} distractors ({acts} Act, {len(records) - acts} Rules) to {OUTPUT}")


if __name__ == "__main__":
    main()

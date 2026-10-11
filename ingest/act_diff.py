"""Diff the current CGST Act (CBIC section pages) against the 2022-09-28 consolidated PDF.

Purpose: the HTML snapshot replaces the PDF as the newest Act source, so every difference
between the two must be a post-2022 amendment. For each section this compares the two texts
word by word (footnote markers, brackets, case and punctuation ignored) and ties each
differing span to the footnote marker(s) open at that point in the HTML. A span is
``explained`` when every such footnote is new (absent from the PDF's footnotes for that
section) and post-dates the PDF (w.e.f. date after 2022-09-28, or, when the footnote gives no
date, an instrument year of 2023 or later). Anything else is ``needs_review``.

Run: ``python -m ingest.act_diff`` -> data/processed/act_html_diff.json and a summary.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ingest.act_html import SectionPage, load_pages, snapshot_date
from ingest.act_snapshot import _HEADER, _key
from ingest.core_provisions import ACT_PDF, _footnote_entries, _read_pdf_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "act_html_diff.json"
PDF_DATE = date(2022, 9, 28)
_ENFORCED = re.compile(r"(?im)^\s*\*?\s*Enforced\b")
_TOKEN = re.compile(
    r"(?P<omit>\d+)\s*\[\s*\*+\s*\]|(?P<marker>\d+)\s*\[|(?P<close>\])|(?P<word>[A-Za-z0-9]+)"
)
PROXIMITY = 12  # words
# PDF-only text between sections: a chapter heading or a repeated section header.
_PDF_ARTIFACT = re.compile(r"chapter [ivxl]+( [a-z]+){1,8}|section \d+[a-z]?( [a-z]+){1,10}")
_QUOTED = re.compile(r"[\"“”‘’'`]{1,2}.*?[\"“”‘’'`]{1,2}", re.S)


def pdf_sections(text: str) -> dict[str, tuple[str, str]]:
    """Section number -> (body, footnote text) from the consolidated PDF text, in Act order."""
    accepted, last = [], (0, "")
    for match in _HEADER.finditer(text):
        key = _key(match.group("num"), match.group("suffix"))
        if key <= last:
            continue  # a footnote quoting an earlier section's old text
        accepted.append(match)
        last = key
    out: dict[str, tuple[str, str]] = {}
    for index, match in enumerate(accepted):
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        out[match.group("num") + match.group("suffix")] = _split_footnotes(
            text[match.start() : end]
        )
    return out


_FIRST_NOTE = re.compile(
    r"(?m)^\s*1\s*\.\s*(?:\(|Substituted|Inserted|Omitted|Renumbered|Re-numbered|Added)"
)


def _split_footnotes(chunk: str) -> tuple[str, str]:
    """Body and footnote list. Footnotes start at the "Enforced" line or, on pages without
    one, at the first line reading "1. Substituted/Inserted/..."."""
    starts = [m.start() for m in (_ENFORCED.search(chunk), _FIRST_NOTE.search(chunk)) if m]
    if not starts:
        return chunk, ""
    cut = min(starts)
    return chunk[:cut], chunk[cut:]


def _prep(text: str) -> str:
    # "01st" / "01 st" and a bracketless marker glued to a word ("1three") read the same.
    return re.sub(r"(?<![A-Za-z0-9])(\d+)(?=[A-Za-z]{2,})", r"\1 ", text)


def _words(text: str) -> list[str]:
    return [m.group("word").lower() for m in _TOKEN.finditer(_prep(text)) if m.group("word")]


def _html_words(
    text: str,
) -> tuple[list[str], list[frozenset[str]], list[frozenset[str]], list[frozenset[str]]]:
    """Words; the innermost footnote marker open at each word; markers opened just before
    it; and omission markers ("N[***]") just before it. Nested brackets ("1[... 4[...]...]")
    credit the innermost amendment."""
    words: list[str] = []
    open_at: list[frozenset[str]] = []
    opened_before: list[frozenset[str]] = []
    omitted_before: list[frozenset[str]] = []
    stack: list[str] = []
    pending: set[str] = set()
    omits: set[str] = set()
    for match in _TOKEN.finditer(_prep(text)):
        if match.group("omit"):
            omits.add(match.group("omit"))
        elif match.group("marker"):
            stack.append(match.group("marker"))
            pending.add(match.group("marker"))
        elif match.group("close"):
            if stack:
                stack.pop()
        else:
            words.append(match.group("word").lower())
            open_at.append(frozenset(stack[-1:]))
            opened_before.append(frozenset(pending | omits))
            omitted_before.append(frozenset(omits))
            pending, omits = set(), set()
    opened_before.append(frozenset(pending | omits))  # markers after the last word
    omitted_before.append(frozenset(omits))
    return words, open_at, opened_before, omitted_before


def _norm_note(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def footnote_is_post_pdf(note: dict[str, Any]) -> bool:
    """In force after the PDF, or made by an instrument dated after it (a later Act may amend
    retrospectively, e.g. "w.e.f. 1st July, 2017 by s. 140 of The Finance Act 2023")."""
    if note.get("effective_from") and date.fromisoformat(note["effective_from"]) > PDF_DATE:
        return True
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", _QUOTED.sub(" ", note["text"]))]
    return bool(years) and max(years) > PDF_DATE.year


@dataclass
class SectionDiff:
    number: str
    status: str  # unchanged | explained | needs_review | missing_in_pdf
    hunks: list[dict[str, Any]] = field(default_factory=list)
    new_footnotes: list[dict[str, Any]] = field(default_factory=list)
    unused_new_footnotes: list[str] = field(default_factory=list)


def diff_section(page: SectionPage, pdf: tuple[str, str] | None) -> SectionDiff:
    html_body, html_notes_text = _split_footnotes(page.text())
    html_notes = _footnote_entries(html_notes_text)
    if pdf is None:
        return SectionDiff(page.number, "missing_in_pdf", new_footnotes=html_notes)
    pdf_body, pdf_notes_text = pdf
    pdf_notes = {_norm_note(n["text"]) for n in _footnote_entries(pdf_notes_text)}
    by_marker = {n["marker"]: n for n in html_notes}
    new = [n for n in html_notes if _norm_note(n["text"]) not in pdf_notes]
    new_markers = {n["marker"] for n in new}

    old_words = _words(pdf_body)
    new_words, open_at, opened_before, omitted_before = _html_words(html_body)
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    hunks, used = [], set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        markers = set(opened_before[j1])
        if tag == "delete" and omitted_before[j1]:
            markers = set(omitted_before[j1])  # "N[***]" marks exactly where words went
        for j in range(j1, j2):
            markers |= open_at[j] | opened_before[j]
        if j1 == j2 and not markers and j1 > 0:
            markers |= open_at[j1 - 1]  # deletion inside a bracket: the innermost open one
        explained = bool(markers) and all(
            m in new_markers and m in by_marker and footnote_is_post_pdf(by_marker[m])
            for m in markers
        )
        reason = "footnote" if explained else None
        pdf_text = " ".join(old_words[i1:i2])
        if not explained and tag == "delete" and _PDF_ARTIFACT.fullmatch(pdf_text):
            explained, reason = True, "pdf_artifact"
        if not explained and not markers:
            # Unmarked span (typically a deleted proviso whose replacement is bracketed just
            # before or after it): credit a new post-PDF footnote marker within a few words.
            near = set()
            for j in range(max(0, j1 - PROXIMITY), min(len(open_at), j2 + PROXIMITY)):
                near |= open_at[j] | opened_before[j]
            near = {m for m in near if m in new_markers and footnote_is_post_pdf(by_marker[m])}
            if near:
                markers, explained, reason = near, True, "nearby_footnote"
        used |= markers
        hunks.append(
            {
                "op": tag,
                "pdf": " ".join(old_words[i1:i2])[:300],
                "html": " ".join(new_words[j1:j2])[:300],
                "context": " ".join(new_words[max(0, j1 - 8) : j1]),
                "markers": sorted(markers, key=int),
                "explained": explained,
                "reason": reason,
            }
        )
    if not hunks:
        status = "unchanged"
    elif all(h["explained"] for h in hunks):
        status = "explained"
    else:
        status = "needs_review"
    return SectionDiff(
        page.number,
        status,
        hunks,
        [
            {k: n[k] for k in ("marker", "operation", "effective_from", "text")}
            | {"post_pdf": footnote_is_post_pdf(n)}
            for n in new
        ],
        sorted(new_markers - used, key=int),
    )


def run(pages: dict[str, SectionPage] | None = None, pdf_text: str | None = None) -> dict:
    pages = pages or load_pages()
    sections = pdf_sections(pdf_text if pdf_text is not None else _read_pdf_text(ACT_PDF))
    diffs = [diff_section(page, sections.get(number)) for number, page in pages.items()]
    snapshot = next(iter(pages.values())).file.parent.parent
    counts: dict[str, int] = {}
    for d in diffs:
        counts[d.status] = counts.get(d.status, 0) + 1
    return {
        "pdf": f"data/raw/{ACT_PDF.name}",
        "html_snapshot": snapshot.relative_to(ROOT).as_posix(),
        "html_snapshot_date": snapshot_date(snapshot),
        "counts": counts,
        "sections": [d.__dict__ for d in diffs],
    }


def main() -> None:
    report = run()
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"]))
    for section in report["sections"]:
        if section["status"] in ("needs_review", "missing_in_pdf"):
            bad = [h for h in section["hunks"] if not h["explained"]]
            print(f"{section['number']}: {section['status']} ({len(bad)} unexplained spans)")


if __name__ == "__main__":
    main()

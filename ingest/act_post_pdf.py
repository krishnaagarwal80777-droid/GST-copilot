"""Act versions after the 2022-09-28 consolidated PDF, built from the CBIC section pages.

``ingest.core_provisions`` builds the core Act chains (s9, s16, s17) from the footnotes of
``cgst-act_cbic-taxinfo_2022-09-28.pdf``. The CBIC per-section pages (``ingest.act_html``)
carry the same footnotes plus the later ones. This module

1. checks that the PDF-built chains still describe the current text: the same builder run on
   the HTML pages must give the same records (so no post-2022 amendment touched them);
2. builds the post-2022 versions of core nodes: the current text comes from the HTML page,
   the earlier text is the current text with the footnote's bracket put back (the footnote's
   old words for a substitution, nothing for an insertion), and every pre-2022 text is
   checked word for word against the PDF;
3. lists the post-2022 footnotes in the core sections that cannot be versioned from the
   sources held (no commencement date in the footnote, or a retrospective substitution) as
   ``pending``, with the reason and what source would resolve it. Nothing is guessed.

Bitemporal fields follow the existing convention: ``in_force_since = effective_from`` for
prospective amendments. Retrospective ones are left pending (see ``PENDING_REASONS``).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from gst_copilot.models import NodeType, ProvisionRecord
from ingest.act_diff import _norm_note, footnote_is_post_pdf
from ingest.act_html import SectionPage
from ingest.core_provisions import (
    ACT_BASE_DATE,
    ACT_PDF,
    _footnote_entries,
    _record,
    _section_parts,
    _strip_markers,
    build_act_core_provisions,
)

CORE_SECTIONS = ("9", "16", "17")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("​", "")).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _strip_markers(text).lower())


def _bracket(raw: str, marker: str) -> re.Match[str]:
    match = re.search(rf"(?<!\d){re.escape(marker)}\s*\[(?P<inner>[^\[\]]*)\]", raw)
    if match is None:
        raise ValueError(f"Footnote bracket {marker}[...] not found in: {raw[:120]}")
    return match


def _undo(raw: str, note: dict[str, Any], *, old_text: str | None = None) -> str:
    """The text before the footnote's amendment: substitution -> old words, insertion -> none."""
    match = _bracket(raw, note["marker"])
    if note["operation"] == "inserted":
        replacement = ""
    elif old_text is not None:
        replacement = old_text
    elif note["operation"] == "substituted" and note.get("old_text"):
        replacement = note["old_text"]
    else:
        raise ValueError(f"Footnote {note['marker']} gives no old text to restore")
    return raw[: match.start()] + replacement + raw[match.end() :]


def _amended_by(note: dict[str, Any]) -> str:
    instrument = note.get("instrument")
    if not instrument:
        found = re.search(r"\bby\s+((?:s\.|section)\s*\d+.*?\d{4})", note["text"], re.I)
        instrument = found.group(1) if found else note["text"]
    return f"{instrument} ({note['notification']})" if note.get("notification") else instrument


def _date(note: dict[str, Any]) -> date:
    if not note.get("effective_from"):
        raise ValueError(f"Footnote {note['marker']} has no w.e.f. date")
    return date.fromisoformat(note["effective_from"])


class _Builder:
    def __init__(self, pages: dict[str, SectionPage], pdf_text: str) -> None:
        self.pages = pages
        self.body: dict[str, str] = {}
        self.notes: dict[str, dict[str, dict[str, Any]]] = {}
        self.pdf_body: dict[str, str] = {}
        self.pdf_notes: dict[str, set[str]] = {}
        for number in CORE_SECTIONS:
            body, notes = _section_parts(pages[number].text(), number)
            self.body[number] = _flat(body)
            self.notes[number] = {n["marker"]: n for n in _footnote_entries(notes)}
            pdf_body, pdf_notes = _section_parts(pdf_text, number)
            self.pdf_body[number] = _norm(pdf_body)
            self.pdf_notes[number] = {_norm_note(n["text"]) for n in _footnote_entries(pdf_notes)}
        self.records: list[ProvisionRecord] = []
        self.used: set[tuple[str, str]] = set()

    def note(self, section: str, marker: str) -> dict[str, Any]:
        self.used.add((section, marker))
        return self.notes[section][marker]

    def find(self, section: str, pattern: str, label: str) -> str:
        match = re.search(pattern, self.body[section], re.IGNORECASE)
        if match is None:
            raise ValueError(f"Could not find {label} on the CBIC page for section {section}")
        return match.group(0)

    def in_pdf(self, section: str, text: str, label: str) -> None:
        if _norm(text) not in self.pdf_body[section]:
            raise ValueError(f"{label}: reconstructed pre-2022 text is not in the 2022 PDF")

    def not_in_pdf(self, section: str, text: str, label: str) -> None:
        if _norm(text) in self.pdf_body[section]:
            raise ValueError(f"{label}: post-2022 text already appears in the 2022 PDF")

    def add(
        self,
        section: str,
        path: str,
        node_type: NodeType,
        chain: list[tuple[str, date, str | None, str]],
    ) -> None:
        """chain: (text, effective_from, amended_by, source_url) oldest first."""
        for index, (text, start, amended_by, source_url) in enumerate(chain):
            end = chain[index + 1][1] - timedelta(days=1) if index + 1 < len(chain) else None
            self.records.append(
                _record(
                    path=path,
                    node_type=node_type,
                    text=text,
                    version=index + 1,
                    effective_from=start,
                    effective_to=end,
                    amended_by=amended_by,
                    source_url=source_url,
                )
            )

    def build(self) -> list[ProvisionRecord]:
        pdf = f"data/raw/{ACT_PDF.name}"
        html16 = self.pages["16"].source_url
        html17 = self.pages["17"].source_url

        # s16(2), second proviso: FA 2023 s.138 substituted the repayment words (fn 9).
        raw = self.find(
            "16", r"Provided further that where a recipient fails.*?prescribed:", "s16(2) proviso 2"
        )
        note = self.note("16", "9")
        old = _strip_markers(_undo(raw, note))
        new = _strip_markers(raw)
        self.in_pdf("16", old, "s16(2) proviso 2")
        self.not_in_pdf("16", new, "s16(2) proviso 2")
        self.add(
            "16",
            "CGST/s16/ss2/proviso2",
            "proviso",
            [(old, ACT_BASE_DATE, None, pdf), (new, _date(note), _amended_by(note), html16)],
        )

        # s16(2), third proviso: FA 2023 s.138 inserted "to the supplier" (fn 10).
        raw = self.find(
            "16",
            r"Provided also that the recipient shall be entitled.*?thereon\.",
            "s16(2) proviso 3",
        )
        note = self.note("16", "10")
        old, new = _strip_markers(_undo(raw, note)), _strip_markers(raw)
        self.in_pdf("16", old, "s16(2) proviso 3")
        self.not_in_pdf("16", new, "s16(2) proviso 3")
        self.add(
            "16",
            "CGST/s16/ss2/proviso3",
            "proviso",
            [(old, ACT_BASE_DATE, None, pdf), (new, _date(note), _amended_by(note), html16)],
        )

        # s17(3) Explanation: inserted from 2019-02-01 (fn 1), "except" limb substituted by
        # FA 2023 s.139 from 2023-10-01 (fn 4).
        raw = self.find(
            "17",
            r"\d+\s*\[\s*Explanation\s*\.?\s*-\s*For the purposes of this sub-section"
            r".*?\]\s*;\s*\]",
            "s17(3) Explanation",
        )
        inserted, note = self.note("17", "1"), self.note("17", "4")
        old = _strip_markers(_undo(raw, note))
        new = _strip_markers(raw)
        self.in_pdf("17", old, "s17(3) Explanation")
        self.not_in_pdf("17", new, "s17(3) Explanation")
        self.add(
            "17",
            "CGST/s17/ss3/explanation1",
            "explanation",
            [
                (old, _date(inserted), _amended_by(inserted), pdf),
                (new, _date(note), _amended_by(note), html17),
            ],
        )

        # s17(5)(fa): inserted by FA 2023 s.139 from 2023-10-01 (fn 5).
        raw = self.find("17", r"\d+\s*\[\s*\(fa\).*?Companies Act, 2013;\s*\]", "s17(5)(fa)")
        note = self.note("17", "5")
        new = _strip_markers(raw)
        self.not_in_pdf("17", new, "s17(5)(fa)")
        self.add(
            "17",
            "CGST/s17/ss5/cl(fa)",
            "clause",
            [(new, _date(note), _amended_by(note), html17)],
        )

        # s17(5) Explanation after clause (d): renumbered "Explanation 1" from 2025-10-01 (fn 8);
        # Explanation 2 inserted from 2025-10-01 (fn 9).
        raw = self.find(
            "17",
            r"\d+\s*\[\s*Explanation\s*1\s*\]\s*\.?\s*-\s*For the purposes of clauses \(c\) and "
            r"\(d\).*?immovable property;",
            "s17(5) Explanation 1",
        )
        note = self.note("17", "8")
        old = _strip_markers(_undo(raw, note, old_text="Explanation"))
        new = _strip_markers(raw)
        self.in_pdf("17", old, "s17(5) Explanation (before renumbering)")
        self.add(
            "17",
            "CGST/s17/ss5/explanation1",
            "explanation",
            [(old, ACT_BASE_DATE, None, pdf), (new, _date(note), _amended_by(note), html17)],
        )
        note = self.note("17", "9")
        raw = _bracket(self.body["17"], "9").group(0)
        new = _strip_markers(raw)
        if not new.startswith("Explanation 2"):
            raise ValueError("s17(5) footnote 9 bracket is not Explanation 2")
        self.not_in_pdf("17", new, "s17(5) Explanation 2")
        self.add(
            "17",
            "CGST/s17/ss5/explanation2",
            "explanation",
            [(new, _date(note), _amended_by(note), html17)],
        )
        return self.records

    def pending(self) -> list[dict[str, Any]]:
        """Footnotes new since the PDF in the core sections that were not versioned."""
        rows = []
        for section in CORE_SECTIONS:
            for marker, note in self.notes[section].items():
                new = _norm_note(note["text"]) not in self.pdf_notes[section]
                if (section, marker) in self.used or not new:
                    continue
                if not footnote_is_post_pdf(note):
                    raise ValueError(f"Section {section} footnote {marker} is new but pre-2022")
                reason = PENDING_REASONS.get((section, marker))
                if reason is None:
                    raise ValueError(
                        f"Section {section} footnote {marker} is post-2022 but neither versioned "
                        "nor listed in PENDING_REASONS"
                    )
                rows.append(
                    {
                        "section": section,
                        "marker": marker,
                        "footnote": note["text"],
                        "source_url": self.pages[section].source_url,
                        **reason,
                    }
                )
        return rows


# Post-2022 footnotes in s9/s16/s17 that the held sources cannot date. Each names the node it
# amends and the source that would resolve it.
PENDING_REASONS: dict[tuple[str, str], dict[str, str]] = {
    ("9", "2"): {
        "path": "CGST/s9/ss1",
        "reason": "Footnote names s.114 of the Finance (No. 2) Act, 2024 (dated 16.08.2024) but "
        "gives no date from which the words take effect.",
        "needs": "Text of s.114, Finance (No. 2) Act, 2024 and its commencement notification.",
    },
    ("16", "11"): {
        "path": "CGST/s16/ss5, CGST/s16/ss6",
        "reason": "Footnote names s.118 of the Finance (No. 2) Act, 2024 (dated 16.08.2024) but "
        "gives no date from which sub-sections (5) and (6) take effect.",
        "needs": "Text of s.118, Finance (No. 2) Act, 2024 (states the effective date).",
    },
    ("17", "6"): {
        "path": "CGST/s17/ss5/cl(i)",
        "reason": "Footnote names s.119 of the Finance (No. 2) Act, 2024 (dated 16.08.2024) "
        "with no w.e.f. date and no old text (the old text is in the 2022 PDF).",
        "needs": "Commencement notification for s.119, Finance (No. 2) Act, 2024.",
    },
    ("17", "7"): {
        "path": "CGST/s17/ss5/cl(d)",
        "reason": "Retrospective substitution ('deemed ... with effect from the 1st day of July, "
        "2017') by s.124 of the Finance Act, 2025: the footnote gives no enactment date "
        "(the knowledge date, in_force_since), and ProvisionRecord has no field to mark the "
        "old 'plant or machinery' text as superseded from that date, so both versions would "
        "overlap on every reference date.",
        "needs": "Enactment date of the Finance Act, 2025; a knowledge-end field "
        "(e.g. superseded_since) on ProvisionRecord.",
    },
}


def check_pdf_chains_still_current(pages: dict[str, SectionPage], pdf_records) -> None:
    """The PDF-built core chains, rebuilt from the HTML pages, must come out identical."""
    import ingest.core_provisions as core

    text = "\n".join(pages[number].text() for number in CORE_SECTIONS)
    original = core._order_02_2018
    try:
        # Order 02/2018 provenance is checked against its own PDF in the main build; here
        # only the Act text matters.
        core._order_02_2018 = lambda proviso: {
            "effective_from": next(
                r.effective_from for r in pdf_records if r.path == "CGST/s16/ss4/proviso1"
            ),
            "amended_by": None,
            "source_url": None,
        }
        html_records, _ = build_act_core_provisions(text, source_name="html")
    finally:
        core._order_02_2018 = original
    pdf_by_id = {r.id: r for r in pdf_records}
    if set(pdf_by_id) != {r.id for r in html_records}:
        raise ValueError("CBIC pages give a different set of core Act versions than the PDF")
    for record in html_records:
        expected = pdf_by_id[record.id]
        if (record.text, record.effective_from, record.effective_to) != (
            expected.text,
            expected.effective_from,
            expected.effective_to,
        ):
            raise ValueError(f"{record.id}: CBIC page text or dates differ from the 2022 PDF")


def build_post_pdf(
    pages: dict[str, SectionPage], pdf_text: str
) -> tuple[list[ProvisionRecord], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    builder = _Builder(pages, pdf_text)
    records = builder.build()
    footnotes = {n: list(builder.notes[n].values()) for n in CORE_SECTIONS}
    return records, builder.pending(), footnotes

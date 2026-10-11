"""Read the CBIC per-section HTML pages of the CGST Act (current text with footnotes).

Source: ``data/raw/cgst-act-sections_cbic-taxinfo_<date>/`` written by
``ingest.fetch_act_sections`` (one HTML page per section plus ``index.json`` with the portal
metadata, URL and sha256 of every page).

The PDF of the complete Act on the portal stops at 28-Sep-2022; these pages carry the later
amendments. ``section_text`` turns a page into the same plain-text shape the PDF parser
expects (one line per block, footnote markers as ``N[`` and the footnote list after an
``Enforced`` line where the page has one), so ``ingest.core_provisions`` can read both sources
with the same footnote-chain code.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SNAPSHOT_GLOB = "cgst-act-sections_cbic-taxinfo_*"

_BLOCK_TAGS = {
    "p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol",
    "blockquote",
}  # fmt: skip
_SKIP_TAGS = {"script", "style", "head", "title"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            # Newlines inside HTML source are layout, not structure: only block tags break lines.
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> str:
    """Block-per-line plain text. Inline whitespace is collapsed; NBSP and zero-width chars go."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = text.replace(" ", " ").replace("​", "").replace("﻿", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class SectionPage:
    number: str  # "16", "11A", "Schedule III"
    name: str
    file: Path
    url: str
    sha256: str

    @property
    def source_url(self) -> str:
        return self.file.relative_to(ROOT).as_posix()

    def html(self) -> str:
        return self.file.read_text(encoding="utf-8", errors="strict")

    def text(self) -> str:
        return html_to_text(self.html())


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path:
    candidates = sorted(p for p in raw_dir.glob(SNAPSHOT_GLOB) if (p / "index.json").is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No {SNAPSHOT_GLOB}/index.json under {raw_dir}; "
            "run python -m ingest.fetch_act_sections"
        )
    return candidates[-1]


def snapshot_date(snapshot: Path) -> str:
    """Knowledge date of a snapshot: the date in its folder name (the day it was fetched)."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})$", snapshot.name)
    if match is None:
        raise ValueError(f"Snapshot folder has no date: {snapshot.name}")
    return match.group(1)


def load_pages(snapshot: Path | None = None, *, verify: bool = True) -> dict[str, SectionPage]:
    """Pages keyed by section number ("16", "11A", "Schedule III"), checked against index.json."""
    snapshot = snapshot or latest_snapshot()
    index = json.loads((snapshot / "index.json").read_text(encoding="utf-8"))
    if index.get("failures"):
        raise ValueError(f"{snapshot.name}/index.json lists failed downloads: {index['failures']}")
    pages: dict[str, SectionPage] = {}
    for row in index["sections"]:
        path = snapshot / row["file"]
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"sha256 mismatch for {path} (file changed since download)")
        number = re.sub(r"^Section\s+", "", row["section_no"].strip())
        pages[number] = SectionPage(number, row["section_name"], path, row["url"], row["sha256"])
    return pages


def section_text(page: SectionPage) -> str:
    """Page text in the PDF parser's shape: body, then "Enforced" and the footnote list."""
    return page.text()

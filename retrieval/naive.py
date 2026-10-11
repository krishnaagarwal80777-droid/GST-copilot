"""Naive baseline (B2): fixed 500-token chunks of the raw PDFs, dense retrieval only.

No legal structure, no versions, no dates: every PDF in data/raw is extracted with pypdf (the
plain-text path a naive pipeline would take) and cut into consecutive 500-token windows using
the bge-m3 tokenizer. Files without a text layer contribute nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHUNK_TOKENS = 500


def raw_pdf_texts(raw_dir: Path = RAW) -> list[tuple[str, str]]:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    skip = {entry["filename"] for entry in manifest if entry.get("no_text_layer")}
    texts = []
    for path in sorted(raw_dir.glob("*.pdf")):
        if path.name in skip:
            continue
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        if text.strip():
            texts.append((path.name, text))
    return texts


def naive_rows(tokenizer, raw_dir: Path = RAW) -> list[dict]:
    rows = []
    for filename, text in raw_pdf_texts(raw_dir):
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        for index, start in enumerate(range(0, len(offsets), CHUNK_TOKENS)):
            window = offsets[start : start + CHUNK_TOKENS]
            chunk = text[window[0][0] : window[-1][1]]
            rows.append(
                {
                    "id": f"{filename}#{index}",
                    "record_id": f"{filename}#{index}",
                    "path": filename,
                    "act_id": "",
                    "doc_type": "raw",
                    "node_type": "chunk",
                    "effective_from": "0001-01-01",
                    "effective_to": "9999-12-31",
                    "in_force_since": "0001-01-01",
                    "text": chunk,
                    "servable": "0",
                }
            )
    return rows


_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}

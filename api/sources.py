"""Record id -> public source URL, from data/raw/manifest.json.

Core Act and Rule versions name their source file (``source_url: data/raw/<file>``); rate
schedule records (``CTR/9-2025/schII@v1``) come from the notification file in
retrieval.corpus.RATE_NOTIFICATIONS. Engine rule versions record no source file, so they have
no URL.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from retrieval.corpus import CORE_FILE, RATE_NOTIFICATIONS

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "manifest.json"


@cache
def manifest_urls(path: Path = MANIFEST) -> dict[str, str]:
    """filename -> download URL."""
    if not path.is_file():
        return {}
    return {entry["filename"]: entry["url"] for entry in json.loads(path.read_text("utf-8"))}


@cache
def record_files(core_file: Path = CORE_FILE) -> dict[str, str]:
    """record id -> source filename, for core provision versions."""
    files: dict[str, str] = {}
    if core_file.is_file():
        for line in core_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("source_url"):
                    files[record["id"]] = Path(record["source_url"]).name
    return files


def source_url(record_id: str) -> str | None:
    path = record_id.split("@", 1)[0]
    if path.startswith("CTR/"):
        number = path.split("/")[1].replace("-", "/")
        filename = next((n.filename for n in RATE_NOTIFICATIONS if n.number == number), None)
    else:
        filename = record_files().get(record_id)
    return manifest_urls().get(filename) if filename else None

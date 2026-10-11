from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from gst_copilot.models import ProvisionRecord
from ingest.segment import parse_act_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_FILE = ROOT / "data" / "processed" / "provisions.jsonl"


def build_provision_records(
    raw_text: str,
    *,
    act_id: str = "CGST",
    version: int = 1,
    effective_from: date = date(2017, 7, 1),
    in_force_since: date = date(2017, 7, 1),
) -> list[ProvisionRecord]:
    """Convert a raw legal text fixture into provision records usable by the retrieval layer."""
    records: list[ProvisionRecord] = []
    for node in parse_act_text(raw_text, act_id):
        if node.node_type == "section" and not node.heading:
            continue
        if not node.text and node.node_type not in {"section"}:
            continue
        records.append(
            ProvisionRecord(
                id=f"{node.path}@v{version}",
                doc_type="act",
                act_id=f"{act_id}2017",
                path=node.path,
                node_type=node.node_type,
                heading=node.heading,
                text=node.text,
                context_prefix=node.heading or "",
                effective_from=effective_from,
                effective_to=None,
                in_force_since=in_force_since,
            )
        )
    return records


def write_processed_records(
    records: list[ProvisionRecord],
    output_path: str | Path = DEFAULT_PROCESSED_FILE,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path


def load_processed_records(path: str | Path = DEFAULT_PROCESSED_FILE) -> list[ProvisionRecord]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[ProvisionRecord] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            records.append(ProvisionRecord.model_validate(json.loads(payload)))
    return records


def ingest(raw_file: str | Path, output_path: str | Path) -> list[ProvisionRecord]:
    """Segment a plain-text Act file into records. Both paths are explicit: the shared
    provisions.jsonl is never a default target, so sample text cannot overwrite real records."""
    raw_path = Path(raw_file)
    records = build_provision_records(raw_path.read_text(encoding="utf-8"))
    write_processed_records(records, output_path)
    return records

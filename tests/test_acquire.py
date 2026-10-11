import base64
import hashlib
import json
from datetime import datetime

import httpx
import pytest

from ingest.acquire import (
    InvalidPayloadError,
    acquire_sources,
    repair_raw_files,
    unwrap_pdf_payload,
)


def test_acquire_writes_file_and_sha256_manifest(tmp_path) -> None:
    source_bytes = b"synthetic source fixture"
    source_config = tmp_path / "sources.yaml"
    source_config.write_text(
        "sources:\n  - name: fixture\n    url: https://example.test/fixture.txt\n",
        encoding="utf-8",
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=source_bytes))
    )
    raw_dir = tmp_path / "raw"

    manifest = acquire_sources(source_config, raw_dir, client=client)

    assert (raw_dir / "fixture.txt").read_bytes() == source_bytes
    assert manifest[0]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert manifest[0]["url"] == "https://example.test/fixture.txt"
    datetime.fromisoformat(manifest[0]["downloaded_at"])
    assert json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
    client.close()


def _blank_pdf() -> bytes:
    """A valid one-page PDF with no text (synthetic container, not legal text)."""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=10, height=10)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


BLANK_PDF = _blank_pdf()
RATE_SOURCE = "sources:\n  - name: rate\n    url: https://example.test/r.pdf\n    filename: r.pdf\n"


def _wrapper(content: bytes) -> bytes:
    return json.dumps({"data": base64.b64encode(content).decode(), "fileName": "x.pdf"}).encode()


def _acquire(tmp_path, body: bytes) -> list[dict]:
    source_config = tmp_path / "sources.yaml"
    source_config.write_text(RATE_SOURCE, encoding="utf-8")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    )
    try:
        return acquire_sources(source_config, tmp_path / "raw", client=client)
    finally:
        client.close()


def test_json_base64_wrapper_is_decoded_and_its_hash_kept(tmp_path) -> None:
    wrapped = _wrapper(BLANK_PDF)
    manifest = _acquire(tmp_path, wrapped)

    assert (tmp_path / "raw" / "r.pdf").read_bytes() == BLANK_PDF
    assert manifest[0]["sha256"] == hashlib.sha256(BLANK_PDF).hexdigest()
    assert manifest[0]["wrapper_sha256"] == hashlib.sha256(wrapped).hexdigest()
    assert manifest[0]["no_text_layer"] is True


def test_non_pdf_payload_is_never_saved_as_pdf(tmp_path) -> None:
    with pytest.raises(InvalidPayloadError):
        unwrap_pdf_payload(b'{"error": "not found"}', "r.pdf")
    with pytest.raises(InvalidPayloadError):
        unwrap_pdf_payload(_wrapper(b"<html>"), "r.pdf")

    assert _acquire(tmp_path, b"<html>") == []
    assert not (tmp_path / "raw" / "r.pdf").exists()


def test_repair_unwraps_saved_wrapper_and_records_old_hash(tmp_path) -> None:
    wrapped = _wrapper(BLANK_PDF)
    (tmp_path / "r.pdf").write_bytes(wrapped)
    manifest = [{"filename": "r.pdf", "sha256": hashlib.sha256(wrapped).hexdigest()}]

    (entry,) = repair_raw_files(tmp_path, manifest)

    assert (tmp_path / "r.pdf").read_bytes() == BLANK_PDF
    assert entry["wrapper_sha256"] == hashlib.sha256(wrapped).hexdigest()
    assert entry["sha256"] == hashlib.sha256(BLANK_PDF).hexdigest()


def test_repair_refuses_a_file_that_no_longer_matches_its_manifest(tmp_path) -> None:
    (tmp_path / "r.pdf").write_bytes(_wrapper(BLANK_PDF))
    with pytest.raises(InvalidPayloadError):
        repair_raw_files(tmp_path, [{"filename": "r.pdf", "sha256": "0" * 64}])

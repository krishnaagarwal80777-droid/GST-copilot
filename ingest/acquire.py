from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = ROOT / "data" / "sources.yaml"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_FILE = RAW_DIR / "manifest.json"


def _filename_for(source: dict[str, object], url: str, index: int) -> str:
    configured = source.get("filename")
    filename = str(configured) if configured else Path(unquote(urlparse(url).path)).name
    if not filename:
        filename = f"source-{index}"
    filename = Path(filename).name
    if filename in {".", ".."} or not re.fullmatch(r"[\w.() -]+", filename):
        raise ValueError(f"Unsafe filename for source {index}: {filename!r}")
    return filename


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
RETRIES = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class InvalidPayloadError(ValueError):
    pass


def unwrap_pdf_payload(payload: bytes, filename: str) -> tuple[bytes, bool]:
    """Return (file bytes, was_wrapped) for a downloaded payload.

    The CBIC tax-information portal can answer a PDF URL with JSON
    ``{"data": "<base64 PDF>", "fileName": ...}``. Such a wrapper is decoded; a ``.pdf`` target
    that is not a PDF after unwrapping is rejected so it is never saved under a .pdf name.
    """
    wrapped = False
    if payload.lstrip()[:1] == b"{":
        try:
            document = json.loads(payload)
            data = document.get("data") if isinstance(document, dict) else None
            decoded = base64.b64decode(data, validate=True) if isinstance(data, str) else None
        except (ValueError, binascii.Error):
            decoded = None
        if decoded is not None:
            payload, wrapped = decoded, True
    if filename.lower().endswith(".pdf") and not payload.startswith(b"%PDF-"):
        raise InvalidPayloadError(f"{filename}: response is not a PDF (starts {payload[:8]!r})")
    return payload, wrapped


def has_text_layer(path: Path) -> bool:
    """True when any page of the PDF yields extractable text (False for scanned images)."""
    from pypdf import PdfReader

    return any((page.extract_text() or "").strip() for page in PdfReader(str(path)).pages)


def _ssl_context():
    """Verify TLS against the OS trust store (Windows/macOS) when available.

    Some government sites (e.g. taxinformation.cbic.gov.in) omit an intermediate
    certificate. Browsers and the OS store resolve it; certifi alone cannot.
    Verification stays ON either way.
    """
    try:
        import ssl

        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return True


def _download(http: httpx.Client, url: str, destination: Path) -> tuple[str, str | None]:
    """Download url to destination; return (sha256 of saved file, sha256 of wrapper or None)."""
    temporary = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with http.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            received = temporary.read_bytes()
            content, wrapped = unwrap_pdf_payload(received, destination.name)
            temporary.write_bytes(content)
            temporary.replace(destination)
            wrapper_sha = hashlib.sha256(received).hexdigest() if wrapped else None
            return hashlib.sha256(content).hexdigest(), wrapper_sha
        except InvalidPayloadError:
            temporary.unlink(missing_ok=True)
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            temporary.unlink(missing_ok=True)
            last_error = exc
            print(f"  attempt {attempt}/{RETRIES} failed: {exc!r}")
            if attempt < RETRIES:
                time.sleep(5 * attempt)
    assert last_error is not None
    raise last_error


LOCAL_SUFFIXES = {".pdf", ".html", ".htm"}


def register_local_files(raw_dir: Path, manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add files placed in raw_dir by hand (no URL in sources.yaml) to the manifest.

    Entries already in the manifest are left untouched. Local entries get url=None
    and source="manual" so provenance stays explicit.
    """
    known = {entry["filename"] for entry in manifest}
    added: list[dict[str, str]] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in LOCAL_SUFFIXES or path.name in known:
            continue
        added.append(
            {
                "name": path.stem,
                "filename": path.name,
                "url": None,
                "source": "manual",
                "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "sha256": _sha256_file(path),
            }
        )
    return manifest + added


def unwrap_existing_file(path: Path) -> tuple[str, str | None]:
    """Decode a saved JSON/base64 wrapper in place; return (sha256, wrapper sha256 or None)."""
    received = path.read_bytes()
    content, wrapped = unwrap_pdf_payload(received, path.name)
    if not wrapped:
        return hashlib.sha256(received).hexdigest(), None
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), hashlib.sha256(received).hexdigest()


def _mark_text_layer(path: Path, entry: dict) -> None:
    if path.suffix.lower() == ".pdf" and not has_text_layer(path):
        entry["no_text_layer"] = True
    else:
        entry.pop("no_text_layer", None)


def repair_raw_files(raw_dir: Path, manifest: list[dict]) -> list[dict]:
    """Unwrap saved JSON/base64 payloads and flag PDFs without a text layer, offline.

    The manifest's sha256 becomes the hash of the decoded file; the hash of what was originally
    saved is kept as wrapper_sha256 (it must equal the sha256 recorded at download time).
    """
    for entry in manifest:
        path = raw_dir / entry["filename"]
        if not path.is_file():
            continue
        recorded = entry.get("sha256")
        if recorded and _sha256_file(path) != recorded:
            raise InvalidPayloadError(f"{path.name}: file differs from its manifest hash")
        sha, wrapper_sha = unwrap_existing_file(path)
        if wrapper_sha:
            entry["wrapper_sha256"] = wrapper_sha
        entry["sha256"] = sha
        _mark_text_layer(path, entry)
    return manifest


def write_manifest(raw_dir: Path, manifest: list[dict[str, str]]) -> Path:
    manifest_file = raw_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_file


def acquire_sources(
    sources_file: Path = SOURCES_FILE,
    raw_dir: Path = RAW_DIR,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
) -> list[dict[str, str]]:
    """Download configured sources and write their URL/date/SHA-256 manifest.

    Files already in raw_dir are kept and hashed, not re-downloaded (unless force).
    A failed source is reported and skipped so the rest still download.
    """
    with sources_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    sources = config.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("data/sources.yaml must contain a 'sources' list")

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    failures: list[tuple[str, str]] = []
    owns_client = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        verify=_ssl_context(),
    )
    try:
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict) or not source.get("url"):
                raise ValueError(f"Source {index} must be a mapping with a non-empty 'url'")
            url = str(source["url"])
            filename = _filename_for(source, url, index)
            destination = raw_dir / filename
            wrapper_sha: str | None = None
            if destination.exists() and destination.stat().st_size > 0 and not force:
                print(f"[{index}] exists, hashing: {filename}")
                try:
                    sha, wrapper_sha = unwrap_existing_file(destination)
                except InvalidPayloadError as exc:
                    failures.append((filename, repr(exc)))
                    continue
            else:
                print(f"[{index}] downloading: {filename}")
                try:
                    sha, wrapper_sha = _download(http, url, destination)
                except Exception as exc:  # keep going; report at the end
                    failures.append((filename, repr(exc)))
                    continue
            entry = {
                "name": str(source.get("name", filename)),
                "filename": filename,
                "url": url,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "sha256": sha,
            }
            if wrapper_sha:
                entry["wrapper_sha256"] = wrapper_sha
            _mark_text_layer(destination, entry)
            manifest.append(entry)
    finally:
        if owns_client:
            http.close()

    manifest = register_local_files(raw_dir, manifest)
    write_manifest(raw_dir, manifest)
    if failures:
        print(
            "\nFailed (download these in a browser into data/raw with the same filename, "
            "then re-run):"
        )
        for filename, error in failures:
            print(f"  - {filename}: {error}")
    return manifest


def main() -> None:
    import sys

    if "--unwrap-existing" in sys.argv:
        manifest = repair_raw_files(RAW_DIR, json.loads(MANIFEST_FILE.read_text(encoding="utf-8")))
        write_manifest(RAW_DIR, manifest)
        wrapped = sum(1 for entry in manifest if entry.get("wrapper_sha256"))
        flagged = [entry["filename"] for entry in manifest if entry.get("no_text_layer")]
        print(f"Unwrapped {wrapped} file(s); no text layer: {flagged}")
        return
    if "--local-only" in sys.argv:
        existing = (
            json.loads(MANIFEST_FILE.read_text(encoding="utf-8")) if MANIFEST_FILE.exists() else []
        )
        manifest = register_local_files(RAW_DIR, existing)
        write_manifest(RAW_DIR, manifest)
        print(
            f"Recorded {len(manifest)} source(s) in {MANIFEST_FILE} (local scan only, no downloads)"
        )
        return
    manifest = acquire_sources()
    print(f"Recorded {len(manifest)} source(s) in {MANIFEST_FILE}")


if __name__ == "__main__":
    main()

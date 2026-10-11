"""Download the current CGST Act, section by section, from the CBIC tax-information portal.

The portal's "complete Act" PDF stops at 28-Sep-2022, but the per-section HTML pages are
kept current (footnotes cite Finance Acts up to 2026). This script saves all of them with
provenance so the parser can build post-2022 versions of s16, s17 and the rest.

Usage (from the project root, venv active):
    python -m ingest.fetch_act_sections            # skips files already saved
    python -m ingest.fetch_act_sections --force    # re-download everything

Output:
    data/raw/cgst-act-sections_cbic-taxinfo_<YYYY-MM-DD>/
        chapterN/sectionN_v1.00.html   (raw bytes as served)
        index.json                     (API metadata + url, sha256, bytes per section)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

BASE = "https://taxinformation.cbic.gov.in"
SECTIONS_API = f"{BASE}/api/cbic-act-section-msts/findByActId/1000006"  # CGST Act, 2017
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = httpx.Timeout(60.0, connect=30.0)
RETRIES = 3
RAW_DIR = Path("data/raw")


def _ssl_context():
    try:
        import truststore

        print("TLS: using OS trust store (truststore)")
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        print("TLS: truststore not installed, using certifi bundle")
        return True


def _aia_context(host: str) -> ssl.SSLContext:
    """Repair a server that omits its intermediate certificate.

    Reads the leaf certificate's AIA "CA Issuers" URL, downloads the intermediate, and adds it
    to the normal certifi roots. The full chain is still verified, so this is not verify=False.
    """
    import certifi
    from cryptography import x509
    from cryptography.x509.oid import AuthorityInformationAccessOID, ExtensionOID

    leaf = x509.load_pem_x509_certificate(ssl.get_server_certificate((host, 443)).encode())
    aia = leaf.extensions.get_extension_for_oid(ExtensionOID.AUTHORITY_INFORMATION_ACCESS).value
    urls = [
        d.access_location.value
        for d in aia
        if d.access_method == AuthorityInformationAccessOID.CA_ISSUERS
    ]
    if not urls:
        raise RuntimeError(f"{host}: leaf certificate has no CA Issuers URL")

    ctx = ssl.create_default_context(cafile=certifi.where())
    for url in urls:
        der_or_pem = httpx.get(url, timeout=TIMEOUT, follow_redirects=True).content
        pem = (
            der_or_pem.decode()
            if der_or_pem.startswith(b"-----BEGIN")
            else ssl.DER_cert_to_PEM_cert(der_or_pem)
        )
        ctx.load_verify_locations(cadata=pem)
        print(f"TLS: added missing intermediate from {url}")
    return ctx


def _get(client: httpx.Client, url: str) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = client.get(url)
            r.raise_for_status()
            return r
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise  # retrying won't help; caller repairs the chain
            last = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed after {RETRIES} tries: {url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument(
        "--out", type=Path, default=None, help="output folder (default: dated folder in data/raw)"
    )
    args = ap.parse_args()

    out = args.out or RAW_DIR / f"cgst-act-sections_cbic-taxinfo_{date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html, */*"}
    client = httpx.Client(
        headers=headers, timeout=TIMEOUT, verify=_ssl_context(), follow_redirects=True
    )
    try:
        sections = _get(client, SECTIONS_API).json()
    except httpx.HTTPError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        print("TLS: server sent an incomplete certificate chain; fetching the intermediate via AIA")
        client.close()
        client = httpx.Client(
            headers=headers,
            timeout=TIMEOUT,
            verify=_aia_context(httpx.URL(BASE).host),
            follow_redirects=True,
        )
        sections = _get(client, SECTIONS_API).json()

    try:
        print(f"{len(sections)} sections listed by the portal")

        index, failures = [], []
        for i, s in enumerate(sections, 1):
            rel = s["contentFilePath"].replace(
                "\\", "/"
            )  # tax_repository/gst/acts/2017_CGST_act/active/chapterN/...
            url = f"{BASE}/content/html/{rel}"
            dest = out / "/".join(rel.split("/")[-2:])  # chapterN/sectionN_v1.00.html
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and not args.force:
                body = dest.read_bytes()
            else:
                try:
                    body = _get(client, url).content
                except RuntimeError as e:
                    failures.append(str(e))
                    print(f"  FAIL {s['sectionNo']}: {e}")
                    continue
                dest.write_bytes(body)
                time.sleep(0.2)  # be polite to the portal

            index.append(
                {
                    "section_no": s.get("sectionNo"),
                    "section_name": s.get("sectionName"),
                    "order_id": s.get("orderId"),
                    "chapter_id": (s.get("chapterId") or {}).get("id"),
                    "portal_id": s.get("id"),
                    "version_no": s.get("versionNo"),
                    "portal_created_dt": s.get("createdDt"),
                    "portal_updated_dt": s.get("updatedDt"),
                    "url": url,
                    "file": dest.relative_to(out).as_posix(),
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            if i % 20 == 0:
                print(f"  {i}/{len(sections)}")

    finally:
        client.close()

    meta = {
        "source": "CBIC tax information portal, CGST Act per-section HTML (active versions)",
        "sections_api": SECTIONS_API,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "count": len(index),
        "failures": failures,
        "sections": index,
    }
    (out / "index.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {len(index)} sections to {out} ({len(failures)} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

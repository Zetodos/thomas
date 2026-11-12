# SPDX-License-Identifier: Apache-2.0
"""Explicit setup downloads for CopSahHorner, Vaticanus B* TEI, and Dilley Thomas."""

from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from zetodos_thomas.ingest.dilley_thomas import TT_FILENAME, TT_TAG_URL
from zetodos_thomas.ingest.manifests import load_manifest
from zetodos_thomas.ingest.ntvmr_vaticanus import DOC_ID, teiraw_filename

COPSAHHORNER_ZIP_URL = (
    "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/CopSahHorner.zip"
)

NTVMR_TRANSCRIPT_GET = "https://ntvmr.uni-muenster.de/community/api/transcript/get/"
# Alias that is often more resilient from cloud egress than the primary path.
NTVMR_TRANSCRIPT_GET_VMR = (
    "https://ntvmr.uni-muenster.de/community/vmr/api/transcript/get/"
)

# Expanded Gospel-first TEI pins (teiraw).
# Must match data/manifests/ntvmr-vaticanus-b.json.
# Primary pin (manifest.checksum) is John.1.1-51 (JAK-43); Phase 1 subranges are
# superseded by these chapter / block pins.
NTVMR_TEIRAW_FILES: dict[str, str] = {
    "John.1.1-51": (
        "sha256:e9e36dc3a09394741c4902a2b0927bf642629c086809959c289612acea39307d"
    ),
    "John.3.1-21": (
        "sha256:10b669cf7a9234afee857e88d455222d23867676363e63e5396bb4bdfd02b6b1"
    ),
    "John.8.1-11": (
        "sha256:8cd9c90bcbed2fee6f7a3241abd79540da712d58db44aa689e7b0c93cf2f6a92"
    ),
    "Mark.1.1-45": (
        "sha256:eb68d9535b02cb8a933460a4ebdcd0b3d7ad39d78d9b1a9192f194404cc9aa71"
    ),
    "Mark.4.1-34": (
        "sha256:18323b2afd9cd3a38c05861f41ef1dcdc13a96235016bd7d92a717b60c2d4453"
    ),
    "Matt.5.1-48": (
        "sha256:d2495062b9cb671dc0cdeda987ff0284bbdaa734b360fbfb7aed58537c166fd0"
    ),
    "Matt.6.1-34": (
        "sha256:2971d526e68409a9cec734178b1328e24054c9ae6c799c13b39fec13ce9c296c"
    ),
    "Matt.7.1-29": (
        "sha256:9e7e3b25684cdfdd69a83e9ababd32a37f40615b9a85613ba539f558d2bcbbe2"
    ),
    "Matt.10.26-33": (
        "sha256:3a6613b51693bc8505373c9430edcd90ce20c30192b2bda3dd0cbf37d3802eea"
    ),
    "Matt.13.1-58": (
        "sha256:3463691f554ce2898f3d7a0edd0b7250210ab83d8385a66957746f32f0a933c9"
    ),
    "Luke.6.1-49": (
        "sha256:e80f93e3e845b480d2e190b90339457b067418d416ef83821ca3a49c36ab9baf"
    ),
    "Luke.8.4-15": (
        "sha256:14d22e28f5700f8cfd45e5ab27e3e76d9bc03cd7c74788cc1a59dbc63cdd20c2"
    ),
    "Luke.11.9-13": (
        "sha256:75a3f46672cd664344f2b43608d2bf753aad64ce61e0d9d19747b482fc08f635"
    ),
    "Luke.12.2-3": (
        "sha256:1685af9116faa06c113710936225ebf3b8e246e70ac4f75d5fb159382d41587f"
    ),
}

# Soft courtesy throttle between large NTVMR pulls (see JAK-25 feasibility).
NTVMR_THROTTLE_SECONDS = 1.0
NTVMR_FETCH_RETRIES = 5


class DownloadError(RuntimeError):
    """Setup download or checksum verification failed."""


def sha256_file(path: Path) -> str:
    """Return ``sha256:<hex>`` for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def verify_checksum(path: Path, expected: str) -> None:
    """Raise :class:`DownloadError` if ``path`` does not match ``expected``."""
    actual = sha256_file(path)
    if actual != expected:
        msg = f"checksum mismatch for {path}: expected {expected}, got {actual}"
        raise DownloadError(msg)


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with TLS certificate verification always on.

    Retries transient transport failures only. Certificate / hostname
    verification is never disabled (no ``ssl._create_unverified_context``).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "zetodos-thomas/0.1 (setup-download)"})
    errors: list[Exception] = []
    for attempt in range(NTVMR_FETCH_RETRIES):
        try:
            with urlopen(req, timeout=180) as response:
                data = response.read()
            if len(data) < 64:
                msg = f"download too small from {url}: {len(data)} bytes"
                raise DownloadError(msg)
            dest.write_bytes(data)
            return
        except (URLError, TimeoutError, DownloadError, OSError) as exc:
            errors.append(exc)
            # Do not retry pure certificate/hostname failures as "soft" TLS
            # downgrades — only retry transport-class errors.
            reason = getattr(exc, "reason", None)
            cert_markers = ("CERTIFICATE", "SSL", "certificate", "hostname")
            reason_text = str(reason or exc)
            if any(marker in reason_text for marker in cert_markers):
                # SSL handshake / cert problems: still allow a few retries on
                # flaky IncompleteRead-style EOF, but never disable verify.
                if "CERTIFICATE_VERIFY_FAILED" in reason_text:
                    break
            if attempt + 1 < NTVMR_FETCH_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    msg = f"failed to download {url}: {errors[-1] if errors else 'unknown error'}"
    raise DownloadError(msg) from (errors[-1] if errors else None)


def download_copsahhorner(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download and unpack CopSahHorner; return the SWORD root path."""
    manifest = load_manifest(manifests_dir / "copsahhorner.json")
    if not manifest.checksum.startswith("sha256:"):
        msg = "copsahhorner manifest checksum must be sha256:…"
        raise DownloadError(msg)

    zip_path = downloads_dir / "CopSahHorner.zip"
    sword_root = downloads_dir / "sword"
    if zip_path.is_file() and not force:
        verify_checksum(zip_path, manifest.checksum)
    else:
        _fetch(COPSAHHORNER_ZIP_URL, zip_path)
        verify_checksum(zip_path, manifest.checksum)

    conf = sword_root / "mods.d" / "copsahhorner.conf"
    if not conf.is_file() or force:
        sword_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(sword_root)

    conf = sword_root / "mods.d" / "copsahhorner.conf"
    if not conf.is_file():
        msg = f"CopSahHorner unpack missing {conf}"
        raise DownloadError(msg)
    return sword_root


def ntvmr_teiraw_url(osis_range: str, *, use_vmr_alias: bool = True) -> str:
    """Build the NTVMR teiraw URL for an OSIS indexContent range."""
    query = urlencode(
        {
            "docID": DOC_ID,
            "indexContent": osis_range,
            "format": "teiraw",
            "briefTEIHeader": "true",
            "joinParts": "true",
            "includeEmpty": "true",
        }
    )
    base = NTVMR_TRANSCRIPT_GET_VMR if use_vmr_alias else NTVMR_TRANSCRIPT_GET
    return f"{base}?{query}"


def _fetch_ntvmr_teiraw(osis_range: str, dest: Path) -> None:
    """Fetch one teiraw pin with retries across primary and vmr API paths."""
    last_error: Exception | None = None
    for attempt in range(NTVMR_FETCH_RETRIES):
        for use_vmr in (True, False):
            url = ntvmr_teiraw_url(osis_range, use_vmr_alias=use_vmr)
            try:
                _fetch(url, dest)
                return
            except DownloadError as exc:
                last_error = exc
        time.sleep(1.5 * (attempt + 1))
    msg = f"NTVMR teiraw fetch failed for {osis_range}: {last_error}"
    raise DownloadError(msg) from last_error


def download_ntvmr_vaticanus(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
    throttle_seconds: float = NTVMR_THROTTLE_SECONDS,
) -> Path:
    """Download pinned Vaticanus B* teiraw passages; return the TEI directory.

    Does **not** download manuscript images.
    """
    manifest = load_manifest(manifests_dir / "ntvmr-vaticanus-b.json")
    primary = NTVMR_TEIRAW_FILES["John.1.1-51"]
    if primary != manifest.checksum:
        msg = (
            "NTVMR_TEIRAW_FILES John.1.1-51 pin does not match "
            f"ntvmr-vaticanus-b manifest checksum ({manifest.checksum})"
        )
        raise DownloadError(msg)

    tei_dir = downloads_dir / "ntvmr-vaticanus"
    tei_dir.mkdir(parents=True, exist_ok=True)

    ranges = list(NTVMR_TEIRAW_FILES.items())
    for index, (osis_range, expected) in enumerate(ranges):
        dest = tei_dir / teiraw_filename(osis_range)
        if dest.is_file() and not force:
            verify_checksum(dest, expected)
        else:
            if index > 0 and throttle_seconds > 0:
                time.sleep(throttle_seconds)
            _fetch_ntvmr_teiraw(osis_range, dest)
            verify_checksum(dest, expected)
    return tei_dir


def download_dilley_thomas(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download pinned Dilley SCRIPTORIUM Thomas ``*.tt``; return the file path."""
    manifest = load_manifest(manifests_dir / "dilley-thomas.json")
    if not manifest.checksum.startswith("sha256:"):
        msg = "dilley-thomas manifest checksum must be sha256:…"
        raise DownloadError(msg)

    dest_dir = downloads_dir / "dilley-thomas"
    dest = dest_dir / TT_FILENAME
    if dest.is_file() and not force:
        verify_checksum(dest, manifest.checksum)
    else:
        _fetch(TT_TAG_URL, dest)
        verify_checksum(dest, manifest.checksum)
    return dest


def download_pilot_sources(
    repo_root: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Download CopSahHorner, Vaticanus TEI, and Dilley Thomas under downloads."""
    manifests_dir = repo_root / "data" / "manifests"
    downloads_dir = repo_root / "data" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    sword_root = download_copsahhorner(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    tei_dir = download_ntvmr_vaticanus(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    dilley_tt = download_dilley_thomas(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    return sword_root, tei_dir, dilley_tt

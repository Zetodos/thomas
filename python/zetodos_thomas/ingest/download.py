# SPDX-License-Identifier: Apache-2.0
"""Explicit setup downloads for CopSahHorner and NTVMR Vaticanus B* TEI."""

from __future__ import annotations

import hashlib
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from zetodos_thomas.ingest.manifests import load_manifest
from zetodos_thomas.ingest.ntvmr_vaticanus import DOC_ID, teiraw_filename

COPSAHHORNER_ZIP_URL = (
    "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/CopSahHorner.zip"
)

NTVMR_TRANSCRIPT_GET = "https://ntvmr.uni-muenster.de/community/api/transcript/get/"

# Pilot passage TEI pins (teiraw). Must match data/manifests/ntvmr-vaticanus-b.json.
NTVMR_TEIRAW_FILES: dict[str, str] = {
    "John.1.1-18": (
        "sha256:2153735c61cfec12877dfbda677b3c2269c323fccb9d67c6f70f45a742a725d9"
    ),
    "Mark.1.1-8": (
        "sha256:6b863d7ee3f76f9ca2d6f79adbe69a153555109fb4dcc88105df6c85fa10445b"
    ),
    "John.8.1-11": (
        "sha256:8cd9c90bcbed2fee6f7a3241abd79540da712d58db44aa689e7b0c93cf2f6a92"
    ),
}

# Soft courtesy throttle between large NTVMR pulls (see JAK-25 feasibility).
NTVMR_THROTTLE_SECONDS = 1.0


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
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as response:
            data = response.read()
    except URLError as exc:
        msg = f"failed to download {url}: {exc}"
        raise DownloadError(msg) from exc
    dest.write_bytes(data)


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


def ntvmr_teiraw_url(osis_range: str) -> str:
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
    return f"{NTVMR_TRANSCRIPT_GET}?{query}"


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
    primary = NTVMR_TEIRAW_FILES["John.1.1-18"]
    if primary != manifest.checksum:
        msg = (
            "NTVMR_TEIRAW_FILES John.1.1-18 pin does not match "
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
            _fetch(ntvmr_teiraw_url(osis_range), dest)
            verify_checksum(dest, expected)
    return tei_dir


def download_pilot_sources(
    repo_root: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download CopSahHorner + NTVMR Vaticanus TEI under ``data/downloads``."""
    manifests_dir = repo_root / "data" / "manifests"
    downloads_dir = repo_root / "data" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    sword_root = download_copsahhorner(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    tei_dir = download_ntvmr_vaticanus(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    return sword_root, tei_dir

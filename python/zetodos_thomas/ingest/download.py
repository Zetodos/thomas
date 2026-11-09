# SPDX-License-Identifier: Apache-2.0
"""Explicit setup downloads for CopSahHorner and Westcott-Hort textonly files."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from zetodos_thomas.ingest.manifests import load_manifest

COPSAHHORNER_ZIP_URL = (
    "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/CopSahHorner.zip"
)

# Pins for Robinson textonly books used by the pilot (must match manifests).
WH_TEXTONLY_FILES: dict[str, str] = {
    "JOH.WH": (
        "sha256:cf8f607b8f7b4c1bc471e211bb51ee86750918485be5a4319b3d8fcda61336d2"
    ),
    "MR.WH": (
        "sha256:8cbc2eeb9d5f4dbadbcd44694ba5f82dc94bc5222b0a6a6e95c9e0b93529f9c2"
    ),
}


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


def download_westcott_hort(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download pinned Robinson WH textonly books; return the textonly dir."""
    manifest = load_manifest(manifests_dir / "westcott-hort.json")
    if WH_TEXTONLY_FILES["JOH.WH"] != manifest.checksum:
        msg = (
            "WH_TEXTONLY_FILES JOH.WH pin does not match westcott-hort "
            f"manifest checksum ({manifest.checksum})"
        )
        raise DownloadError(msg)

    commit = manifest.version
    textonly_dir = downloads_dir / "westcott-hort" / "textonly"
    textonly_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected in WH_TEXTONLY_FILES.items():
        dest = textonly_dir / filename
        if dest.is_file() and not force:
            verify_checksum(dest, expected)
            continue
        url = (
            "https://raw.githubusercontent.com/byztxt/greektext-westcott-hort/"
            f"{commit}/textonly/{filename}"
        )
        _fetch(url, dest)
        verify_checksum(dest, expected)
    return textonly_dir


def download_pilot_sources(
    repo_root: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download CopSahHorner + WH textonly under ``repo_root/data/downloads``."""
    manifests_dir = repo_root / "data" / "manifests"
    downloads_dir = repo_root / "data" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    sword_root = download_copsahhorner(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    wh_dir = download_westcott_hort(
        downloads_dir, manifests_dir=manifests_dir, force=force
    )
    return sword_root, wh_dir

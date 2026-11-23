# SPDX-License-Identifier: Apache-2.0
"""Explicit setup downloads for all pinned, licensed corpora and lexicons."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from zetodos_thomas.ingest.dilley_thomas import TT_FILENAME, TT_TAG_URL
from zetodos_thomas.ingest.manifests import load_manifest

COPSAHHORNER_ZIP_URL = (
    "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/CopSahHorner.zip"
)
CCL_TEI_URL = (
    "https://raw.githubusercontent.com/KELLIA/dictionary/master/xml/"
    "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
)
CCL_TEI_FILENAME = "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"

SAHIDIC_OT_ZIP_URL = (
    "https://raw.githubusercontent.com/CopticScriptorium/corpora/"
    "v6.2.0/sahidic.ot/sahidic.ot_TT.zip"
)
SAHIDIC_OT_ZIP_FILENAME = "sahidic.ot_TT.zip"

MORPHGNT_GIT_URL = "https://github.com/morphgnt/sblgnt.git"
MORPHGNT_COMMIT_SHA = "aaed91e57c8e4a8dc9a2383e129ca5e75fe6393d"

SWETE_GIT_URL = "https://github.com/nathans/lxx-swete.git"
SWETE_COMMIT_SHA = "26bad3eb42bba98471d154c954e36a6f30a0279d"

FETCH_RETRIES = 5


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
    """Download ``url`` to ``dest`` with TLS certificate verification always on."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "zetodos-thomas/0.1 (setup-download)"})
    errors: list[Exception] = []
    for attempt in range(FETCH_RETRIES):
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
            reason = getattr(exc, "reason", None)
            reason_text = str(reason or exc)
            if "CERTIFICATE_VERIFY_FAILED" in reason_text:
                break
            if attempt + 1 < FETCH_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    msg = f"failed to download {url}: {errors[-1] if errors else 'unknown error'}"
    raise DownloadError(msg) from (errors[-1] if errors else None)


def _clone_pinned_repo(
    url: str,
    sha: str,
    dest: Path,
    *,
    force: bool = False,
) -> Path:
    """Clone an exact pinned commit SHA into ``dest`` cleanly."""
    if dest.is_dir() and not force:
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=dest,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if head == sha:
                return dest
        except (subprocess.SubprocessError, OSError):
            pass
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
        subprocess.run(["git", "remote", "add", "origin", url], cwd=dest, check=True)
        subprocess.run(
            ["git", "fetch", "-q", "--depth", "1", "origin", sha],
            cwd=dest,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "-q", "--force", "FETCH_HEAD"],
            cwd=dest,
            check=True,
        )
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dest, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError) as exc:
        msg = f"git clone failed for {url} @ {sha}: {exc}"
        raise DownloadError(msg) from exc
    if head != sha:
        msg = f"git pin mismatch for {url}: expected {sha}, got {head}"
        raise DownloadError(msg)
    return dest


def download_copsahhorner(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download and unpack CopSahHorner; return the SWORD root path."""
    manifest = load_manifest(manifests_dir / "copsahhorner.json")
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
    if not conf.is_file():
        msg = f"CopSahHorner unpack missing {conf}"
        raise DownloadError(msg)
    return sword_root


def download_dilley_thomas(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download pinned Dilley SCRIPTORIUM Thomas ``*.tt``."""
    manifest = load_manifest(manifests_dir / "dilley-thomas.json")
    dest = downloads_dir / "dilley-thomas" / TT_FILENAME
    if dest.is_file() and not force:
        verify_checksum(dest, manifest.checksum)
    else:
        _fetch(TT_TAG_URL, dest)
        verify_checksum(dest, manifest.checksum)
    return dest


def download_ccl_tei(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download pinned Comprehensive Coptic Lexicon v1.2 TEI."""
    manifest = load_manifest(manifests_dir / "ccl-ddglc-tei.json")
    dest = downloads_dir / "ccl-ddglc" / CCL_TEI_FILENAME
    if dest.is_file() and not force:
        verify_checksum(dest, manifest.checksum)
    else:
        _fetch(CCL_TEI_URL, dest)
        verify_checksum(dest, manifest.checksum)
    return dest


def download_sahidic_ot(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Download pinned SCRIPTORIUM sahidic.ot_TT.zip."""
    manifest = load_manifest(manifests_dir / "scriptorium-sahidic-ot.json")
    dest = downloads_dir / "sahidic-ot" / SAHIDIC_OT_ZIP_FILENAME
    if dest.is_file() and not force:
        verify_checksum(dest, manifest.checksum)
    else:
        _fetch(SAHIDIC_OT_ZIP_URL, dest)
        verify_checksum(dest, manifest.checksum)
    return dest


def download_morphgnt(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Clone pinned morphgnt/sblgnt and verify checksum of Matthew."""
    manifest = load_manifest(manifests_dir / "morphgnt-sblgnt.json")
    dest = downloads_dir / "morphgnt-sblgnt"
    _clone_pinned_repo(MORPHGNT_GIT_URL, MORPHGNT_COMMIT_SHA, dest, force=force)
    verify_checksum(dest / "61-Mt-morphgnt.txt", manifest.checksum)
    return dest


def download_swete_lxx(
    downloads_dir: Path,
    *,
    manifests_dir: Path,
    force: bool = False,
) -> Path:
    """Clone pinned nathans/lxx-swete and verify checksum of Genesis."""
    manifest = load_manifest(manifests_dir / "swete-lxx.json")
    dest = downloads_dir / "lxx-swete"
    _clone_pinned_repo(SWETE_GIT_URL, SWETE_COMMIT_SHA, dest, force=force)
    verify_checksum(dest / "data" / "01.Genesis.txt", manifest.checksum)
    return dest


def download_all_sources(
    repo_root: Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Download and checksum-verify all setup corpora and lexicons."""
    manifests_dir = repo_root / "data" / "manifests"
    downloads_dir = repo_root / "data" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return {
        "copsahhorner": download_copsahhorner(
            downloads_dir, manifests_dir=manifests_dir, force=force
        ),
        "dilley_thomas": download_dilley_thomas(
            downloads_dir, manifests_dir=manifests_dir, force=force
        ),
        "ccl_tei": download_ccl_tei(
            downloads_dir, manifests_dir=manifests_dir, force=force
        ),
        "sahidic_ot": download_sahidic_ot(
            downloads_dir, manifests_dir=manifests_dir, force=force
        ),
        "morphgnt": download_morphgnt(
            downloads_dir, manifests_dir=manifests_dir, force=force
        ),
        "swete_lxx": download_swete_lxx(
            downloads_dir, manifests_dir=manifests_dir, force=force
        ),
    }

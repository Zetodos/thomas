# SPDX-License-Identifier: Apache-2.0
"""Verify local download inputs against pinned manifests before rebuild."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from zetodos_thomas.ingest.download import (
    NTVMR_TEIRAW_FILES,
    DownloadError,
    sha256_file,
    verify_checksum,
)
from zetodos_thomas.ingest.manifests import load_manifest
from zetodos_thomas.ingest.ntvmr_vaticanus import teiraw_filename


def _verify_sword_matches_zip(sword_root: Path, zip_path: Path) -> None:
    """Ensure every file inside the zip matches the unpacked SWORD tree."""
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            dest = sword_root / info.filename
            if not dest.is_file():
                msg = f"SWORD tree missing {dest} from {zip_path.name}"
                raise DownloadError(msg)
            expected = f"sha256:{hashlib.sha256(archive.read(info)).hexdigest()}"
            actual = sha256_file(dest)
            if actual != expected:
                msg = (
                    f"SWORD file mismatch for {info.filename}: "
                    f"zip has {expected}, disk has {actual}"
                )
                raise DownloadError(msg)


def verify_pilot_rebuild_inputs(
    *,
    manifests_dir: Path,
    sword_root: Path,
    vaticanus_tei_dir: Path,
    downloads_dir: Path,
) -> None:
    """Reject rebuild when Horner or Vaticanus inputs diverge from pins."""
    copsah = load_manifest(manifests_dir / "copsahhorner.json")
    vaticanus = load_manifest(manifests_dir / "ntvmr-vaticanus-b.json")

    if NTVMR_TEIRAW_FILES["John.1.1-18"] != vaticanus.checksum:
        msg = "internal NTVMR John.1.1-18 pin disagrees with manifest"
        raise DownloadError(msg)

    for osis_range, expected in NTVMR_TEIRAW_FILES.items():
        path = vaticanus_tei_dir / teiraw_filename(osis_range)
        if not path.is_file():
            msg = f"missing Vaticanus teiraw required for rebuild: {path}"
            raise DownloadError(msg)
        verify_checksum(path, expected)

    zip_path = downloads_dir / "CopSahHorner.zip"
    if not zip_path.is_file():
        msg = (
            f"missing {zip_path}; run `thomas sources download` so the "
            "checked archive can verify the unpacked SWORD tree"
        )
        raise DownloadError(msg)
    verify_checksum(zip_path, copsah.checksum)

    conf = sword_root / "mods.d" / "copsahhorner.conf"
    if not conf.is_file():
        msg = f"missing CopSahHorner module conf at {conf}"
        raise DownloadError(msg)

    conf_text = conf.read_text(encoding="utf-8", errors="replace")
    if f"Version={copsah.version}" not in conf_text:
        msg = (
            f"copsahhorner.conf Version does not match manifest "
            f"version {copsah.version}"
        )
        raise DownloadError(msg)
    if copsah.sword_version_date and (
        f"SwordVersionDate={copsah.sword_version_date}" not in conf_text
    ):
        msg = (
            "copsahhorner.conf SwordVersionDate does not match manifest "
            f"{copsah.sword_version_date}"
        )
        raise DownloadError(msg)

    _verify_sword_matches_zip(sword_root, zip_path)

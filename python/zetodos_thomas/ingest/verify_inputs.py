# SPDX-License-Identifier: Apache-2.0
"""Verify local download and fragment inputs against pinned manifests."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from zetodos_thomas.ingest.dclp_fragments import DCLP_THOMAS_FRAGMENTS
from zetodos_thomas.ingest.download import (
    CCL_TEI_FILENAME,
    SAHIDIC_OT_ZIP_FILENAME,
    DownloadError,
    sha256_file,
    verify_checksum,
)
from zetodos_thomas.ingest.manifests import load_manifest


def _verify_sword_matches_zip(sword_root: Path, zip_path: Path) -> None:
    """Ensure every file inside CopSahHorner.zip matches the unpacked SWORD tree."""
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


def verify_downloaded_sources(
    *,
    manifests_dir: Path,
    downloads_dir: Path,
) -> None:
    """Verify checksums of all setup-downloaded inputs before building corpora."""
    copsah = load_manifest(manifests_dir / "copsahhorner.json")
    zip_path = downloads_dir / "CopSahHorner.zip"
    if not zip_path.is_file():
        msg = f"missing {zip_path}; run `thomas setup` first"
        raise DownloadError(msg)
    verify_checksum(zip_path, copsah.checksum)
    sword_root = downloads_dir / "sword"
    conf = sword_root / "mods.d" / "copsahhorner.conf"
    if not conf.is_file():
        msg = f"missing CopSahHorner module conf at {conf}"
        raise DownloadError(msg)
    _verify_sword_matches_zip(sword_root, zip_path)

    morphgnt = load_manifest(manifests_dir / "morphgnt-sblgnt.json")
    mt_path = downloads_dir / "morphgnt-sblgnt" / "61-Mt-morphgnt.txt"
    if not mt_path.is_file():
        msg = f"missing MorphGNT Matthew at {mt_path}; run `thomas setup` first"
        raise DownloadError(msg)
    verify_checksum(mt_path, morphgnt.checksum)

    ot_manifest = load_manifest(manifests_dir / "scriptorium-sahidic-ot.json")
    ot_zip = downloads_dir / "sahidic-ot" / SAHIDIC_OT_ZIP_FILENAME
    if not ot_zip.is_file():
        msg = f"missing Sahidic OT zip at {ot_zip}; run `thomas setup` first"
        raise DownloadError(msg)
    verify_checksum(ot_zip, ot_manifest.checksum)

    swete_manifest = load_manifest(manifests_dir / "swete-lxx.json")
    swete_gen = downloads_dir / "lxx-swete" / "data" / "01.Genesis.txt"
    if not swete_gen.is_file():
        msg = f"missing Swete LXX Genesis at {swete_gen}; run `thomas setup` first"
        raise DownloadError(msg)
    verify_checksum(swete_gen, swete_manifest.checksum)

    ccl_manifest = load_manifest(manifests_dir / "ccl-ddglc-tei.json")
    ccl_path = downloads_dir / "ccl-ddglc" / CCL_TEI_FILENAME
    if not ccl_path.is_file():
        msg = f"missing CCL TEI at {ccl_path}; run `thomas setup` first"
        raise DownloadError(msg)
    verify_checksum(ccl_path, ccl_manifest.checksum)


def verify_fragment_rebuild_inputs(
    *,
    manifests_dir: Path,
    tei_dir: Path,
) -> None:
    """Reject fragment rebuild when TEI bytes diverge from manifest checksums."""
    for meta in DCLP_THOMAS_FRAGMENTS:
        source_id = meta["source_id"]
        filename = meta["filename"]
        manifest_path = manifests_dir / f"{source_id}.json"
        if not manifest_path.is_file():
            msg = f"missing fragment manifest required for rebuild: {manifest_path}"
            raise DownloadError(msg)
        manifest = load_manifest(manifest_path)
        if not manifest.checksum:
            msg = f"{source_id}: manifest checksum is empty"
            raise DownloadError(msg)
        path = tei_dir / filename
        if not path.is_file():
            msg = f"missing DCLP TEI required for rebuild: {path}"
            raise DownloadError(msg)
        verify_checksum(path, manifest.checksum)

# SPDX-License-Identifier: Apache-2.0
"""Verify local download and fragment inputs against pinned manifests."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from zetodos_thomas.ingest.dclp_fragments import DCLP_THOMAS_FRAGMENTS
from zetodos_thomas.ingest.download import (
    APOPHTHEGMATA_PIN_REL,
    APOSTOLIC_FATHERS_PIN_REL,
    CCL_TEI_FILENAME,
    FIRST1K_PINNED_FILES,
    SAHIDIC_OT_ZIP_FILENAME,
    STRONGS_XML_FILENAME,
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

    # Optional/v3 pinned sources: verify when manifest exists
    strongs_mf = manifests_dir / "strongs-greek.json"
    if strongs_mf.is_file():
        strongs_m = load_manifest(strongs_mf)
        strongs_p = downloads_dir / "strongs-greek" / STRONGS_XML_FILENAME
        if not strongs_p.is_file():
            msg = f"missing Strong's Greek XML at {strongs_p}; run `thomas setup` first"
            raise DownloadError(msg)
        verify_checksum(strongs_p, strongs_m.checksum)

    af_mf = manifests_dir / "apostolic-fathers.json"
    if af_mf.is_file():
        af_m = load_manifest(af_mf)
        af_p = downloads_dir / "apostolic-fathers" / APOSTOLIC_FATHERS_PIN_REL
        if not af_p.is_file():
            msg = f"missing Apostolic Fathers at {af_p}; run `thomas setup` first"
            raise DownloadError(msg)
        verify_checksum(af_p, af_m.checksum)

    f1k_mf = manifests_dir / "first1k-koine-prose.json"
    if f1k_mf.is_file():
        f1k_m = load_manifest(f1k_mf)
        f1k_dir = downloads_dir / "first1k-koine-prose"
        for rel_path, expected_sha in FIRST1K_PINNED_FILES:
            fpath = f1k_dir / Path(rel_path).name
            if not fpath.is_file():
                msg = f"missing First1KGreek file at {fpath}; run `thomas setup` first"
                raise DownloadError(msg)
            verify_checksum(fpath, expected_sha)
        verify_checksum(f1k_dir / "tlg0018.tlg001.1st1K-grc1.xml", f1k_m.checksum)

    ap_mf = manifests_dir / "scriptorium-apophthegmata.json"
    if ap_mf.is_file():
        ap_m = load_manifest(ap_mf)
        ap_p = downloads_dir / "apophthegmata-patrum" / APOPHTHEGMATA_PIN_REL
        if not ap_p.is_file():
            msg = f"missing Apophthegmata Patrum at {ap_p}; run `thomas setup` first"
            raise DownloadError(msg)
        verify_checksum(ap_p, ap_m.checksum)


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

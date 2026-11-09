# SPDX-License-Identifier: Apache-2.0
"""Tests for source manifests, Vaticanus B* TEI, and pilot import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.bundled import bundled_manifests_dir, bundled_pilot_path
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.download import (
    NTVMR_TEIRAW_FILES,
    DownloadError,
    verify_checksum,
)
from zetodos_thomas.ingest.manifests import (
    load_manifest,
    validate_default_manifests,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.normalize import normalize_coptic, normalize_greek
from zetodos_thomas.ingest.ntvmr_vaticanus import SOURCE_ID, parse_teiraw
from zetodos_thomas.ingest.paths import find_repo_root, resolve_manifests_dir
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.ingest.verify_inputs import verify_pilot_rebuild_inputs
from zetodos_thomas.schemas.source import (
    GATED_DEFAULT_SOURCE_IDS,
    AvailabilityMode,
    SourceManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
FIXTURES = Path(__file__).parent / "fixtures"
NTVMR_FIXTURES = FIXTURES / "ntvmr"


def test_parse_vaticanus_john_1_firsthand(tmp_path: Path) -> None:
    src = NTVMR_FIXTURES / "John_1_1-18.teiraw.xml"
    units = parse_teiraw(src)
    by_id = {u.unit_id: u for u in units}
    assert by_id["john.1.1"].original_text.startswith("εν αρχη ην ο λογος")
    # B*: firsthand omits corrector1 supply των ανθρωπων
    assert by_id["john.1.4"].original_text == "εν αυτω ζωη ην και η ζωη ην το φως"
    assert "ανθρωπων" not in by_id["john.1.4"].original_text
    assert "[η]" in by_id["john.1.15"].original_text


def test_parse_vaticanus_empty_pa() -> None:
    units = parse_teiraw(NTVMR_FIXTURES / "John_8_1-11.teiraw.xml")
    assert len(units) == 11
    assert all(not u.original_text for u in units)


def test_normalize_coptic_strips_overlines() -> None:
    original = "ϩⲛ̅ⲧⲉϩⲟⲩⲉ͡ⲓⲧⲉ ⲛ̅ϭⲓⲡϣⲁϫⲉ"
    normalized = normalize_coptic(original)
    assert "̅" not in normalized
    assert "͡" not in normalized
    assert "ϩⲛⲧⲉϩⲟⲩⲉⲓⲧⲉ" in normalized
    assert normalize_greek("  εν   αρχη  ") == "εν αρχη"


def test_committed_manifests_validate() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    ids = {m.id for m in manifests}
    assert ids == {"copsahhorner", "ntvmr-vaticanus-b"}
    assert ids.isdisjoint(GATED_DEFAULT_SOURCE_IDS)
    horner = load_manifest(MANIFESTS_DIR / "copsahhorner.json")
    assert horner.mode is AvailabilityMode.SETUP_DOWNLOAD
    assert horner.checksum.startswith("sha256:")
    assert horner.counterpart_source_id == "ntvmr-vaticanus-b"
    vat = load_manifest(MANIFESTS_DIR / "ntvmr-vaticanus-b.json")
    assert vat.counterpart_source_id == "copsahhorner"
    assert vat.spdx_or_custom == "CC-BY-4.0"
    assert "westcott" not in vat.id


def test_gated_source_rejected_in_default_set() -> None:
    good = SourceManifest(
        id="copsahhorner",
        title="t",
        version="1.5",
        url="https://example.invalid/horner",
        spdx_or_custom="custom:Public-Domain",
        attribution="Horner",
        mode=AvailabilityMode.SETUP_DOWNLOAD,
        checksum="sha256:" + ("a" * 64),
    )
    bad = SourceManifest(
        id="sahidica",
        title="Wells",
        version="1",
        url="https://example.invalid/sahidica",
        spdx_or_custom="custom:Wells",
        attribution="Wells",
        mode=AvailabilityMode.MANUAL_IMPORT,
        checksum="sha256:" + ("b" * 64),
        gate="JAK-14",
    )
    problems = validate_default_manifests([good, bad])
    assert any("sahidica" in p for p in problems)


def test_empty_checksum_rejected() -> None:
    problems = validate_default_manifests(
        [
            SourceManifest(
                id="ntvmr-vaticanus-b",
                title="Vaticanus",
                version="1",
                url="https://example.invalid/ntvmr",
                spdx_or_custom="CC-BY-4.0",
                attribution="INTF",
                mode=AvailabilityMode.SETUP_DOWNLOAD,
                checksum="",
            )
        ]
    )
    assert any("checksum" in p for p in problems)


def test_pilot_corpus_committed() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    assert corpus.greek_source_id == SOURCE_ID
    assert corpus.sahidic_source_id == "copsahhorner"
    assert len(corpus.units) == 37
    john_1_1 = next(u for u in corpus.units if u.unit_id == "john.1.1")
    assert "λογος" in john_1_1.greek.original_text
    assert "ⲡϣⲁϫⲉ" in john_1_1.sahidic.original_text
    john_1_4 = next(u for u in corpus.units if u.unit_id == "john.1.4")
    assert "ανθρωπων" not in john_1_4.greek.original_text
    empty = [u for u in corpus.units if not u.sahidic.original_text]
    assert [u.unit_id for u in empty] == [f"john.8.{v}" for v in range(1, 12)]
    assert all(not u.greek.original_text for u in empty)
    assert all(u.notes for u in empty)


def test_cli_sources_validate() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "copsahhorner" in result.stdout
    assert "ntvmr-vaticanus-b" in result.stdout
    assert "Sahidica" in result.stdout
    assert "westcott" not in result.stdout.lower()


def test_cli_help_lists_sources() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sources" in result.stdout


def test_cli_sources_download_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "download", "--help"])
    assert result.exit_code == 0
    assert "Vaticanus" in result.stdout or "NTVMR" in result.stdout


def test_find_repo_root_from_tests() -> None:
    assert find_repo_root(Path(__file__)) == REPO_ROOT


def test_ntvmr_download_pins_match_manifest() -> None:
    vat = load_manifest(MANIFESTS_DIR / "ntvmr-vaticanus-b.json")
    assert NTVMR_TEIRAW_FILES["John.1.1-18"] == vat.checksum
    assert "Mark.1.1-8" in NTVMR_TEIRAW_FILES


def test_verify_checksum(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    verify_checksum(
        path,
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )


def test_fixture_manifest_roundtrip() -> None:
    path = FIXTURES / "sample_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = SourceManifest.model_validate(data)
    assert manifest.id == "copsahhorner"


def test_bundled_resources_match_checkout() -> None:
    for name in ("copsahhorner.json", "ntvmr-vaticanus-b.json"):
        checkout = (MANIFESTS_DIR / name).read_text(encoding="utf-8")
        bundled = (bundled_manifests_dir() / name).read_text(encoding="utf-8")
        assert checkout == bundled
    assert PILOT_PATH.read_text(encoding="utf-8") == bundled_pilot_path().read_text(
        encoding="utf-8"
    )


def test_resolve_manifests_prefers_checkout() -> None:
    assert resolve_manifests_dir(REPO_ROOT) == MANIFESTS_DIR


def test_verify_rebuild_rejects_bad_inputs(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in ("copsahhorner.json", "ntvmr-vaticanus-b.json"):
        manifests.joinpath(name).write_text(
            (MANIFESTS_DIR / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    tei_dir = tmp_path / "tei"
    tei_dir.mkdir()
    for osis in NTVMR_TEIRAW_FILES:
        (tei_dir / f"{osis.replace('.', '_')}.teiraw.xml").write_bytes(b"bad")
    sword = tmp_path / "sword"
    sword.mkdir()
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    with pytest.raises(DownloadError):
        verify_pilot_rebuild_inputs(
            manifests_dir=manifests,
            sword_root=sword,
            vaticanus_tei_dir=tei_dir,
            downloads_dir=downloads,
        )

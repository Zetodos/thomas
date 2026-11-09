# SPDX-License-Identifier: Apache-2.0
"""Tests for source manifests, WH/Coptic normalize, and pilot import."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.download import WH_TEXTONLY_FILES, verify_checksum
from zetodos_thomas.ingest.manifests import (
    load_manifest,
    validate_default_manifests,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.normalize import normalize_coptic, normalize_greek
from zetodos_thomas.ingest.paths import find_repo_root
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.ingest.westcott_hort import beta_to_unicode, parse_wh_textonly
from zetodos_thomas.schemas.source import (
    GATED_DEFAULT_SOURCE_IDS,
    AvailabilityMode,
    SourceManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-wh-nt-pilot.json"
FIXTURES = Path(__file__).parent / "fixtures"


def test_beta_to_unicode_john_1_1() -> None:
    beta = "en arch hn o logov kai o logov hn prov ton yeon kai yeov hn o logov"
    assert beta_to_unicode(beta) == (
        "εν αρχη ην ο λογος και ο λογος ην προς τον θεον και θεος ην ο λογος"
    )


def test_parse_wh_textonly_fixture(tmp_path: Path) -> None:
    sample = tmp_path / "JOH.WH"
    sample.write_text(
        "     1:1 en arch hn o logov\n     1:2 outov hn en arch prov ton yeon\n",
        encoding="utf-8",
    )
    units = parse_wh_textonly(sample, book="john")
    assert len(units) == 2
    assert units[0].unit_id == "john.1.1"
    assert "λογος" in units[0].original_text
    assert units[0].normalized_text == units[0].original_text


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
    assert ids == {"copsahhorner", "westcott-hort"}
    assert ids.isdisjoint(GATED_DEFAULT_SOURCE_IDS)
    horner = load_manifest(MANIFESTS_DIR / "copsahhorner.json")
    assert horner.mode is AvailabilityMode.SETUP_DOWNLOAD
    assert horner.checksum.startswith("sha256:")
    assert horner.counterpart_source_id == "westcott-hort"
    wh = load_manifest(MANIFESTS_DIR / "westcott-hort.json")
    assert wh.counterpart_source_id == "copsahhorner"
    assert "SBLGNT" not in wh.title


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
                id="westcott-hort",
                title="WH",
                version="1",
                url="https://example.invalid/wh",
                spdx_or_custom="custom:Public-Domain",
                attribution="Robinson",
                mode=AvailabilityMode.SETUP_DOWNLOAD,
                checksum="",
            )
        ]
    )
    assert any("checksum" in p for p in problems)


def test_pilot_corpus_committed() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    assert corpus.greek_source_id == "westcott-hort"
    assert corpus.sahidic_source_id == "copsahhorner"
    assert len(corpus.units) == 37
    john_1_1 = next(u for u in corpus.units if u.unit_id == "john.1.1")
    assert "λογος" in john_1_1.greek.original_text
    assert "ⲡϣⲁϫⲉ" in john_1_1.sahidic.original_text
    assert john_1_1.sahidic.original_text != john_1_1.sahidic.normalized_text
    empty = [u for u in corpus.units if not u.sahidic.original_text]
    assert [u.unit_id for u in empty] == [f"john.8.{v}" for v in range(1, 12)]
    assert all(u.notes for u in empty)


def test_cli_sources_validate() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "copsahhorner" in result.stdout
    assert "westcott-hort" in result.stdout
    assert "Sahidica" in result.stdout


def test_cli_help_lists_sources() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sources" in result.stdout


def test_cli_sources_download_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "download", "--help"])
    assert result.exit_code == 0
    assert "CopSahHorner" in result.stdout


def test_find_repo_root_from_tests() -> None:
    assert find_repo_root(Path(__file__)) == REPO_ROOT


def test_wh_download_pins_match_manifest() -> None:
    wh = load_manifest(MANIFESTS_DIR / "westcott-hort.json")
    assert WH_TEXTONLY_FILES["JOH.WH"] == wh.checksum
    assert "MR.WH" in WH_TEXTONLY_FILES


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

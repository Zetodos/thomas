# SPDX-License-Identifier: Apache-2.0
"""Tests for source manifests, MorphGNT/OT/Dilley ingest, and CLI checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.dilley_thomas import (
    EXPECTED_LOGIA,
    INGEST_LEVEL,
    TT_TAG_URL,
    expected_unit_ids,
    load_thomas_corpus,
)
from zetodos_thomas.ingest.dilley_thomas import (
    SOURCE_ID as DILLEY_SOURCE_ID,
)
from zetodos_thomas.ingest.download import (
    DownloadError,
    verify_checksum,
)
from zetodos_thomas.ingest.manifests import (
    load_manifest,
    validate_default_manifests,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.morphgnt import MORPHGNT_BOOKS, load_morphgnt_nt
from zetodos_thomas.ingest.normalize import normalize_coptic, normalize_greek
from zetodos_thomas.ingest.ot_swete import load_ot_swete_pairs, load_sahidic_ot_chapters
from zetodos_thomas.ingest.paths import (
    find_repo_root,
    resolve_manifests_dir,
    resolve_thomas_path,
)
from zetodos_thomas.ingest.verify_inputs import verify_downloaded_sources
from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION
from zetodos_thomas.schemas.source import (
    GATED_DEFAULT_SOURCE_IDS,
    AvailabilityMode,
    SourceManifest,
    UnitKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
DOWNLOADS_DIR = REPO_ROOT / "data" / "downloads"
THOMAS_PATH = REPO_ROOT / "data" / "sources" / "thomas" / "dilley-thomas-v6.2.0.json"
FIXTURES = Path(__file__).parent / "fixtures"


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
    assert ids == {
        "apostolic-fathers",
        "ccl-ddglc-tei",
        "copsahhorner",
        "dclp-poxy-1",
        "dclp-poxy-654",
        "dclp-poxy-655",
        "dilley-thomas",
        "first1k-koine-prose",
        "morphgnt-sblgnt",
        "scriptorium-apophthegmata",
        "scriptorium-sahidic-ot",
        "strongs-greek",
        "swete-lxx",
    }
    assert ids.isdisjoint(GATED_DEFAULT_SOURCE_IDS)

    horner = load_manifest(MANIFESTS_DIR / "copsahhorner.json")
    assert horner.mode is AvailabilityMode.SETUP_DOWNLOAD
    assert horner.checksum.startswith("sha256:")
    assert horner.counterpart_source_id == "morphgnt-sblgnt"

    mg = load_manifest(MANIFESTS_DIR / "morphgnt-sblgnt.json")
    assert mg.counterpart_source_id == "copsahhorner"
    assert mg.spdx_or_custom == "CC-BY-SA-3.0"

    dilley = load_manifest(MANIFESTS_DIR / "dilley-thomas.json")
    assert dilley.mode is AvailabilityMode.SETUP_DOWNLOAD
    assert dilley.spdx_or_custom == "CC-BY-4.0"
    assert dilley.checksum.startswith("sha256:")
    assert dilley.url == TT_TAG_URL
    assert "Dilley" in dilley.attribution or "Dilley" in dilley.title


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
                id="morphgnt-sblgnt",
                title="MorphGNT",
                version="1",
                url="https://example.invalid/morphgnt",
                spdx_or_custom="CC-BY-SA-3.0",
                attribution="MorphGNT",
                mode=AvailabilityMode.SETUP_DOWNLOAD,
                checksum="",
            )
        ]
    )
    assert any("checksum" in p for p in problems)


def test_thomas_corpus_committed() -> None:
    corpus = load_thomas_corpus(THOMAS_PATH)
    assert corpus.source_id == DILLEY_SOURCE_ID
    assert corpus.id_scheme_version == ID_SCHEME_VERSION
    assert corpus.annotation_level == INGEST_LEVEL
    assert [u.unit_id for u in corpus.units] == expected_unit_ids()
    assert [u.chapter for u in corpus.units] == list(range(EXPECTED_LOGIA + 1))
    assert all(u.original_text for u in corpus.units)
    assert all(u.normalized_text for u in corpus.units)
    assert all(u.morphemes for u in corpus.units)

    prologue = corpus.units[0]
    assert prologue.unit_id == "thomas.0"
    assert prologue.chapter == 0
    assert prologue.kind is UnitKind.PROLOGUE
    assert "ⲛⲁⲉⲓ" in prologue.original_text
    assert prologue.normalized_text.split()[0] == "ⲛⲁⲓ"
    assert corpus.units[7].unit_id == "thomas.7"
    assert corpus.units[7].kind is UnitKind.LOGION
    assert corpus.units[114].unit_id == "thomas.114"


def test_morphgnt_nt_loads_all_27_books() -> None:
    mg_dir = DOWNLOADS_DIR / "morphgnt-sblgnt"
    if not mg_dir.is_dir():
        pytest.skip("MorphGNT not downloaded in this test environment")
    assert len(MORPHGNT_BOOKS) == 27
    units = load_morphgnt_nt(mg_dir)
    assert len(units) > 7900
    john1_1 = units["john.1.1"]
    assert john1_1.tokens[0].lemma == "ἐν"
    assert john1_1.tokens[0].morph.pos == "P-"
    assert any(t.lemma == "λόγος" and t.morph.case == "N" for t in john1_1.tokens)


def test_sahidic_ot_and_swete_lxx_pairs() -> None:
    ot_zip = DOWNLOADS_DIR / "sahidic-ot" / "sahidic.ot_TT.zip"
    swete_dir = DOWNLOADS_DIR / "lxx-swete"
    if not (ot_zip.is_file() and swete_dir.is_dir()):
        pytest.skip("Sahidic OT or Swete LXX not downloaded in this test environment")
    ot = load_sahidic_ot_chapters(ot_zip)
    assert len(ot["Ruth"]) >= 80
    assert len(ot["Jonah"]) >= 45
    pairs = load_ot_swete_pairs(tt_zip=ot_zip, swete_dir=swete_dir)
    assert len(pairs) >= 15000
    gen1_1 = next(g for c, g in pairs if c.unit_id == "ot.Genesis.1.1")
    assert gen1_1.tokens[0].normalized == "εν"
    assert gen1_1.tokens[2].normalized == "ἐποίησεν"
    assert gen1_1.tokens[4].normalized == "θεὸς"


def test_cli_sources_validate() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "copsahhorner" in result.stdout
    assert "morphgnt-sblgnt" in result.stdout
    assert "scriptorium-sahidic-ot" in result.stdout
    assert "swete-lxx" in result.stdout
    assert "dilley-thomas" in result.stdout
    assert "thomas.0..thomas.114" in result.stdout
    assert "token-complete" in result.stdout
    assert "fragments" in result.stdout.lower()
    assert "evidence pilot" in result.stdout.lower()
    assert "Sahidica" in result.stdout


def test_cli_help_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "setup" in result.stdout
    assert "sources" in result.stdout


def test_find_repo_root_from_tests() -> None:
    assert find_repo_root(Path(__file__)) == REPO_ROOT


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


def test_resolve_paths_prefer_checkout() -> None:
    assert resolve_manifests_dir(REPO_ROOT) == MANIFESTS_DIR
    assert resolve_thomas_path(REPO_ROOT) == THOMAS_PATH


def test_verify_downloaded_sources_rejects_missing_dir(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for p in MANIFESTS_DIR.glob("*.json"):
        manifests.joinpath(p.name).write_text(
            p.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    with pytest.raises(DownloadError, match="missing"):
        verify_downloaded_sources(
            manifests_dir=manifests,
            downloads_dir=downloads,
        )

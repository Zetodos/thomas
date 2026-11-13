# SPDX-License-Identifier: Apache-2.0
"""Tests for source manifests, Vaticanus B* TEI, and pilot import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.align.tokenize import TokenCountMismatchError, whitespace_tokens
from zetodos_thomas.bundled import (
    bundled_manifests_dir,
    bundled_pilot_path,
    bundled_thomas_path,
)
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.dilley_thomas import (
    EXPECTED_LOGIA,
    INGEST_LEVEL,
    TT_TAG_URL,
    expected_unit_ids,
    load_thomas_corpus,
    parse_tt,
    validate_thomas_unit_sequence,
)
from zetodos_thomas.ingest.dilley_thomas import (
    SOURCE_ID as DILLEY_SOURCE_ID,
)
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
from zetodos_thomas.ingest.paths import (
    find_repo_root,
    resolve_manifests_dir,
    resolve_thomas_path,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.ingest.verify_inputs import verify_pilot_rebuild_inputs
from zetodos_thomas.schemas.alignment import ThomasCorpus
from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION
from zetodos_thomas.schemas.source import (
    GATED_DEFAULT_SOURCE_IDS,
    AvailabilityMode,
    SourceManifest,
    UnitKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
THOMAS_PATH = REPO_ROOT / "data" / "sources" / "thomas" / "dilley-thomas-v6.2.0.json"
FIXTURES = Path(__file__).parent / "fixtures"
NTVMR_FIXTURES = FIXTURES / "ntvmr"
DILLEY_FIXTURE = FIXTURES / "dilley" / "thomas_gospel_snippet.tt"


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
    assert ids == {
        "ccl-ddglc-tei",
        "copsahhorner",
        "dilley-thomas",
        "ntvmr-vaticanus-b",
        "dclp-poxy-1",
        "dclp-poxy-654",
        "dclp-poxy-655",
    }
    assert ids.isdisjoint(GATED_DEFAULT_SOURCE_IDS)
    horner = load_manifest(MANIFESTS_DIR / "copsahhorner.json")
    assert horner.mode is AvailabilityMode.SETUP_DOWNLOAD
    assert horner.checksum.startswith("sha256:")
    assert horner.counterpart_source_id == "ntvmr-vaticanus-b"
    vat = load_manifest(MANIFESTS_DIR / "ntvmr-vaticanus-b.json")
    assert vat.counterpart_source_id == "copsahhorner"
    assert vat.spdx_or_custom == "CC-BY-4.0"
    assert "westcott" not in vat.id
    # Vaticanus B* TEI / notes must remain the sole default Greek NT spine file.
    assert (MANIFESTS_DIR / "ntvmr-vaticanus-b.json").is_file()
    dilley = load_manifest(MANIFESTS_DIR / "dilley-thomas.json")
    assert dilley.mode is AvailabilityMode.SETUP_DOWNLOAD
    assert dilley.spdx_or_custom == "CC-BY-4.0"
    assert dilley.checksum.startswith("sha256:")
    assert dilley.url == TT_TAG_URL
    assert "Dilley" in dilley.attribution or "Dilley" in dilley.title


def test_parse_dilley_fixture_orig_norm_layers() -> None:
    corpus = parse_tt(DILLEY_FIXTURE, expect_complete=False)
    assert [u.chapter for u in corpus.units] == [0, 1, 2]
    prologue = corpus.units[0]
    assert prologue.unit_id == "thomas.0"
    assert prologue.kind is UnitKind.PROLOGUE
    assert prologue.original_text.startswith("ⲛⲁⲉⲓ")
    assert prologue.normalized_text.startswith("ⲛⲁⲓ")
    assert prologue.original_text != prologue.normalized_text
    assert INGEST_LEVEL in corpus.notes
    assert "not alignment-ready" in corpus.notes
    # Shared JAK-32 ThomasCorpus: tokens/spans present but empty at unit-level.
    assert corpus.tokens == []
    assert corpus.spans == []


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
    assert len(corpus.units) == 407
    john_1_1 = next(u for u in corpus.units if u.unit_id == "john.1.1")
    assert "λογος" in john_1_1.greek.original_text
    assert "ⲡϣⲁϫⲉ" in john_1_1.sahidic.original_text
    john_1_4 = next(u for u in corpus.units if u.unit_id == "john.1.4")
    assert "ανθρωπων" not in john_1_4.greek.original_text
    assert corpus.id == "horner-vaticanus-nt-pilot-v3"
    assert any(u.unit_id == "matthew.13.31" for u in corpus.units)
    assert any(u.unit_id == "matthew.7.7" for u in corpus.units)
    empty = [u for u in corpus.units if not u.sahidic.original_text]
    assert [u.unit_id for u in empty] == [f"john.8.{v}" for v in range(1, 12)]
    assert all(not u.greek.original_text for u in empty)
    assert all(u.notes for u in empty)


def test_thomas_corpus_committed() -> None:
    corpus = load_thomas_corpus(THOMAS_PATH)
    assert corpus.source_id == DILLEY_SOURCE_ID
    assert corpus.id_scheme_version == ID_SCHEME_VERSION
    assert INGEST_LEVEL in corpus.notes
    assert [u.unit_id for u in corpus.units] == expected_unit_ids()
    assert [u.chapter for u in corpus.units] == list(range(EXPECTED_LOGIA + 1))
    assert all(u.original_text for u in corpus.units)
    assert all(u.normalized_text for u in corpus.units)
    # Recoverable distinct layers (Dilley orig vs norm) on prologue.
    prologue = corpus.units[0]
    assert prologue.unit_id == "thomas.0"
    assert prologue.chapter == 0
    assert prologue.kind is UnitKind.PROLOGUE
    assert "ⲛⲁⲉⲓ" in prologue.original_text
    assert prologue.normalized_text.split()[0] == "ⲛⲁⲓ"
    assert corpus.units[7].unit_id == "thomas.7"
    assert corpus.units[7].kind is UnitKind.LOGION
    assert corpus.units[114].unit_id == "thomas.114"
    assert corpus.tokens == []
    assert corpus.spans == []
    assert "unit-level" in corpus.notes.lower()
    assert "not alignment-ready" in corpus.notes.lower()
    assert "whitespace_tokens" in corpus.notes


def test_whitespace_tokens_rejects_dilley_units() -> None:
    corpus = load_thomas_corpus(THOMAS_PATH)
    with pytest.raises(TokenCountMismatchError, match=r"Dilley/SCRIPTORIUM"):
        whitespace_tokens(corpus.units[1])


def test_thomas_validate_rejects_wrong_sequence() -> None:
    good = load_thomas_corpus(THOMAS_PATH)
    # Drop prologue — count of logia alone would still be 114.
    broken_units = list(good.units[1:])
    with pytest.raises(ValueError, match=r"thomas\.0"):
        validate_thomas_unit_sequence(broken_units)
    # Duplicate last logion instead of a missing middle one.
    dup = list(good.units)
    dup[50] = good.units[114]
    with pytest.raises(ValueError, match=r"thomas\.0 through thomas\.114"):
        validate_thomas_unit_sequence(dup)
    # load_thomas_corpus must enforce the same check.
    tmp = THOMAS_PATH.parent / "_bad_sequence.json"
    try:
        bad = ThomasCorpus(
            id="bad",
            title="bad",
            source_id=DILLEY_SOURCE_ID,
            units=broken_units,
            notes=f"{INGEST_LEVEL}; not alignment-ready",
        )
        tmp.write_text(bad.model_dump_json(), encoding="utf-8")
        with pytest.raises(ValueError, match=r"thomas\.0"):
            load_thomas_corpus(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()


def test_cli_sources_validate() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "copsahhorner" in result.stdout
    assert "ntvmr-vaticanus-b" in result.stdout
    assert "dilley-thomas" in result.stdout
    assert "thomas.0..thomas.114" in result.stdout
    assert "unit-level" in result.stdout
    assert "not alignment-ready" in result.stdout
    assert "fragments" in result.stdout.lower() or "poxy" in result.stdout.lower()
    assert "evidence pilot" in result.stdout.lower()
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
    assert NTVMR_TEIRAW_FILES["John.1.1-51"] == vat.checksum
    assert "Mark.1.1-45" in NTVMR_TEIRAW_FILES


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
    for name in (
        "copsahhorner.json",
        "ntvmr-vaticanus-b.json",
        "dilley-thomas.json",
        "dclp-poxy-1.json",
        "dclp-poxy-654.json",
        "dclp-poxy-655.json",
    ):
        checkout = (MANIFESTS_DIR / name).read_text(encoding="utf-8")
        bundled = (bundled_manifests_dir() / name).read_text(encoding="utf-8")
        assert checkout == bundled
    assert PILOT_PATH.read_text(encoding="utf-8") == bundled_pilot_path().read_text(
        encoding="utf-8"
    )
    assert THOMAS_PATH.read_text(encoding="utf-8") == bundled_thomas_path().read_text(
        encoding="utf-8"
    )


def test_resolve_manifests_prefers_checkout() -> None:
    assert resolve_manifests_dir(REPO_ROOT) == MANIFESTS_DIR
    assert resolve_thomas_path(REPO_ROOT) == THOMAS_PATH


def test_verify_rebuild_rejects_bad_inputs(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in (
        "copsahhorner.json",
        "ntvmr-vaticanus-b.json",
        "dilley-thomas.json",
        "dclp-poxy-1.json",
        "dclp-poxy-654.json",
        "dclp-poxy-655.json",
    ):
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

# SPDX-License-Identifier: Apache-2.0
"""Tests for Greek Thomas fragment ingest (JAK-31)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.dclp_fragments import (
    DCLP_THOMAS_FRAGMENTS,
    EXCLUDED_NON_THOMAS_WITNESSES,
    parse_dclp_edition,
    read_dclp_license,
)
from zetodos_thomas.ingest.download import DownloadError
from zetodos_thomas.ingest.fragments import (
    FragmentValidationError,
    build_fragment_corpus,
    load_fragment_corpus,
    validate_fragment_corpus,
    validate_fragment_corpus_or_raise,
)
from zetodos_thomas.ingest.manifests import load_manifest, validate_manifests_dir
from zetodos_thomas.ingest.paths import resolve_fragments_corpus_path
from zetodos_thomas.ingest.verify_inputs import verify_fragment_rebuild_inputs
from zetodos_thomas.schemas.fragment import (
    CorpusRole,
    EvidenceType,
    FragmentLayerKind,
)
from zetodos_thomas.schemas.source import AvailabilityMode

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
TEI_DIR = REPO_ROOT / "data" / "sources" / "fragments" / "dclp"
CORPUS_PATH = (
    REPO_ROOT / "data" / "sources" / "fragments" / "greek-thomas-fragments.json"
)


def test_dclp_tei_license_is_cc_by_30() -> None:
    for meta in DCLP_THOMAS_FRAGMENTS:
        path = TEI_DIR / meta["filename"]
        license_url = read_dclp_license(path)
        assert license_url is not None
        assert "creativecommons.org/licenses/by/3.0" in license_url


def test_manifest_checksums_match_tei_bytes() -> None:
    for meta in DCLP_THOMAS_FRAGMENTS:
        tei = TEI_DIR / meta["filename"]
        digest = hashlib.sha256(tei.read_bytes()).hexdigest()
        manifest = load_manifest(MANIFESTS_DIR / f"{meta['source_id']}.json")
        assert manifest.checksum == f"sha256:{digest}"
        assert manifest.mode is AvailabilityMode.BUNDLED
        assert manifest.spdx_or_custom == "CC-BY-3.0"
        assert manifest.manuscript_id == meta["manuscript_id"]
        assert manifest.evidence_type == "G"
        assert manifest.corpus_role == "evaluation-witness"


def test_layers_keep_surviving_and_restoration_distinct() -> None:
    lines = parse_dclp_edition(TEI_DIR / "poxy-1-tm62838.xml")
    mixed = next(
        line
        for line in lines
        if any(t.kind is FragmentLayerKind.SURVIVING for t in line.tokens)
        and any(t.kind is FragmentLayerKind.RESTORATION for t in line.tokens)
    )
    surviving = "".join(
        t.text for t in mixed.tokens if t.kind is FragmentLayerKind.SURVIVING
    )
    restored = "".join(
        t.text for t in mixed.tokens if t.kind is FragmentLayerKind.RESTORATION
    )
    assert surviving
    assert restored
    # Restorations must not be silently folded into the surviving string alone.
    assert restored not in surviving or surviving != restored


def test_lacuna_layer_present_on_poxy_1() -> None:
    lines = parse_dclp_edition(TEI_DIR / "poxy-1-tm62838.xml")
    assert any(
        t.kind is FragmentLayerKind.LACUNA for line in lines for t in line.tokens
    )


def test_nested_gap_inside_supplied_survives_parse_and_corpus() -> None:
    """P.Oxy. 655 col. ii line 23: <supplied>…<gap/></supplied> must keep lacuna."""
    lines = parse_dclp_edition(TEI_DIR / "poxy-655-tm62839.xml")
    line = next(ln for ln in lines if ln.line_id == "column-ii.23")
    kinds = [t.kind for t in line.tokens]
    assert FragmentLayerKind.LACUNA in kinds
    lacuna = next(t for t in line.tokens if t.kind is FragmentLayerKind.LACUNA)
    assert lacuna.reason == "illegible"
    assert lacuna.unit == "character"
    # Derived JSON must retain the nested marker.
    corpus = load_fragment_corpus(CORPUS_PATH)
    tx = next(t for t in corpus.transcriptions if t.id == "poxy-655-dclp")
    derived = next(ln for ln in tx.lines if ln.line_id == "column-ii.23")
    assert any(t.kind is FragmentLayerKind.LACUNA for t in derived.tokens)


def test_verify_fragment_rebuild_rejects_checksum_mismatch(tmp_path: Path) -> None:
    tei_dir = tmp_path / "tei"
    tei_dir.mkdir()
    for meta in DCLP_THOMAS_FRAGMENTS:
        dest = tei_dir / meta["filename"]
        dest.write_bytes((TEI_DIR / meta["filename"]).read_bytes())
    # Tamper one TEI after copy — must not inherit pinned provenance.
    target = tei_dir / "poxy-1-tm62838.xml"
    target.write_bytes(target.read_bytes() + b"\n<!-- tampered -->\n")
    with pytest.raises(DownloadError, match="checksum mismatch"):
        verify_fragment_rebuild_inputs(manifests_dir=MANIFESTS_DIR, tei_dir=tei_dir)


def test_verify_fragment_rebuild_rejects_tei_dir_checksum_mismatch(
    tmp_path: Path,
) -> None:
    tei_dir = tmp_path / "tei"
    tei_dir.mkdir()
    for meta in DCLP_THOMAS_FRAGMENTS:
        (tei_dir / meta["filename"]).write_bytes(
            (TEI_DIR / meta["filename"]).read_bytes()
        )
    (tei_dir / "poxy-654-tm62840.xml").write_text(
        "not-the-pinned-tei", encoding="utf-8"
    )
    with pytest.raises(DownloadError, match="checksum"):
        verify_fragment_rebuild_inputs(manifests_dir=MANIFESTS_DIR, tei_dir=tei_dir)


def test_committed_fragment_corpus_validates() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    corpus = load_fragment_corpus(CORPUS_PATH)
    validate_fragment_corpus_or_raise(corpus, manifests)
    assert corpus.evidence_type is EvidenceType.G
    assert corpus.corpus_role is CorpusRole.EVALUATION_WITNESS
    ids = {tx.id for tx in corpus.transcriptions}
    assert ids == {"poxy-1-dclp", "poxy-654-dclp", "poxy-655-dclp"}
    assert any("5575" in item for item in corpus.excluded_witnesses)
    assert any("5575" in item for item in EXCLUDED_NON_THOMAS_WITNESSES)


def test_dangling_source_id_rejected() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    corpus = load_fragment_corpus(CORPUS_PATH)
    corpus.transcriptions[0].source_id = "missing-source"
    problems = validate_fragment_corpus(corpus, manifests)
    assert any("dangling source_id" in p for p in problems)


def test_missing_license_on_fragment_manifest_rejected() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    for manifest in manifests:
        if manifest.id == "dclp-poxy-1":
            manifest.spdx_or_custom = ""
            break
    corpus = load_fragment_corpus(CORPUS_PATH)
    problems = validate_fragment_corpus(corpus, manifests)
    assert any("spdx_or_custom" in p or "license" in p for p in problems)


def test_poxy_5575_mislabeled_rejected() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    corpus = load_fragment_corpus(CORPUS_PATH)
    corpus.transcriptions[0].manuscript_id = "P.Oxy. 5575"
    problems = validate_fragment_corpus(corpus, manifests)
    assert any("5575" in p for p in problems)


def test_build_fragments_roundtrip(tmp_path: Path) -> None:
    corpus = build_fragment_corpus(TEI_DIR)
    out = tmp_path / "greek-thomas-fragments.json"
    from zetodos_thomas.ingest.fragments import write_fragment_corpus

    write_fragment_corpus(corpus, out)
    reloaded = load_fragment_corpus(out)
    assert len(reloaded.transcriptions) == 3
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    validate_fragment_corpus_or_raise(reloaded, manifests)


def test_cli_sources_validate_includes_fragments() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "dclp-poxy-1" in result.stdout
    assert (
        "fragments" in result.stdout.lower()
        or "greek-thomas-fragments" in result.stdout
    )


def test_fragment_words_preserve_truncation_and_abbreviations() -> None:
    corpus = load_fragment_corpus(CORPUS_PATH)
    assert resolve_fragments_corpus_path(REPO_ROOT) == CORPUS_PATH
    tx654 = next(tx for tx in corpus.transcriptions if tx.id == "poxy-654-dclp")
    thalassa = next(w for w in tx654.words if w.full_editorial_word == "θαλάσσης")
    assert thalassa.surviving_text == "θαλά"
    assert thalassa.restored_text == "σσης"
    assert thalassa.is_truncated is True
    assert thalassa.is_abbreviation is False

    tx1 = next(tx for tx in corpus.transcriptions if tx.id == "poxy-1-dclp")
    is_words = [
        w for w in tx1.words if w.is_abbreviation and w.full_editorial_word == "Ἰησοῦς"
    ]
    assert is_words
    assert is_words[0].surviving_text == "Ἰς"
    assert is_words[0].abbreviated_text == "ησοῦ"


def test_validate_or_raise() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    corpus = load_fragment_corpus(CORPUS_PATH)
    corpus.transcriptions.clear()
    with pytest.raises(FragmentValidationError):
        validate_fragment_corpus_or_raise(corpus, manifests)

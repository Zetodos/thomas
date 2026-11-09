# SPDX-License-Identifier: Apache-2.0
"""Tests for Greek Thomas fragment ingest (JAK-31)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.bundled import (
    bundled_fragments_corpus_path,
    bundled_manifests_dir,
)
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.dclp_fragments import (
    DCLP_THOMAS_FRAGMENTS,
    EXCLUDED_NON_THOMAS_WITNESSES,
    parse_dclp_edition,
    read_dclp_license,
)
from zetodos_thomas.ingest.fragments import (
    FragmentValidationError,
    build_fragment_corpus,
    load_fragment_corpus,
    validate_fragment_corpus,
    validate_fragment_corpus_or_raise,
)
from zetodos_thomas.ingest.manifests import load_manifest, validate_manifests_dir
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


def test_bundled_fragment_corpus_matches_checkout() -> None:
    assert CORPUS_PATH.read_text(
        encoding="utf-8"
    ) == bundled_fragments_corpus_path().read_text(encoding="utf-8")
    for meta in DCLP_THOMAS_FRAGMENTS:
        name = f"{meta['source_id']}.json"
        assert (MANIFESTS_DIR / name).read_text(encoding="utf-8") == (
            bundled_manifests_dir() / name
        ).read_text(encoding="utf-8")


def test_validate_or_raise() -> None:
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    corpus = load_fragment_corpus(CORPUS_PATH)
    corpus.transcriptions.clear()
    with pytest.raises(FragmentValidationError):
        validate_fragment_corpus_or_raise(corpus, manifests)

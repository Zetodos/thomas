# SPDX-License-Identifier: Apache-2.0
"""Strict AST, runtime refusal, checksum, and provenance zero-leakage checks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.ingest.holdouts import (
    SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS,
    HoldoutValidationError,
    assert_clean_training_unit,
    clean_nt_train_ids,
    load_manifest_sha256,
    load_nt_split,
    sha256_file,
    sha256_or_lfs_oid,
)
from zetodos_thomas.lexicon.idioms import IdiomAndValencyTable
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.linguistic import (
    CopticMorpheme,
    CopticUnit,
    GreekMorphFeatures,
    GreekToken,
    GreekUnit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "python" / "zetodos_thomas"
SPLIT_PATH = REPO_ROOT / "data" / "training" / "splits" / "nt-split-v1.json"
CCL_MANIFEST = REPO_ROOT / "data" / "manifests" / "ccl-ddglc-tei.json"

# Training & reconstruction packages that must NEVER touch fragment witnesses
SEALED_MODEL_PACKAGES = ("morphology", "lexicon", "align", "reconstruct", "neural")
FORBIDDEN_FRAG_IMPORTS = (
    "dclp_fragments",
    "fragments",
    "fragment_clause_map",
    "eval",
)
FORBIDDEN_PATH_LITERALS = (
    "greek-thomas-fragments",
    "fragment-clause-map",
    "dclp/",
)
FORBIDDEN_GATED_LITERALS = (
    "sahidica",
    "ntaeg",
    "bohairica",
    "ud_coptic",
    "coptic_scriptorium_mark",
)
# Hardcoded Thomas/P.Oxy. lookup tables permanently banned from the codebase
BANNED_CURATED_SYMBOLS = (
    "_CURATED_IDIOMS",
    "_KNOWN_VERB_OBJECT_CASES",
    "_PROPER_AND_COMPOUND_FALLBACKS",
    "_POXY_LEX_STEM_FALLBACK",
)


def _dummy_coptic(unit_id: str, source_id: str = "copsahhorner") -> CopticUnit:
    return CopticUnit(
        unit_id=unit_id,
        source_id=source_id,
        book="john",
        chapter=1,
        verse=1,
        original_text="ⲡ ϣⲁϫⲉ",
        normalized_text="ⲡ ϣⲁϫⲉ",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲡ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(group_seq=0, norm="ϣⲁϫⲉ", lemma="ϣⲁϫⲉ", pos="N"),
        ],
    )


def _dummy_greek(unit_id: str, source_id: str = "morphgnt-sblgnt") -> GreekUnit:
    return GreekUnit(
        unit_id=unit_id,
        source_id=source_id,
        book="john",
        chapter=1,
        verse=1,
        original_text="λόγος",
        normalized_text="λόγος",
        tokens=[
            GreekToken(
                id=f"{unit_id}.0",
                unit_id=unit_id,
                seq=0,
                surface="λόγος",
                normalized="λόγος",
                lemma="λόγος",
                morph=GreekMorphFeatures.from_morphgnt("N-", "----NSM-"),
            )
        ],
    )


def test_model_packages_never_import_fragments_or_eval() -> None:
    """AST scan ensuring morphology/lexicon/align/reconstruct/neural stay sealed."""
    for pkg in SEALED_MODEL_PACKAGES:
        pkg_dir = PKG_ROOT / pkg
        for py_file in pkg_dir.rglob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in FORBIDDEN_FRAG_IMPORTS:
                            assert bad not in alias.name, (
                                f"{py_file} imports forbidden module {alias.name!r}"
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for bad in FORBIDDEN_FRAG_IMPORTS:
                        assert bad not in node.module, (
                            f"{py_file} imports from forbidden {node.module!r}"
                        )
            low = src.lower()
            for lit in FORBIDDEN_PATH_LITERALS:
                assert lit not in low, (
                    f"{py_file} references sealed path literal {lit!r}"
                )
            for gated in FORBIDDEN_GATED_LITERALS:
                assert gated not in low, f"{py_file} references gated corpus {gated!r}"


def test_no_banned_curated_thomas_or_poxy_tables() -> None:
    """Ensure hardcoded idiom/loanword/eval fallback dicts stay permanently deleted."""
    for py_file in PKG_ROOT.rglob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        for sym in BANNED_CURATED_SYMBOLS:
            assert sym not in src, f"banned hardcoded table {sym} found in {py_file}"


def test_thomas_not_used_by_morphology_trainer() -> None:
    src = (
        (PKG_ROOT / "morphology" / "coptic_analyzer.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "dilley-thomas" not in src
    assert "thomas_gospel" not in src


def test_synoptic_thomas_parallels_excluded_from_clean_nt_train() -> None:
    split = load_nt_split(SPLIT_PATH)
    clean_ids = clean_nt_train_ids(split, exclude_synoptic_thomas=True)
    assert len(SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS) >= 50
    assert not (clean_ids & SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS)
    # Spot-check key Synoptic parallels of P.Oxy. logia (e.g. logia 1, 26, 31, 36, 39)
    for uid in (
        "matthew.7.5",
        "john.8.51",
        "luke.4.24",
        "luke.8.17",
        "matthew.6.25",
        "matthew.23.13",
    ):
        assert uid in SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
        assert uid not in clean_ids
    assert len(clean_ids) >= 7700


def test_runtime_refusal_of_thomas_poxy_and_excluded_units() -> None:
    """Training builders must reject Thomas, P.Oxy., and holdout units at runtime."""
    for bad_uid, bad_src in (
        ("thomas.1", "dilley-thomas"),
        ("poxy-654-dclp.1", "poxy-654-dclp"),
        ("matthew.13.31", "morphgnt-sblgnt"),
        ("john.1.1", "dilley-thomas-v6.2.0"),
    ):
        with pytest.raises(HoldoutValidationError):
            assert_clean_training_unit(
                bad_uid,
                source_id=bad_src,
                excluded_ids={"matthew.13.31"},
            )

    # IdiomAndValencyTable builders must refuse Thomas / fragment units
    with pytest.raises(HoldoutValidationError):
        IdiomAndValencyTable.build_from_morphgnt([_dummy_greek("thomas.1")])
    with pytest.raises(HoldoutValidationError):
        IdiomAndValencyTable.build_from_bitext(
            nt_pairs=[(_dummy_coptic("thomas.26"), _dummy_greek("poxy-1-dclp.0"))]
        )

    # LexicalTranslationModel.build_from_bitext must refuse Thomas / fragment units
    inf = GreekInflector(
        paradigms={}, coarse_paradigms={}, lemma_genders={}, surface_analyses={}
    )
    with pytest.raises(HoldoutValidationError):
        LexicalTranslationModel.build_from_bitext(
            nt_pairs=[(_dummy_coptic("thomas.1"), _dummy_greek("thomas.1"))],
            ot_pairs=[],
            inflector=inf,
        )


@pytest.mark.parametrize("model_name", ("model-v1", "model-v2"))
def test_frozen_model_provenance_and_ccl_checksum(model_name: str) -> None:
    """Verify cryptographic corpus provenance, CCL checksum, and LFS weight OIDs."""
    model_dir = REPO_ROOT / "data" / "mappings" / model_name
    prov_path = model_dir / "training-provenance.json"
    if not prov_path.is_file():
        pytest.skip(f"{prov_path} not generated yet")

    expected_ccl_sha = load_manifest_sha256(CCL_MANIFEST)
    expected_split_sha = sha256_file(SPLIT_PATH)

    ccl_json = json.loads(
        (model_dir / "ccl-loanwords.json").read_text(encoding="utf-8")
    )
    assert ccl_json.get("source_sha256") == expected_ccl_sha

    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["model_id"] == model_name
    assert prov["nt_split_sha256"] == expected_split_sha
    assert prov["ccl_xml_sha256"] == expected_ccl_sha
    assert prov["zero_curated_thomas_idioms"] is True
    assert prov["zero_ccl_manual_fallbacks"] is True
    assert prov["excluded_holdout_count"] == 41
    assert prov["excluded_dev_count"] == 51
    assert prov["excluded_synoptic_thomas_count"] == len(
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
    )
    assert set(prov["excluded_synoptic_thomas_unit_ids"]) == set(
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
    )

    artifact_hashes: dict[str, str] = prov["artifact_sha256"]
    assert artifact_hashes, "training-provenance.json must hash all frozen artifacts"
    for fname, expected_digest in artifact_hashes.items():
        fpath = model_dir / fname
        assert fpath.is_file(), f"missing provenance-tracked file {fpath}"
        # Handles both materialized files locally and Git LFS pointer files on CI
        assert sha256_or_lfs_oid(fpath) == expected_digest, (
            f"SHA-256/LFS-OID mismatch for {fpath}"
        )

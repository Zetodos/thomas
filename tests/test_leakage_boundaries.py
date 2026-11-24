# SPDX-License-Identifier: Apache-2.0
"""Strict AST, runtime refusal, checksum, and provenance zero-leakage checks."""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from zetodos_thomas.align.lemma_aligner import LexicalTranslationModel
from zetodos_thomas.eval.benchmark import strip_greek_diacritics
from zetodos_thomas.ingest.dilley_thomas import load_thomas_corpus
from zetodos_thomas.ingest.fragments import load_fragment_corpus
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
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt
from zetodos_thomas.ingest.paths import (
    resolve_fragments_corpus_path,
    resolve_thomas_path,
)
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon
from zetodos_thomas.lexicon.idioms import IdiomAndValencyTable
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.reconstruct.decoder import KoineBeamDecoder
from zetodos_thomas.schemas.evidence import FragmentClauseMap
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
CCL_XML_PATH = (
    REPO_ROOT
    / "data"
    / "downloads"
    / "ccl-ddglc"
    / "Comprehensive_Coptic_Lexicon-v1.2-2020.xml"
)
FRAGMENT_CLAUSE_MAP_PATH = REPO_ROOT / "data" / "evidence" / "fragment-clause-map.json"
MORPHGNT_DIR = REPO_ROOT / "data" / "downloads" / "morphgnt-sblgnt"

# Keys previously added via manual _PROPER_AND_COMPOUND_FALLBACKS (absent from pure CCL)
DELETED_CCL_FALLBACK_KEYS = (
    "ⲇⲓⲇⲩⲙⲟⲥ",
    "ⲑⲱⲙⲁⲥ",
    "ⲓⲏⲥⲟⲩⲥ",
    "ⲣⲛⲏⲥⲧⲉⲩⲉ",
    "ⲣⲑⲉⲣⲁⲡⲉⲩⲉ",
    "ⲣⲡⲁⲣⲁⲧⲏⲣⲉⲓ",
    "ϯⲉⲗⲉⲏⲙⲟⲥⲩⲛⲏ",
    "ⲣⲉⲡⲓⲑⲩⲙⲉⲓ",
)
# Previously hand-curated Thomas/P.Oxy.-targeted idioms absent from empirical NT/OT MWE
BANNED_THOMAS_CRIB_IDIOMS = (
    ("ϫⲓ", "ϯⲡⲉ"),
    ("ⲣ", "ⲟⲩⲁ"),
    ("ⲥⲁ", "ϩⲟⲩⲛ"),
    ("ⲥⲁ", "ⲃⲟⲗ"),
    ("ϯ", "ⲙⲧⲟⲛ"),
)
# Generic Gospel speech-introduction 5-gram (not saying body) allowed in NT train
GENERIC_NARRATIVE_5GRAMS = frozenset({"λεγουσιν αυτω οι μαθηται αυτου"})

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


def test_clean_nt_train_retains_canonical_nt_and_supports_synoptic_ablation() -> None:
    split = load_nt_split(SPLIT_PATH)
    # Default policy: retain full canonical NT TRAIN (7,782 verses including Synoptic
    # parallels), excluding only NT HOLDOUT (41), DEV (51), omissions, and Thomas/P.Oxy.
    default_train = clean_nt_train_ids(split)
    assert len(default_train) == 7782
    for uid in (
        "matthew.7.5",
        "john.8.51",
        "luke.4.24",
        "luke.8.17",
        "matthew.6.25",
        "matthew.23.13",
    ):
        assert uid in SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
        assert uid in default_train

    # Opt-in Synoptic-exclusion ablation remains supported cleanly
    ablated_train = clean_nt_train_ids(split, exclude_synoptic_thomas=True)
    assert len(SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS) >= 50
    assert not (ablated_train & SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS)
    assert len(ablated_train) == len(default_train) - len(
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
    )


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


@pytest.mark.parametrize("model_name", ("model-v1", "model-v2", "model-v3"))
def test_frozen_model_provenance_and_ccl_checksum(model_name: str) -> None:
    """Verify corpus split/CCL checksums and manifest-to-artifact digest consistency.

    Note: on CI where large weights are checked out as Git LFS pointer files,
    ``sha256_or_lfs_oid`` verifies pointer-to-provenance manifest consistency
    (not independent weight provenance, which requires materializing LFS blobs
    or re-running deterministic ``thomas model build-v2``).
    """
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
    # Do not rely on hardcoded constant assertion booleans in provenance JSON
    assert "zero_curated_thomas_idioms" not in prov
    assert "zero_ccl_manual_fallbacks" not in prov
    assert prov["excluded_holdout_count"] == 41
    assert prov["excluded_dev_count"] == 51
    assert prov["excluded_synoptic_thomas_count"] == len(
        prov["excluded_synoptic_thomas_unit_ids"]
    )
    assert set(prov["excluded_synoptic_thomas_unit_ids"]).issubset(
        SYNOPTIC_THOMAS_PARALLEL_EXCLUSIONS
    )

    artifact_hashes: dict[str, str] = prov["artifact_sha256"]
    assert artifact_hashes, "training-provenance.json must hash all frozen artifacts"
    for fname, expected_digest in artifact_hashes.items():
        fpath = model_dir / fname
        assert fpath.is_file(), f"missing provenance-tracked file {fpath}"
        assert sha256_or_lfs_oid(fpath) == expected_digest, (
            f"SHA-256/LFS-OID mismatch for {fpath}"
        )


def test_real_thomas_and_fragment_corpora_refused_at_runtime() -> None:
    """Read actual Dilley Thomas & P.Oxy. corpora and verify runtime refusal."""
    thomas = load_thomas_corpus(resolve_thomas_path(REPO_ROOT))
    assert len(thomas.units) == 115
    for unit in thomas.units:
        with pytest.raises(HoldoutValidationError):
            assert_clean_training_unit(unit.unit_id, source_id=unit.source_id)

    frags = load_fragment_corpus(resolve_fragments_corpus_path(REPO_ROOT))
    assert len(frags.transcriptions) == 3
    for tx in frags.transcriptions:
        with pytest.raises(HoldoutValidationError):
            assert_clean_training_unit(tx.id, source_id=tx.source_id)

    # Passing real Dilley Thomas units directly into builders must raise
    thomas_7 = next(u for u in thomas.units if u.unit_id == "thomas.7")
    with pytest.raises(HoldoutValidationError):
        IdiomAndValencyTable.build_from_bitext(
            nt_pairs=[(thomas_7, _dummy_greek("john.1.1"))]
        )
    inf = GreekInflector(
        paradigms={}, coarse_paradigms={}, lemma_genders={}, surface_analyses={}
    )
    with pytest.raises(HoldoutValidationError):
        LexicalTranslationModel.build_from_bitext(
            nt_pairs=[(thomas_7, _dummy_greek("john.1.1"))],
            ot_pairs=[],
            inflector=inf,
        )


@pytest.mark.parametrize("model_name", ("model-v1", "model-v2", "model-v3"))
def test_committed_ccl_and_idiom_tables_contain_no_manual_fallbacks(
    model_name: str, tmp_path: Path
) -> None:
    """Read actual ccl-loanwords.json & idiom-valency.json and verify purity."""
    model_dir = REPO_ROOT / "data" / "mappings" / model_name
    ccl_path = model_dir / "ccl-loanwords.json"
    if not ccl_path.is_file():
        pytest.skip(f"{ccl_path} not generated yet")
    ccl_data = json.loads(ccl_path.read_text(encoding="utf-8"))
    loanwords = ccl_data["loanwords"]
    assert len(loanwords) > 5000
    for banned_key in DELETED_CCL_FALLBACK_KEYS:
        assert banned_key not in loanwords, (
            f"Deleted manual fallback {banned_key!r} found in {ccl_path}"
        )

    # When CCL TEI XML is present locally, prove byte-for-byte parity with a fresh parse
    if CCL_XML_PATH.is_file():
        expected_sha = load_manifest_sha256(CCL_MANIFEST)
        fresh = CclLexicon.from_tei_xml(CCL_XML_PATH, expected_sha256=expected_sha)
        fresh_path = tmp_path / "fresh-ccl.json"
        fresh.save_json(fresh_path)
        assert ccl_path.read_bytes() == fresh_path.read_bytes(), (
            f"{ccl_path} diverges from a pure fresh parse of {CCL_XML_PATH.name}"
        )

    idiom_path = model_dir / "idiom-valency.json"
    if idiom_path.is_file():
        idiom_data = json.loads(idiom_path.read_text(encoding="utf-8"))
        idiom_keys = {tuple(row[0]) for row in idiom_data["idioms"]}
        for banned_idiom in BANNED_THOMAS_CRIB_IDIOMS:
            assert banned_idiom not in idiom_keys, (
                f"Banned Thomas-crib idiom {banned_idiom!r} found in {idiom_path}"
            )


def test_poxy_surviving_5grams_absent_from_clean_nt_train() -> None:
    """Verify no saying-body P.Oxy. unrestored 5-gram appears in clean NT train."""
    frags = load_fragment_corpus(resolve_fragments_corpus_path(REPO_ROOT))
    frag_by_id = {tx.id: tx for tx in frags.transcriptions}
    cmap = FragmentClauseMap.model_validate(
        json.loads(FRAGMENT_CLAUSE_MAP_PATH.read_text(encoding="utf-8"))
    )

    poxy_5grams: set[str] = set()
    for span in cmap.spans:
        tx = frag_by_id[span.transcription_id]
        run: list[str] = []
        for w in tx.words[span.frag_word_start : span.frag_word_end]:
            if (
                w.surviving_text.strip()
                and not w.is_truncated
                and not w.restored_text.strip()
                and not w.is_abbreviation
            ):
                norm = strip_greek_diacritics(w.surviving_text)
                if norm:
                    run.append(norm)
                    if len(run) >= 5:
                        poxy_5grams.add(" ".join(run[-5:]))
                    continue
            run.clear()
    assert len(poxy_5grams) >= 50

    if not MORPHGNT_DIR.is_dir():
        pytest.skip("MorphGNT not downloaded locally; skipping NT 5-gram scan")

    split = load_nt_split(SPLIT_PATH)
    clean_ids = clean_nt_train_ids(split, exclude_synoptic_thomas=True)
    morphgnt = load_morphgnt_nt(MORPHGNT_DIR)

    overlaps: list[tuple[str, str]] = []
    for uid in sorted(clean_ids):
        unit = morphgnt.get(uid)
        if unit is None:
            continue
        toks = [
            strip_greek_diacritics(t.surface)
            for t in unit.tokens
            if strip_greek_diacritics(t.surface)
        ]
        for i in range(max(0, len(toks) - 4)):
            ng = " ".join(toks[i : i + 5])
            if ng in poxy_5grams and ng not in GENERIC_NARRATIVE_5GRAMS:
                overlaps.append((uid, ng))
    assert not overlaps, f"Unexpected P.Oxy. 5-gram overlap in clean NT: {overlaps}"


def test_runtime_decode_never_opens_fragments_and_mutation_breaks_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ported JAK-40/integrity checks: sealed decode at runtime & tamper detection."""
    thomas = load_thomas_corpus(resolve_thomas_path(REPO_ROOT))
    thomas_7 = next(u for u in thomas.units if u.unit_id == "thomas.7")

    real_open: Any = builtins.open
    real_path_open: Any = Path.open

    def _check_path(file: object) -> None:
        s = str(file).lower()
        for forbidden in ("greek-thomas-fragments", "fragment-clause-map", "dclp"):
            if forbidden in s:
                raise AssertionError(f"Decoder opened forbidden witness file: {file}")

    def _guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        _check_path(file)
        return real_open(file, *args, **kwargs)

    def _guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        _check_path(self)
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)
    monkeypatch.setattr(Path, "open", _guarded_path_open)
    # Self-check: guard must actually intercept reading the fragment corpus
    with pytest.raises(AssertionError, match="forbidden witness"):
        resolve_fragments_corpus_path(REPO_ROOT).read_text(encoding="utf-8")
    decoder = KoineBeamDecoder.from_model_dir(
        REPO_ROOT / "data" / "mappings" / "model-v1"
    )
    gen = decoder.decode_unit(thomas_7)
    assert gen.greek_text.strip()
    assert len(gen.tokens) > 0
    monkeypatch.undo()

    # Mutating a single byte of a provenance-tracked file must fail digest verification
    sample = tmp_path / "artifact.json"
    sample.write_text('{"id":"model-v1"}\n', encoding="utf-8")
    digest = sha256_or_lfs_oid(sample)
    sample.write_text('{"id":"tampered"}\n', encoding="utf-8")
    assert sha256_or_lfs_oid(sample) != digest

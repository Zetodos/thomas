# SPDX-License-Identifier: Apache-2.0
"""Load JAK-37/38 evidence into an offline decode model (JAK-39)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zetodos_thomas.align.splits import SplitRole, load_canonical_splits
from zetodos_thomas.ingest.paths import (
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
)
from zetodos_thomas.lexicon.candidates import (
    DEFAULT_PHASE2_ALIGNERS,
    default_table_path,
    load_lexical_candidate_table,
)
from zetodos_thomas.lexicon.lemmas import infer_greek_lemma
from zetodos_thomas.lexicon.validate import validate_lexical_candidate_table
from zetodos_thomas.mappings.ablation import default_ablation_paths
from zetodos_thomas.mappings.loanwords import (
    default_verified_loanwords_path,
    load_verified_loanword_table,
)
from zetodos_thomas.mappings.phrases import (
    default_statistical_constructions_path,
    load_statistical_construction_table,
)
from zetodos_thomas.mappings.validate import (
    validate_statistical_construction_table,
    validate_verified_loanword_table,
)
from zetodos_thomas.schemas.lexicon import (
    LexicalCandidateTable,
    LexicalLayer,
    ObservedWordformCandidate,
)
from zetodos_thomas.schemas.mappings import (
    AblationCondition,
    AblationPackage,
    StatisticalConstruction,
    StatisticalConstructionTable,
    VerifiedLoanword,
    VerifiedLoanwordTable,
)

PHASE2_ALIGNERS = tuple(a.value for a in DEFAULT_PHASE2_ALIGNERS)
PROVENANCE_KEYS = (
    "pilot_corpus_id",
    "split_id",
    "greek_source_id",
    "sahidic_source_id",
    "coptic_segmentation",
    "bitext_checksum_sha256",
)


@dataclass
class DecodeModel:
    """Frozen evidence tables for one ablation arm (no network, no LLM)."""

    condition: AblationCondition
    package: AblationPackage
    lexical: LexicalCandidateTable
    constructions: StatisticalConstructionTable
    loanwords: VerifiedLoanwordTable | None
    # Indexes
    constructions_by_id: dict[str, StatisticalConstruction] = field(
        default_factory=dict
    )
    constructions_by_pattern: dict[tuple[str, ...], list[StatisticalConstruction]] = (
        field(default_factory=dict)
    )
    observed_by_coptic: dict[str, list[ObservedWordformCandidate]] = field(
        default_factory=dict
    )
    heuristic_by_pair: dict[tuple[str, str], float] = field(default_factory=dict)
    loanwords_by_form: dict[str, list[VerifiedLoanword]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def includes_verified_loanwords(self) -> bool:
        return (
            self.condition is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
            and self.loanwords is not None
        )


def index_decode_model(model: DecodeModel) -> DecodeModel:
    model.constructions_by_id = {
        row.construction_id: row for row in model.constructions.constructions
    }
    by_pat: dict[tuple[str, ...], list[StatisticalConstruction]] = {}
    for row in model.constructions.constructions:
        key = tuple(row.coptic_pattern)
        by_pat.setdefault(key, []).append(row)
    for rows in by_pat.values():
        rows.sort(key=lambda r: (-r.observation_count, r.construction_id))
    model.constructions_by_pattern = by_pat

    by_cop: dict[str, list[ObservedWordformCandidate]] = {}
    for row in model.lexical.observed_wordforms:
        by_cop.setdefault(row.coptic_wordform, []).append(row)
    for rows in by_cop.values():
        rows.sort(
            key=lambda r: (
                -r.observation_count,
                -len(r.contributing_aligners),
                r.greek_wordform,
            )
        )
    model.observed_by_coptic = by_cop

    heur: dict[tuple[str, str], float] = {}
    for conf in model.lexical.confidence_estimates:
        if conf.applies_to_layer is not LexicalLayer.OBSERVED_WORDFORM:
            continue
        if conf.heuristic_ranking_score is None:
            continue
        c = conf.key.get("coptic_wordform")
        g = conf.key.get("greek_wordform")
        if c and g:
            heur[(c, g)] = float(conf.heuristic_ranking_score)
    model.heuristic_by_pair = heur

    loan_idx: dict[str, list[VerifiedLoanword]] = {}
    if model.loanwords is not None:
        for entry in model.loanwords.entries:
            forms = set(entry.coptic_forms)
            forms.add(entry.coptic_lemma)
            for form in forms:
                loan_idx.setdefault(form, []).append(entry)
        for rows in loan_idx.values():
            rows.sort(key=lambda e: e.loanword_id)
    model.loanwords_by_form = loan_idx
    return model


def load_ablation_package(path: Path) -> AblationPackage:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AblationPackage.model_validate(payload)


def resolve_ablation_path(
    repo_root: Path,
    condition: AblationCondition | str,
) -> Path:
    key = (
        condition.value if isinstance(condition, AblationCondition) else str(condition)
    )
    paths = default_ablation_paths(repo_root)
    if key not in paths:
        msg = f"unknown ablation condition: {key}"
        raise ValueError(msg)
    return paths[key]


def _resolve_rel(repo_root: Path, rel: str | None) -> Path | None:
    if rel is None:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prov_get(table_prov: dict[str, Any], key: str) -> object:
    return table_prov.get(key)


def assert_compatible_provenance(
    lexical: LexicalCandidateTable,
    constructions: StatisticalConstructionTable,
) -> dict[str, Any]:
    """Fail closed when JAK-37/38 provenance / split fingerprints disagree."""
    lex_p = lexical.provenance or {}
    con_p = constructions.provenance or {}
    shared: dict[str, Any] = {}
    for key in PROVENANCE_KEYS:
        lv = _prov_get(lex_p, key)
        cv = _prov_get(con_p, key)
        if lv is None and cv is None:
            msg = f"missing provenance key on both tables: {key}"
            raise ValueError(msg)
        if lv is None or cv is None:
            msg = (
                f"provenance key {key!r} missing on "
                f"{'lexical' if lv is None else 'constructions'} table"
            )
            raise ValueError(msg)
        if lv != cv:
            msg = (
                f"provenance mismatch for {key!r}: lexical={lv!r} constructions={cv!r}"
            )
            raise ValueError(msg)
        shared[key] = lv
    return shared


def validate_decode_evidence(
    *,
    repo_root: Path,
    lexical: LexicalCandidateTable,
    constructions: StatisticalConstructionTable,
    loanwords: VerifiedLoanwordTable | None,
    require_loanword_tei: bool,
) -> set[str]:
    """Run JAK-37/38 validators; return train unit ids used."""
    splits = load_canonical_splits(
        resolve_nt_split_path(repo_root),
        resolve_omission_fixtures_path(repo_root),
    )
    train_unit_ids = set(splits.unit_ids_for(SplitRole.TRAIN))

    lex_report = validate_lexical_candidate_table(
        lexical, train_unit_ids=train_unit_ids
    )
    if not lex_report.ok:
        issues = "; ".join(f"{i.code}:{i.message}" for i in lex_report.issues[:8])
        msg = f"lexical candidate validation failed: {issues}"
        raise ValueError(msg)

    phrase_report = validate_statistical_construction_table(
        constructions, train_unit_ids=train_unit_ids
    )
    if not phrase_report.ok:
        issues = "; ".join(f"{i.code}:{i.message}" for i in phrase_report.issues[:8])
        msg = f"statistical construction validation failed: {issues}"
        raise ValueError(msg)

    if loanwords is not None and require_loanword_tei:
        loan_report = validate_verified_loanword_table(
            loanwords, repo_root=repo_root, train_unit_ids=train_unit_ids
        )
        if not loan_report.ok:
            issues = "; ".join(f"{i.code}:{i.message}" for i in loan_report.issues[:8])
            msg = f"verified loanword validation failed: {issues}"
            raise ValueError(msg)
    return train_unit_ids


def load_decode_model(
    repo_root: Path,
    *,
    condition: AblationCondition | str,
    ablation_path: Path | None = None,
    lexical_path: Path | None = None,
    constructions_path: Path | None = None,
    loanwords_path: Path | None = None,
    skip_validation: bool = False,
) -> DecodeModel:
    """Load committed JAK-37/38 artifacts for one ablation arm.

    Fails closed on JAK-37/38 validation errors and provenance/split mismatches
    unless ``skip_validation`` is set (tests only).
    """
    cond = (
        condition
        if isinstance(condition, AblationCondition)
        else AblationCondition(condition)
    )
    package_path = ablation_path or resolve_ablation_path(repo_root, cond)
    package = load_ablation_package(package_path)
    if package.condition is not cond:
        msg = (
            f"ablation package condition {package.condition.value!r} "
            f"!= requested {cond.value!r}"
        )
        raise ValueError(msg)
    if list(package.phase2_active_aligners) != list(PHASE2_ALIGNERS):
        msg = (
            "decode model requires keep-both phase2_active_aligners="
            f"{list(PHASE2_ALIGNERS)}; got {package.phase2_active_aligners}"
        )
        raise ValueError(msg)
    if package.adopted_aligner is not None:
        msg = "decode model requires adopted_aligner=null (keep-both)"
        raise ValueError(msg)

    includes = package.includes
    lex_path = lexical_path or _resolve_rel(
        repo_root, includes.get("lexical_candidates")
    )
    if lex_path is None:
        lex_path = default_table_path(repo_root)
    phrase_path = constructions_path or _resolve_rel(
        repo_root, includes.get("statistical_phrase_constructions")
    )
    if phrase_path is None:
        phrase_path = default_statistical_constructions_path(repo_root)

    loan_rel = includes.get("verified_loanwords")
    loan_path = loanwords_path
    if loan_path is None and loan_rel:
        loan_path = _resolve_rel(repo_root, loan_rel)
    if (
        cond is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
        and loan_path is None
    ):
        loan_path = default_verified_loanwords_path(repo_root)
    # Arm A still indexes the verified table when present so abstention can
    # report ablation_excludes_loanwords — candidates are never emitted.
    if cond is AblationCondition.NT_ALIGNER_ONLY and loan_path is None:
        default_loan = default_verified_loanwords_path(repo_root)
        if default_loan.is_file():
            loan_path = default_loan

    lexical = load_lexical_candidate_table(lex_path)
    constructions = load_statistical_construction_table(phrase_path)
    loanwords: VerifiedLoanwordTable | None = None
    if loan_path is not None:
        if not loan_path.is_file():
            if cond is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS:
                msg = "arm B requires verified loanwords artifact"
                raise ValueError(msg)
        else:
            loanwords = load_verified_loanword_table(loan_path)
    elif cond is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS:
        msg = "arm B requires verified loanwords artifact"
        raise ValueError(msg)

    shared_prov = assert_compatible_provenance(lexical, constructions)
    if not skip_validation:
        validate_decode_evidence(
            repo_root=repo_root,
            lexical=lexical,
            constructions=constructions,
            loanwords=loanwords,
            require_loanword_tei=(
                cond is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
                or loanwords is not None
            ),
        )

    artifact_digests = {
        "lexical_sha256": _sha256_file(lex_path),
        "constructions_sha256": _sha256_file(phrase_path),
    }
    if loan_path is not None and loan_path.is_file():
        artifact_digests["loanwords_sha256"] = _sha256_file(loan_path)
    artifact_digests["ablation_sha256"] = _sha256_file(package_path)

    model = DecodeModel(
        condition=cond,
        package=package,
        lexical=lexical,
        constructions=constructions,
        loanwords=loanwords,
        provenance={
            "ablation_path": str(package_path),
            "lexical_path": str(lex_path),
            "constructions_path": str(phrase_path),
            "loanwords_path": str(loan_path) if loanwords is not None else None,
            "loanwords_emit_candidates": (
                cond is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
            ),
            "phase2_active_aligners": list(PHASE2_ALIGNERS),
            "adopted_aligner": None,
            "owned_by": "JAK-39",
            "artifact_digests": artifact_digests,
            "evidence_provenance": shared_prov,
            "lexical_table_id": lexical.id,
            "constructions_table_id": constructions.id,
            "loanwords_table_id": (loanwords.id if loanwords is not None else None),
            "validation": "jak37_jak38_fail_closed",
        },
    )
    return index_decode_model(model)


def normalized_loanword_greek(entry: VerifiedLoanword) -> str:
    """Prefer committed normalized lemma; else strip diacritics."""
    if entry.greek_lemma_normalized:
        return entry.greek_lemma_normalized
    return infer_greek_lemma(entry.greek_lemma)

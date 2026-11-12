# SPDX-License-Identifier: Apache-2.0
"""Validate JAK-38 statistical constructions, loanwords, and ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from zetodos_thomas.lexicon.candidates import FORBIDDEN_UNIT_MARKERS
from zetodos_thomas.schemas.ids import parse_parallel_id, parse_span_id, parse_token_id
from zetodos_thomas.schemas.mappings import (
    AblationCondition,
    AblationPackage,
    EvidenceClass,
    StatisticalConstructionTable,
    VerifiedLoanwordTable,
)


@dataclass
class MappingValidationIssue:
    """One structural / leakage / layering problem."""

    severity: str
    code: str
    message: str


@dataclass
class MappingValidationReport:
    """Offline validation result for JAK-38 mapping artifacts."""

    ok: bool
    artifact_id: str
    issues: list[MappingValidationIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "artifact_id": self.artifact_id,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message}
                for i in self.issues
            ],
        }


def _unit_forbidden(unit_id: str) -> bool:
    lowered = unit_id.lower().replace(" ", "")
    return any(marker in lowered for marker in FORBIDDEN_UNIT_MARKERS) or (
        "p.oxy" in lowered
    )


def validate_statistical_construction_table(
    table: StatisticalConstructionTable,
    *,
    train_unit_ids: set[str] | None = None,
) -> MappingValidationReport:
    """Check statistical labeling, train-only units, span IDs, dual aligners."""
    issues: list[MappingValidationIssue] = []
    if table.evidence_class is not EvidenceClass.STATISTICAL:
        issues.append(
            MappingValidationIssue(
                "error",
                "table_evidence_class",
                "table must be evidence_class=statistical",
            )
        )
    aligners = table.summary.get("phase2_active_aligners") or table.provenance.get(
        "phase2_active_aligners"
    )
    if list(aligners or []) != ["eflomal", "fast_align"]:
        issues.append(
            MappingValidationIssue(
                "error",
                "keep_both_aligners",
                f"phase2_active_aligners must be eflomal+fast_align; got {aligners}",
            )
        )
    if table.provenance.get("adopted_aligner") not in (None,):
        # summary may also carry it
        pass
    if table.summary.get("adopted_aligner") is not None:
        issues.append(
            MappingValidationIssue(
                "error",
                "adopted_aligner_must_be_null",
                "adopted_aligner must remain null while both aligners are active",
            )
        )

    seen_ids: set[str] = set()
    for row in table.constructions:
        if row.evidence_class is not EvidenceClass.STATISTICAL:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "row_not_statistical",
                    f"{row.construction_id} must be statistical",
                )
            )
        if row.construction_id in seen_ids:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "duplicate_construction_id",
                    row.construction_id,
                )
            )
        seen_ids.add(row.construction_id)
        if set(row.contributing_aligners) != {"eflomal", "fast_align"}:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "construction_missing_aligner",
                    f"{row.construction_id} aligners={row.contributing_aligners}",
                )
            )
        for unit_id in row.source_unit_ids:
            if _unit_forbidden(unit_id):
                issues.append(
                    MappingValidationIssue(
                        "error",
                        "leakage_unit",
                        f"forbidden unit {unit_id} in {row.construction_id}",
                    )
                )
            elif train_unit_ids is not None and unit_id not in train_unit_ids:
                issues.append(
                    MappingValidationIssue(
                        "error",
                        "non_train_unit",
                        f"unit {unit_id} not in train membership "
                        f"({row.construction_id})",
                    )
                )
        for ex in row.examples:
            if _unit_forbidden(ex.unit_id):
                issues.append(
                    MappingValidationIssue(
                        "error",
                        "example_leakage",
                        f"forbidden example unit {ex.unit_id}",
                    )
                )
            try:
                parse_parallel_id(ex.parallel_id)
            except ValueError as exc:
                issues.append(
                    MappingValidationIssue("error", "bad_parallel_id", str(exc))
                )
            for span in (ex.coptic_span, ex.greek_span):
                try:
                    parsed = parse_span_id(span.span_id)
                except ValueError as exc:
                    issues.append(
                        MappingValidationIssue("error", "bad_span_id", str(exc))
                    )
                    continue
                if parsed.source_id != span.source_id or parsed.unit_id != span.unit_id:
                    issues.append(
                        MappingValidationIssue(
                            "error",
                            "span_id_field_mismatch",
                            f"{span.span_id} vs fields",
                        )
                    )
                for tid in span.token_ids:
                    try:
                        tok = parse_token_id(tid)
                    except ValueError as exc:
                        issues.append(
                            MappingValidationIssue("error", "bad_token_id", str(exc))
                        )
                        continue
                    if tok.unit_id != span.unit_id:
                        issues.append(
                            MappingValidationIssue(
                                "error",
                                "token_unit_mismatch",
                                f"{tid} vs span unit {span.unit_id}",
                            )
                        )

    errors = [i for i in issues if i.severity == "error"]
    return MappingValidationReport(
        ok=not errors,
        artifact_id=table.id,
        issues=issues,
    )


def validate_verified_loanword_table(
    table: VerifiedLoanwordTable,
) -> MappingValidationReport:
    """Check independently_verified labeling, provenance, and spot-checks."""
    issues: list[MappingValidationIssue] = []
    if table.evidence_class is not EvidenceClass.INDEPENDENTLY_VERIFIED:
        issues.append(
            MappingValidationIssue(
                "error",
                "table_evidence_class",
                "loanword table must be independently_verified",
            )
        )
    if table.license != "CC-BY-SA-4.0":
        issues.append(
            MappingValidationIssue(
                "error",
                "license",
                f"expected CC-BY-SA-4.0, got {table.license}",
            )
        )
    required_prov = (
        "ccl_tei_sha256",
        "doi",
        "source_manifest_id",
        "attribution",
    )
    for key in required_prov:
        if not table.provenance.get(key):
            issues.append(
                MappingValidationIssue(
                    "error",
                    "missing_provenance",
                    f"provenance.{key} required",
                )
            )
    seen: set[str] = set()
    for entry in table.entries:
        if entry.evidence_class is not EvidenceClass.INDEPENDENTLY_VERIFIED:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "row_not_verified",
                    entry.loanword_id,
                )
            )
        if entry.loanword_id in seen:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "duplicate_loanword_id",
                    entry.loanword_id,
                )
            )
        seen.add(entry.loanword_id)
        if not entry.tla_entry_id.startswith("C"):
            issues.append(
                MappingValidationIssue(
                    "error",
                    "tla_id",
                    f"{entry.loanword_id} missing TLA C-id",
                )
            )
        if not entry.ddglc_greek_lemma_id:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "ddglc_id",
                    f"{entry.loanword_id} missing DDGLC greek lemma id",
                )
            )
        if entry.spot_check.result != "pass":
            issues.append(
                MappingValidationIssue(
                    "error",
                    "spot_check_failed",
                    entry.loanword_id,
                )
            )
        # No Thomas/P.Oxy Greek in illustrative spans.
        for sid in entry.illustrative_nt_span_ids:
            lowered = sid.lower()
            if "thomas." in lowered or "poxy" in lowered:
                issues.append(
                    MappingValidationIssue(
                        "error",
                        "illustrative_leakage",
                        f"{entry.loanword_id} span {sid}",
                    )
                )
            else:
                try:
                    parse_span_id(sid)
                except ValueError as exc:
                    issues.append(
                        MappingValidationIssue(
                            "error", "bad_illustrative_span", str(exc)
                        )
                    )

    errors = [i for i in issues if i.severity == "error"]
    return MappingValidationReport(
        ok=not errors,
        artifact_id=table.id,
        issues=issues,
    )


def validate_ablation_package(package: AblationPackage) -> MappingValidationReport:
    """Check ablation arm wiring and keep-both aligners."""
    issues: list[MappingValidationIssue] = []
    if package.phase2_active_aligners != ["eflomal", "fast_align"]:
        issues.append(
            MappingValidationIssue(
                "error",
                "keep_both_aligners",
                str(package.phase2_active_aligners),
            )
        )
    if package.adopted_aligner is not None:
        issues.append(
            MappingValidationIssue(
                "error",
                "adopted_aligner_must_be_null",
                str(package.adopted_aligner),
            )
        )
    includes = package.includes
    if not includes.get("lexical_candidates"):
        issues.append(
            MappingValidationIssue("error", "missing_lexical_candidates", package.id)
        )
    if not includes.get("statistical_phrase_constructions"):
        issues.append(
            MappingValidationIssue("error", "missing_statistical_phrases", package.id)
        )
    verified = includes.get("verified_loanwords")
    if package.condition is AblationCondition.NT_ALIGNER_ONLY:
        if verified is not None:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "aligner_only_must_omit_loanwords",
                    str(verified),
                )
            )
    elif package.condition is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS:
        if not verified:
            issues.append(
                MappingValidationIssue(
                    "error",
                    "plus_loanwords_requires_verified_table",
                    package.id,
                )
            )
    errors = [i for i in issues if i.severity == "error"]
    return MappingValidationReport(
        ok=not errors,
        artifact_id=package.id,
        issues=issues,
    )


def validate_mappings_dir(
    repo_root: Path,
    *,
    train_unit_ids: set[str] | None = None,
) -> MappingValidationReport:
    """Validate all committed JAK-38 mapping artifacts under mappings/."""
    from zetodos_thomas.mappings.ablation import default_ablation_paths
    from zetodos_thomas.mappings.loanwords import (
        default_verified_loanwords_path,
        load_verified_loanword_table,
    )
    from zetodos_thomas.mappings.phrases import (
        default_statistical_constructions_path,
        load_statistical_construction_table,
    )

    issues: list[MappingValidationIssue] = []
    ok = True

    phrases_path = default_statistical_constructions_path(repo_root)
    loan_path = default_verified_loanwords_path(repo_root)
    if not phrases_path.is_file():
        issues.append(
            MappingValidationIssue("error", "missing_file", str(phrases_path))
        )
        ok = False
    else:
        report = validate_statistical_construction_table(
            load_statistical_construction_table(phrases_path),
            train_unit_ids=train_unit_ids,
        )
        issues.extend(report.issues)
        ok = ok and report.ok

    if not loan_path.is_file():
        issues.append(MappingValidationIssue("error", "missing_file", str(loan_path)))
        ok = False
    else:
        report = validate_verified_loanword_table(
            load_verified_loanword_table(loan_path)
        )
        issues.extend(report.issues)
        ok = ok and report.ok

    for path in default_ablation_paths(repo_root).values():
        if not path.is_file():
            issues.append(MappingValidationIssue("error", "missing_file", str(path)))
            ok = False
            continue
        payload = path.read_text(encoding="utf-8")
        import json

        package = AblationPackage.model_validate(json.loads(payload))
        report = validate_ablation_package(package)
        issues.extend(report.issues)
        ok = ok and report.ok

    return MappingValidationReport(
        ok=ok,
        artifact_id="jak-38-mappings",
        issues=issues,
    )

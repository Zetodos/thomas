# SPDX-License-Identifier: Apache-2.0
"""Validate Coptic→Greek lexical candidate tables (JAK-37)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from zetodos_thomas.lexicon.candidates import load_lexical_candidate_table
from zetodos_thomas.schemas.lexicon import (
    LexicalCandidateTable,
    LexicalLayer,
    SplitMembership,
)

FORBIDDEN_MARKERS = ("thomas.", "poxy", "p.oxy")


@dataclass
class LexicalValidationIssue:
    """One structural / leakage / layering problem."""

    severity: str  # error | warning
    code: str
    message: str


@dataclass
class LexicalValidationReport:
    """Offline validation result for a lexical candidate table."""

    ok: bool
    table_id: str
    observed_count: int
    inferred_count: int
    confidence_count: int
    issues: list[LexicalValidationIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "table_id": self.table_id,
            "observed_count": self.observed_count,
            "inferred_count": self.inferred_count,
            "confidence_count": self.confidence_count,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message}
                for i in self.issues
            ],
        }


def _unit_forbidden(unit_id: str) -> bool:
    lowered = unit_id.lower().replace(" ", "")
    return any(marker in lowered for marker in FORBIDDEN_MARKERS)


def _check_unit_membership(
    unit_id: str,
    *,
    train_unit_ids: set[str] | None,
    issues: list[LexicalValidationIssue],
    code_prefix: str,
    context: str,
) -> None:
    if _unit_forbidden(unit_id):
        issues.append(
            LexicalValidationIssue(
                "error",
                f"{code_prefix}_leakage",
                f"forbidden unit_id {unit_id} in {context}",
            )
        )
        return
    if train_unit_ids is None:
        return
    if unit_id not in train_unit_ids:
        issues.append(
            LexicalValidationIssue(
                "error",
                f"{code_prefix}_non_train_or_unknown",
                f"unit_id {unit_id} is not in authoritative train membership "
                f"({context})",
            )
        )


def validate_lexical_candidate_table(
    table: LexicalCandidateTable,
    *,
    train_unit_ids: set[str] | None = None,
) -> LexicalValidationReport:
    """Check layering, train-only membership, non-fabrication, no leakage.

    When ``train_unit_ids`` is provided (canonical split train membership),
    every source/example unit must be a known train id — labeled ``train`` alone
    is not enough (blocks holdout/dev/omission leakage).
    """
    issues: list[LexicalValidationIssue] = []

    if table.owned_by != "JAK-37":
        issues.append(
            LexicalValidationIssue(
                "warning",
                "owned_by",
                f"expected owned_by=JAK-37, got {table.owned_by!r}",
            )
        )

    # Layer catalogs must match list lengths.
    layer_meta = table.layers or {}
    for key, rows, layer_name in (
        ("observed_wordform", table.observed_wordforms, "observed_wordforms"),
        ("inferred_lemma", table.inferred_lemmas, "inferred_lemmas"),
        ("confidence_estimate", table.confidence_estimates, "confidence_estimates"),
    ):
        meta = layer_meta.get(key) or {}
        declared = meta.get("entry_count")
        if declared is not None and declared != len(rows):
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "layer_count_mismatch",
                    (
                        f"layers[{key!r}].entry_count={declared} != "
                        f"len({layer_name})={len(rows)}"
                    ),
                )
            )

    for row in table.observed_wordforms:
        if row.layer is not LexicalLayer.OBSERVED_WORDFORM:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "layer_collapse",
                    "observed_wordforms entry has wrong layer tag",
                )
            )
        if row.split_membership is not SplitMembership.TRAIN:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "non_train_observation",
                    f"wordform {row.coptic_wordform!r}↔{row.greek_wordform!r} "
                    f"split_membership={row.split_membership!r}",
                )
            )
        if row.observation_count <= 0:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "fabricated_or_empty_count",
                    f"non-positive observation_count for "
                    f"{row.coptic_wordform!r}↔{row.greek_wordform!r}",
                )
            )
        if not row.per_aligner_observation_counts:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "missing_aligner_provenance",
                    f"no per_aligner counts for "
                    f"{row.coptic_wordform!r}↔{row.greek_wordform!r}",
                )
            )
        else:
            max_pa = max(row.per_aligner_observation_counts.values())
            sum_pa = sum(row.per_aligner_observation_counts.values())
            # Union of (unit, c_tid, g_tid) across aligners: max ≤ |U| ≤ sum.
            if not (max_pa <= row.observation_count <= sum_pa):
                issues.append(
                    LexicalValidationIssue(
                        "error",
                        "observation_count_inconsistent",
                        f"observation_count {row.observation_count} outside "
                        f"union bounds [max={max_pa}, sum={sum_pa}] for "
                        f"{row.coptic_wordform!r}↔{row.greek_wordform!r}",
                    )
                )
            if row.observation_count < len(row.source_unit_ids):
                issues.append(
                    LexicalValidationIssue(
                        "error",
                        "observation_count_below_unit_evidence",
                        f"observation_count {row.observation_count} < "
                        f"|source_unit_ids|={len(row.source_unit_ids)} for "
                        f"{row.coptic_wordform!r}↔{row.greek_wordform!r}",
                    )
                )
        ctx = f"observed wordform {row.coptic_wordform!r}↔{row.greek_wordform!r}"
        for unit_id in row.source_unit_ids:
            _check_unit_membership(
                unit_id,
                train_unit_ids=train_unit_ids,
                issues=issues,
                code_prefix="holdout",
                context=ctx,
            )
        for ex in row.examples:
            if ex.split_role is not SplitMembership.TRAIN:
                issues.append(
                    LexicalValidationIssue(
                        "error",
                        "example_non_train",
                        f"example unit {ex.unit_id} not train",
                    )
                )
            _check_unit_membership(
                ex.unit_id,
                train_unit_ids=train_unit_ids,
                issues=issues,
                code_prefix="example",
                context=f"example in {ctx}",
            )

    for row in table.inferred_lemmas:
        if row.layer is not LexicalLayer.INFERRED_LEMMA:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "layer_collapse",
                    "inferred_lemmas entry has wrong layer tag",
                )
            )
        if "coptic" not in row.inference_method or "greek" not in row.inference_method:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "missing_inference_method",
                    f"lemma {row.coptic_lemma!r}↔{row.greek_lemma!r} "
                    "missing inference_method labels",
                )
            )
        if row.split_membership is not SplitMembership.TRAIN:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "non_train_lemma",
                    f"lemma split_membership={row.split_membership!r}",
                )
            )
        for unit_id in row.source_unit_ids:
            _check_unit_membership(
                unit_id,
                train_unit_ids=train_unit_ids,
                issues=issues,
                code_prefix="lemma",
                context=(f"inferred lemma {row.coptic_lemma!r}↔{row.greek_lemma!r}"),
            )

    for row in table.confidence_estimates:
        if row.layer is not LexicalLayer.CONFIDENCE_ESTIMATE:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "layer_collapse",
                    "confidence_estimates entry has wrong layer tag",
                )
            )
        if row.applies_to_layer is LexicalLayer.CONFIDENCE_ESTIMATE:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "confidence_self_key",
                    "confidence row must apply to wordform or lemma layer",
                )
            )
        # Calibrated confidence may be null (insufficient / uncalibrated) — OK.
        # Fabrication would be inventing observation_count in evidence.
        obs = row.evidence.get("observation_count")
        if obs is not None and not isinstance(obs, int):
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "bad_evidence_count",
                    f"confidence evidence observation_count must be int, got {obs!r}",
                )
            )
        if isinstance(obs, int) and obs < 0:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "negative_evidence_count",
                    "confidence evidence observation_count must be >= 0",
                )
            )
        # Sparse / single-aligner evidence cannot carry a calibrated confidence.
        aligners = row.evidence.get("contributing_aligners") or []
        if (
            row.confidence is not None
            and isinstance(obs, int)
            and (obs < 3 or len(aligners) < 2)
        ):
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "uncalibrated_confidence",
                    "confidence must be null when evidence is sparse/noisy "
                    f"(observation_count={obs}, aligners={aligners!r})",
                )
            )

    # Dual-aligner keep-both: declared phase2 set must match both aligners and
    # the aligners that actually contributed observations (reject mismatch).
    summary = table.summary or {}
    expected_aligners = {"eflomal", "fast_align"}
    active = summary.get("phase2_active_aligners")
    if active is not None:
        if set(active) != expected_aligners:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "phase2_aligners",
                    f"keep-both requires both eflomal and fast_align, got {active!r}",
                )
            )
    actual_aligners = {
        name for row in table.observed_wordforms for name in row.contributing_aligners
    }
    if active is not None and actual_aligners and set(active) != actual_aligners:
        # Declared set must not silently omit or invent an aligner relative to data.
        # Partial contribution (one aligner saw fewer pairs) is fine; the *set* of
        # names present in provenance must equal the declared keep-both set when
        # both appear anywhere in the table. If only one name appears at all, that
        # is a declared-vs-actual mismatch under keep-both.
        if actual_aligners != expected_aligners:
            issues.append(
                LexicalValidationIssue(
                    "error",
                    "phase2_aligner_mismatch",
                    "declared phase2_active_aligners "
                    f"{active!r} != actual contributing aligners "
                    f"{sorted(actual_aligners)!r}",
                )
            )
    if summary.get("adopted_aligner") not in (None,):
        issues.append(
            LexicalValidationIssue(
                "error",
                "adopted_aligner",
                f"Jake keep-both expects adopted_aligner=null, got "
                f"{summary.get('adopted_aligner')!r}",
            )
        )

    errors = [i for i in issues if i.severity == "error"]
    return LexicalValidationReport(
        ok=not errors,
        table_id=table.id,
        observed_count=len(table.observed_wordforms),
        inferred_count=len(table.inferred_lemmas),
        confidence_count=len(table.confidence_estimates),
        issues=issues,
    )


def validate_lexical_candidate_path(
    path: Path,
    *,
    train_unit_ids: set[str] | None = None,
) -> LexicalValidationReport:
    """Load JSON from disk and validate."""
    table = load_lexical_candidate_table(path)
    return validate_lexical_candidate_table(table, train_unit_ids=train_unit_ids)

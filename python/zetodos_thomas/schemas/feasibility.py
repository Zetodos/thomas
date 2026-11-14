# SPDX-License-Identifier: Apache-2.0
"""Phase 2 feasibility metrics + decision-gate schemas (JAK-41).

No invented numeric pass threshold. Gate recommendation is advisory input for
Jake (Proceed / Pause and improve / Reassess). Evaluation-ready is separate
from the qualitative recommendation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zetodos_thomas.schemas.mappings import AblationCondition


class GateRecommendation(StrEnum):
    """Phase 3 decision-gate options (Jake chooses; not auto-started)."""

    PROCEED = "proceed"
    PAUSE_AND_IMPROVE = "pause_and_improve"
    REASSESS = "reassess"


class SystemLabel(StrEnum):
    """Reconstruction system under comparison."""

    DECODER = "decoder"
    WORD_FOR_WORD = "word_for_word"


class RatioMetric(BaseModel):
    """Explicit numerator / denominator ratio (null when denominator is 0)."""

    numerator: int
    denominator: int
    value: float | None
    definition: str = ""


class UnitSystemMetrics(BaseModel):
    """Multi-level metrics for one unit x one system."""

    unit_id: str
    system: SystemLabel
    letter_true_positives: int
    letter_predicted_denominator: int
    letter_gold_denominator: int
    letter_precision: float | None
    letter_recall: float | None
    letter_f1: float | None
    incidental_global_lcs: int = 0
    letter_metric_kind: str = "token-edit-aligned-letter-v1"
    surface_exact: RatioMetric
    lemma_proxy: RatioMetric
    lemma_lcs: RatioMetric
    near_miss_tokens: int = 0
    predicted_tokens: int
    gold_tokens: int
    additions: int
    omissions: int
    substitutions: int = 0
    coptic_surfaces: int
    predicted_non_null_spans: int
    predicted_covered_tokens: int = 0
    coverage: float | None = Field(
        default=None,
        description=(
            "Source-token coverage (non-abstained input tokens / Coptic tokens)."
        ),
    )
    span_coverage: float | None = None
    token_coverage: float | None = None
    abstention_rate: float | None = None
    span_abstention_rate: float | None = None
    mean_top1_uncertainty: float | None = None
    construction_spans: int = 0
    construction_phrase_hits: int = 0
    phrase_fidelity_available: bool = False
    lemma_inflection_accuracy_available: bool = False
    alignment_protocol: str = "token-edit-alignment-v1"
    notes: list[str] = Field(default_factory=list)


class AggregateSystemMetrics(BaseModel):
    """Macro averages + pooled denominators across units."""

    system: SystemLabel
    n_units: int
    mean_letter_precision: float | None
    mean_letter_recall: float | None
    mean_letter_f1: float | None
    mean_surface_exact_recall: float | None
    mean_lemma_proxy_recall: float | None
    mean_lemma_lcs_recall: float | None
    mean_coverage: float | None
    mean_abstention_rate: float | None = None
    mean_span_coverage: float | None = None
    mean_token_coverage: float | None = None
    pooled_letter: RatioMetric | None = None
    pooled_surface_exact: RatioMetric | None = None
    pooled_lemma_proxy: RatioMetric | None = None
    total_additions: int = 0
    total_omissions: int = 0
    total_substitutions: int = 0
    total_construction_spans: int = 0
    total_construction_phrase_hits: int = 0
    letter_f1_unit_wins_vs_other: int | None = None
    notes: list[str] = Field(default_factory=list)


class AblationArmComparison(BaseModel):
    """Decoder metrics for one JAK-38 ablation arm (NT holdouts)."""

    ablation_condition: AblationCondition
    aggregate: AggregateSystemMetrics | None = None
    vs_word_for_word_mean_letter_f1_delta: float | None = None
    informative_for_loanword_decision: bool = False
    evidence_status: str = "uninformative_regenerated_post_gold"
    loanword_hypothesis_changes: int | None = None
    thomas7_blinded_coverage: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class FailureModeExample(BaseModel):
    """Concrete failure-mode instance from evaluated reconstructions."""

    category: str
    unit_id: str
    system: SystemLabel | None = None
    detail: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class Thomas7GateStatus(BaseModel):
    """Honest human-checked / official surviving-letter gate status."""

    eval_package_id: str
    review_status: str
    official_surviving_letter_scores: bool
    gate_blocked: bool
    freeze_present: bool
    freeze_verified: bool = False
    human_checked_required: bool = True
    notes: list[str] = Field(default_factory=list)


class FeasibilityReport(BaseModel):
    """Auditable Phase 2 feasibility report package (JAK-41)."""

    id: str = "jak41-feasibility-metrics-v1"
    title: str = "Phase 2 feasibility metrics vs word-for-word baseline (JAK-41)"
    owned_by: str = "JAK-41"
    schema_version: str = "2"
    gate_recommendation: GateRecommendation
    gate_rationale: list[str] = Field(default_factory=list)
    evaluation_ready: bool = False
    evaluation_ready_notes: list[str] = Field(default_factory=list)
    thomas7_gate: Thomas7GateStatus
    nt_holdout_review_status: str = "provisional-draft"
    primary_ablation: AblationCondition
    decoder_aggregate: AggregateSystemMetrics
    word_for_word_aggregate: AggregateSystemMetrics
    ablation_arms: list[AblationArmComparison] = Field(default_factory=list)
    unit_metrics: list[UnitSystemMetrics] = Field(default_factory=list)
    failure_modes: list[FailureModeExample] = Field(default_factory=list)
    metric_definitions: dict[str, str] = Field(default_factory=dict)
    identical_inputs: dict[str, Any] = Field(default_factory=dict)
    leakage_safeguards: dict[str, Any] = Field(default_factory=dict)
    artifact_digests: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

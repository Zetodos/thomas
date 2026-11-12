# SPDX-License-Identifier: Apache-2.0
"""Coptic→Greek lexical candidate table schemas (JAK-37).

Three layers are first-class and must not be collapsed:

* ``observed_wordform`` — attested surface forms from train alignments
* ``inferred_lemma`` — derived/normalized lemma proxies (never claimed as morph
  analyzer output unless method says so)
* ``confidence_estimate`` — separate uncertainty scores over a keyed entry

Observation counts are never fabricated: missing evidence stays missing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class LexicalLayer(StrEnum):
    """Which consumer-facing layer an entry belongs to."""

    OBSERVED_WORDFORM = "observed_wordform"
    INFERRED_LEMMA = "inferred_lemma"
    CONFIDENCE_ESTIMATE = "confidence_estimate"


class SplitMembership(StrEnum):
    """Split role that fed observations (train-only for JAK-37 table)."""

    TRAIN = "train"


class ObservationExample(BaseModel):
    """One inspectable attestation of a Coptic↔Greek link."""

    unit_id: str
    split_role: SplitMembership = SplitMembership.TRAIN
    coptic_token_id: str
    greek_token_id: str
    coptic_wordform: str
    greek_wordform: str
    aligners: list[str] = Field(min_length=1)


class ObservedWordformCandidate(BaseModel):
    """Attested Coptic↔Greek surface pair aggregated from train alignments."""

    layer: LexicalLayer = LexicalLayer.OBSERVED_WORDFORM
    coptic_wordform: str
    greek_wordform: str
    observation_count: int = Field(ge=0)
    per_aligner_observation_counts: dict[str, int] = Field(default_factory=dict)
    contributing_aligners: list[str] = Field(min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    split_membership: SplitMembership = SplitMembership.TRAIN
    examples: list[ObservationExample] = Field(default_factory=list)
    dice_cooccurrence: float | None = Field(
        default=None,
        description=(
            "Optional JAK-28 Dice from the same train units (secondary metric; "
            "not the elementary baseline redo)."
        ),
    )

    @model_validator(mode="after")
    def _counts_consistent(self) -> Self:
        if self.observation_count < 0:
            msg = "observation_count must be >= 0"
            raise ValueError(msg)
        for name, count in self.per_aligner_observation_counts.items():
            if count < 0:
                msg = f"per_aligner count for {name} must be >= 0"
                raise ValueError(msg)
        if self.contributing_aligners:
            missing = [
                a
                for a in self.contributing_aligners
                if a not in self.per_aligner_observation_counts
            ]
            if missing:
                msg = f"contributing_aligners missing counts: {missing}"
                raise ValueError(msg)
        return self


class InferredLemmaCandidate(BaseModel):
    """Lemma-proxy candidate derived from observed wordforms (separate layer)."""

    layer: LexicalLayer = LexicalLayer.INFERRED_LEMMA
    coptic_lemma: str
    greek_lemma: str
    inference_method: dict[str, str]
    observation_count: int = Field(
        ge=0,
        description="Sum of underlying wordform observation counts (never invented).",
    )
    underlying_wordform_pairs: list[dict[str, str | int]] = Field(
        default_factory=list,
        description="Observed wordform pairs rolled into this lemma proxy.",
    )
    contributing_aligners: list[str] = Field(min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    split_membership: SplitMembership = SplitMembership.TRAIN
    per_aligner_observation_counts: dict[str, int] = Field(default_factory=dict)


class ConfidenceComponents(BaseModel):
    """Explicit score parts; null means unavailable (not fabricated)."""

    relative_frequency_among_coptic_candidates: float | None = None
    aligner_agreement: float | None = None
    observation_support: float | None = None
    dice_baseline: float | None = None
    notes: str = ""


class ConfidenceEstimate(BaseModel):
    """Confidence/uncertainty for one keyed candidate (separate layer).

    ``heuristic_ranking_score`` is an uncalibrated ranking aid only.
    ``confidence`` is reserved for calibrated correctness estimates and stays
    null unless an independently justified calibration rule applies.
    """

    layer: LexicalLayer = LexicalLayer.CONFIDENCE_ESTIMATE
    applies_to_layer: LexicalLayer
    key: dict[str, str] = Field(
        description="Identity keys for the target layer entry (wordforms or lemmas).",
    )
    method: str
    heuristic_ranking_score: float | None = Field(
        default=None,
        description=(
            "Uncalibrated ranking heuristic from available components. "
            "Not a probability of a correct Coptic→Greek mapping."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Calibrated correctness confidence. Null when evidence is insufficient "
            "or no independent calibration rule is adopted."
        ),
    )
    uncertainty: float | None = None
    components: ConfidenceComponents = Field(default_factory=ConfidenceComponents)
    evidence: dict[str, Any] = Field(default_factory=dict)


class LexicalCandidateTable(BaseModel):
    """Inspectable Coptic→Greek lexical candidate export (JAK-37)."""

    id: str
    title: str
    owned_by: str = "JAK-37"
    schema_version: str = "1"
    layers: dict[str, Any]
    observed_wordforms: list[ObservedWordformCandidate]
    inferred_lemmas: list[InferredLemmaCandidate]
    confidence_estimates: list[ConfidenceEstimate]
    provenance: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _layers_labeled(self) -> Self:
        for row in self.observed_wordforms:
            if row.layer is not LexicalLayer.OBSERVED_WORDFORM:
                msg = "observed_wordforms entries must use layer=observed_wordform"
                raise ValueError(msg)
        for row in self.inferred_lemmas:
            if row.layer is not LexicalLayer.INFERRED_LEMMA:
                msg = "inferred_lemmas entries must use layer=inferred_lemma"
                raise ValueError(msg)
        for row in self.confidence_estimates:
            if row.layer is not LexicalLayer.CONFIDENCE_ESTIMATE:
                msg = "confidence_estimates entries must use layer=confidence_estimate"
                raise ValueError(msg)
        return self

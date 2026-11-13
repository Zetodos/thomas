# SPDX-License-Identifier: Apache-2.0
"""Blinded reconstruction + surviving-letter eval schemas (JAK-40).

Reconstruction artifacts must be producible without loading P.Oxy. / fragment
Greek. Surviving-letter gold is resolved only at scoring time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zetodos_thomas.schemas.decode import DecodeResult
from zetodos_thomas.schemas.mappings import AblationCondition


class BlindedTargetKind(StrEnum):
    """What kind of Coptic input is being reconstructed."""

    THOMAS_PILOT = "thomas_pilot"
    NT_HOLDOUT = "nt_holdout"


class BlindedUnitReconstruction(BaseModel):
    """One unit's blinded decode (Coptic in → Greek hypotheses / abstentions)."""

    unit_id: str
    target_kind: BlindedTargetKind
    source_id: str
    coptic_surfaces: list[str]
    decode_result: DecodeResult
    top1_greek_surfaces: list[str | None] = Field(
        default_factory=list,
        description="Per-span top-1 Greek surface sequence, or null when abstained.",
    )
    notes: list[str] = Field(default_factory=list)


class WordForWordUnitBaseline(BaseModel):
    """Elementary train-only lexical argmax baseline (no target Greek)."""

    unit_id: str
    target_kind: BlindedTargetKind
    source_id: str
    coptic_surfaces: list[str]
    predicted_greek: list[str | None]
    method: str = "train-cooccurrence-argmax-v1"
    notes: list[str] = Field(default_factory=list)


class BlindedReconstructionPackage(BaseModel):
    """Versioned package of blinded reconstructions (no evaluation Greek)."""

    id: str
    title: str
    owned_by: str = "JAK-40"
    schema_version: str = "1"
    ablation_condition: AblationCondition
    ranking_rules_id: str
    phase2_active_aligners: list[str] = Field(
        default_factory=lambda: ["eflomal", "fast_align"]
    )
    adopted_aligner: str | None = None
    units: list[BlindedUnitReconstruction]
    word_for_word_baseline: list[WordForWordUnitBaseline] = Field(default_factory=list)
    artifact_digests: dict[str, str] = Field(default_factory=dict)
    evidence_provenance: dict[str, Any] = Field(default_factory=dict)
    blinding: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SurvivingLetterSpan(BaseModel):
    """Surviving letters from one fragment line (restorations excluded)."""

    line_id: str
    n: str
    surviving_text: str
    surviving_letters: str
    restoration_text: str = ""
    lacuna_count: int = 0


class SurvivingLetterGold(BaseModel):
    """Gold surviving-letter string resolved at scoring time only."""

    transcription_id: str
    manuscript_id: str
    source_id: str
    line_ids: list[str]
    spans: list[SurvivingLetterSpan]
    surviving_text: str
    surviving_letters: str
    restoration_letters_excluded: str
    notes: list[str] = Field(default_factory=list)


class SurvivingLetterScore(BaseModel):
    """Letter-level agreement vs surviving gold (editorial restorations excluded)."""

    unit_id: str
    official: bool
    hypothesis_letters: str
    gold_surviving_letters: str
    true_positives: int
    predicted_denominator: int
    gold_denominator: int
    precision: float | None
    recall: float | None
    f1: float | None
    longest_common_subsequence: int
    review_status: str
    notes: list[str] = Field(default_factory=list)


class SurvivingLetterEvalReport(BaseModel):
    """Auditable surviving-letter evaluation report (JAK-40)."""

    id: str
    title: str
    owned_by: str = "JAK-40"
    schema_version: str = "1"
    reconstruction_package_id: str
    eval_package_id: str
    review_status: str
    official: bool
    gold: SurvivingLetterGold | None = None
    scores: list[SurvivingLetterScore] = Field(default_factory=list)
    nt_holdout_letter_scores: list[SurvivingLetterScore] = Field(default_factory=list)
    artifact_digests: dict[str, str] = Field(default_factory=dict)
    blinding: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

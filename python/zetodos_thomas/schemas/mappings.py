# SPDX-License-Identifier: Apache-2.0
"""Phrase/construction and verified-loanword mapping schemas (JAK-38).

Statistical correspondences (train-split aligner observations) and independently
verified lexical evidence (CCL/DDGLC) are first-class separate classes. Do not
collapse aligner-derived pairs into verified loanword facts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.schemas.lexicon import SplitMembership


class EvidenceClass(StrEnum):
    """Provenance class for a mapping row (Jake / JAK-38 separation)."""

    STATISTICAL = "statistical"
    INDEPENDENTLY_VERIFIED = "independently_verified"


class ConstructionFamily(StrEnum):
    """Coarse construction families prioritized for Phase 2."""

    NEGATION = "negation"
    ARTICLE = "article"
    PRONOUN = "pronoun"
    VERBAL = "verbal"
    OTHER = "other"


class SpanRef(BaseModel):
    """Stable span handle (JAK-32) with ordered token coverage."""

    span_id: str
    source_id: str
    unit_id: str
    token_ids: list[str] = Field(min_length=1)
    surfaces: list[str] = Field(default_factory=list)


class ConstructionExample(BaseModel):
    """One train-split attestation of a statistical construction."""

    unit_id: str
    split_role: SplitMembership = SplitMembership.TRAIN
    aligners: list[str] = Field(min_length=1)
    coptic_span: SpanRef
    greek_span: SpanRef
    parallel_id: str


class StatisticalConstruction(BaseModel):
    """Recurring multiword / construction pattern from train alignments."""

    evidence_class: EvidenceClass = EvidenceClass.STATISTICAL
    construction_id: str
    family: ConstructionFamily
    coptic_pattern: list[str] = Field(min_length=1)
    greek_pattern: list[str] = Field(min_length=1)
    observation_count: int = Field(ge=1)
    per_aligner_observation_counts: dict[str, int] = Field(default_factory=dict)
    contributing_aligners: list[str] = Field(min_length=1)
    source_unit_ids: list[str] = Field(default_factory=list)
    split_membership: SplitMembership = SplitMembership.TRAIN
    examples: list[ConstructionExample] = Field(default_factory=list)
    pilot_priority_notes: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _statistical_only(self) -> Self:
        if self.evidence_class is not EvidenceClass.STATISTICAL:
            msg = "StatisticalConstruction must use evidence_class=statistical"
            raise ValueError(msg)
        return self


class SpotCheckRecord(BaseModel):
    """Manual spot-check metadata for a verified loanword row."""

    checked_by: str
    checked_at_utc: str
    method: str
    result: str
    notes: str = ""


class VerifiedLoanword(BaseModel):
    """Independently verified Greek-loanword correspondence (CCL/DDGLC)."""

    evidence_class: EvidenceClass = EvidenceClass.INDEPENDENTLY_VERIFIED
    loanword_id: str
    coptic_lemma: str
    coptic_forms: list[str] = Field(min_length=1)
    greek_lemma: str
    greek_lemma_normalized: str = ""
    greek_pos: str = ""
    greek_meaning: str = ""
    tla_entry_id: str
    ddglc_greek_lemma_id: str
    ccl_form_ids: list[str] = Field(default_factory=list)
    license: str = "CC-BY-SA-4.0"
    attribution: str
    source_manifest_id: str = "ccl-ddglc-tei"
    spot_check: SpotCheckRecord
    restrictions: str = (
        "Lexical clue only — not automatic proof of case/tense/number/syntax."
    )
    illustrative_nt_span_ids: list[str] = Field(
        default_factory=list,
        description="Optional JAK-32 span ids from NT gold (not Thomas Greek).",
    )
    pilot_priority_notes: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _verified_only(self) -> Self:
        if self.evidence_class is not EvidenceClass.INDEPENDENTLY_VERIFIED:
            msg = "VerifiedLoanword must use evidence_class=independently_verified"
            raise ValueError(msg)
        return self


class StatisticalConstructionTable(BaseModel):
    """Export of minimal statistical phrase/construction mappings (JAK-38)."""

    id: str
    title: str
    owned_by: str = "JAK-38"
    schema_version: str = "1"
    evidence_class: EvidenceClass = EvidenceClass.STATISTICAL
    id_scheme_version: str = "1"
    segmented_id_scheme_version: str = "2"
    constructions: list[StatisticalConstruction]
    provenance: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class VerifiedLoanwordTable(BaseModel):
    """Minimal provenance-tracked CCL/DDGLC verified loanword subset (JAK-38)."""

    id: str
    title: str
    owned_by: str = "JAK-38"
    schema_version: str = "1"
    evidence_class: EvidenceClass = EvidenceClass.INDEPENDENTLY_VERIFIED
    license: str = "CC-BY-SA-4.0"
    entries: list[VerifiedLoanword]
    provenance: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AblationCondition(StrEnum):
    """Phase 2 ablation arms (JAK-38 decision)."""

    NT_ALIGNER_ONLY = "nt_aligner_only"
    NT_ALIGNER_PLUS_VERIFIED_LOANWORDS = "nt_aligner_plus_verified_loanwords"


class AblationPackage(BaseModel):
    """Documented ablation input set for Phase 2 feasibility experiments."""

    id: str
    title: str
    owned_by: str = "JAK-38"
    schema_version: str = "1"
    condition: AblationCondition
    includes: dict[str, Any]
    excludes: dict[str, Any] = Field(default_factory=dict)
    phase2_active_aligners: list[str] = Field(
        default_factory=lambda: ["eflomal", "fast_align"]
    )
    adopted_aligner: str | None = None
    decoder_consumer: str = "JAK-39"
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

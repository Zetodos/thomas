# SPDX-License-Identifier: Apache-2.0
"""Minimal deterministic decoder I/O schemas (JAK-39).

Span / construction_id inputs follow the JAK-38 surface. Candidates keep
evidence classes distinct (statistical constructions vs independently verified
loanwords vs lexical observed wordforms). Abstention is first-class: the
decoder must not invent unsupported Greek.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.schemas.mappings import AblationCondition, EvidenceClass, SpanRef


class DecoderEvidenceKind(StrEnum):
    """Where a Greek candidate came from (do not collapse these)."""

    STATISTICAL_CONSTRUCTION = "statistical_construction"
    INDEPENDENTLY_VERIFIED_LOANWORD = "independently_verified_loanword"
    LEXICAL_OBSERVED_WORDFORM = "lexical_observed_wordform"


class AbstentionReason(StrEnum):
    """Why the decoder refused to emit Greek for a span."""

    EMPTY_SPAN = "empty_span"
    NO_ATTESTED_EVIDENCE = "no_attested_evidence"
    CONSTRUCTION_ID_UNKNOWN = "construction_id_unknown"
    CONSTRUCTION_PATTERN_MISMATCH = "construction_pattern_mismatch"
    ABLATION_EXCLUDES_LOANWORDS = "ablation_excludes_loanwords"
    FORBIDDEN_QUERY_UNIT = "forbidden_query_unit"


class DecodeSpanInput(BaseModel):
    """One Coptic span to decode (JAK-32 ids + surfaces)."""

    span_id: str
    source_id: str = ""
    unit_id: str = ""
    token_ids: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(min_length=0)
    construction_id: str | None = Field(
        default=None,
        description="Optional JAK-38 construction_id hint for this span.",
    )


class DecodeRequest(BaseModel):
    """Offline decode request consuming a JAK-38 ablation arm."""

    id: str
    title: str = ""
    owned_by: str = "JAK-39"
    schema_version: str = "1"
    ablation_condition: AblationCondition
    unit_id: str
    spans: list[DecodeSpanInput] = Field(min_length=1)
    max_candidates_per_span: int = Field(default=5, ge=1, le=50)
    include_inferred_lemmas: bool = Field(
        default=False,
        description=(
            "When true, allow inferred-lemma proxies as low-priority candidates. "
            "Default false: attested observed wordforms / constructions / "
            "verified loanwords only."
        ),
    )
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CandidateEvidenceRef(BaseModel):
    """Pointer back to the evidence row that licensed a Greek candidate."""

    kind: DecoderEvidenceKind
    evidence_class: EvidenceClass | None = None
    ref_id: str
    layer: str | None = None
    notes: str = ""


class GreekCandidate(BaseModel):
    """One ranked Greek reconstruction hypothesis for a span."""

    rank: int = Field(ge=1)
    greek_surfaces: list[str] = Field(min_length=1)
    score: float
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence: list[CandidateEvidenceRef] = Field(min_length=1)
    ranking_key: list[str | int | float] = Field(default_factory=list)
    grammatical_notes: list[str] = Field(default_factory=list)
    restrictions: str = ""


class SpanDecodeStatus(StrEnum):
    """Per-span decode outcome."""

    CANDIDATES = "candidates"
    ABSTAIN = "abstain"


class SpanDecodeResult(BaseModel):
    """Decode outcome for one input span."""

    span_id: str
    unit_id: str = ""
    surfaces: list[str] = Field(default_factory=list)
    construction_id: str | None = None
    status: SpanDecodeStatus
    candidates: list[GreekCandidate] = Field(default_factory=list)
    abstention_reason: AbstentionReason | None = None
    abstention_detail: str = ""
    input_span: SpanRef | None = None

    @model_validator(mode="after")
    def _status_consistent(self) -> Self:
        if self.status is SpanDecodeStatus.ABSTAIN:
            if self.abstention_reason is None:
                msg = "abstain results require abstention_reason"
                raise ValueError(msg)
            if self.candidates:
                msg = "abstain results must not carry greek candidates"
                raise ValueError(msg)
        elif self.status is SpanDecodeStatus.CANDIDATES:
            if not self.candidates:
                msg = "candidates status requires at least one greek candidate"
                raise ValueError(msg)
            if self.abstention_reason is not None:
                msg = "candidates status must not set abstention_reason"
                raise ValueError(msg)
        return self


class DecodeResult(BaseModel):
    """Full decoder output document (schema-valid, reproducible)."""

    id: str
    title: str = ""
    owned_by: str = "JAK-39"
    schema_version: str = "1"
    request_id: str
    ablation_condition: AblationCondition
    unit_id: str
    spans: list[SpanDecodeResult]
    ranking_rules_id: str = "jak39-stable-ranking-v1"
    phase2_active_aligners: list[str] = Field(
        default_factory=lambda: ["eflomal", "fast_align"]
    )
    adopted_aligner: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

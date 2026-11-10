# SPDX-License-Identifier: Apache-2.0
"""Pilot evidence-set models spanning PLAN.md evidence types (JAK-34)."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.schemas.alignment import AgreementDegree
from zetodos_thomas.schemas.fragment import EvidenceType
from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION

PILOT_LOGIA: tuple[int, ...] = (7, 17, 30, 77)


class EvidenceStatus(StrEnum):
    """Whether a typed evidence slot is filled, reserved, or absent."""

    ATTESTED = "attested"
    PARTIAL_COVERAGE = "partial-coverage"
    PLACEHOLDER = "placeholder"
    MISSING = "missing"


class WitnessAlignmentKind(StrEnum):
    """How a Greek fragment relates to Coptic logion sequence."""

    COVERAGE = "coverage"
    DIVERGENT_SEQUENCE = "divergent-sequence"


class EvidenceRecord(BaseModel):
    """One typed evidence slot for a pilot saying (PLAN.md ``Evidence``)."""

    id: str = Field(description="Stable evidence record id")
    evidence_type: EvidenceType
    status: EvidenceStatus
    citation: str = ""
    source_id: str | None = Field(
        default=None,
        description="Manifest source id when the slot points at ingested text",
    )
    unit_id: str | None = Field(
        default=None,
        description="Linked TextUnit.unit_id when known (e.g. thomas.7)",
    )
    transcription_id: str | None = Field(
        default=None,
        description="FragmentTranscription.id when evidence_type is G",
    )
    agreement: AgreementDegree | None = None
    restrictions: str = ""
    notes: str = ""


class WitnessAlignment(BaseModel):
    """Separate witness alignment (never a silent merge into Coptic sequence)."""

    id: str
    kind: WitnessAlignmentKind
    greek_source_id: str
    greek_transcription_id: str
    coptic_unit_ids: list[str] = Field(min_length=1)
    logia_coverage: str = ""
    notes: str = ""


class PilotSaying(BaseModel):
    """One Phase 1 pilot logion with Coptic link and typed evidence slots."""

    id: str
    logion: int = Field(ge=1, le=114)
    unit_id: str = Field(description="Dilley Coptic unit id, e.g. thomas.7")
    coptic_source_id: str = "dilley-thomas"
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    attested: list[str] = Field(
        default_factory=list,
        description="Human-readable attested inventory for this saying",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="Human-readable missing / not-yet-ingested inventory",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _unit_matches_logion(self) -> Self:
        expected = f"thomas.{self.logion}"
        if self.unit_id != expected:
            msg = f"unit_id {self.unit_id!r} must be {expected!r}"
            raise ValueError(msg)
        return self


class PilotEvidenceSet(BaseModel):
    """Offline-inspectable Phase 1 pilot package for Thomas 7 / 17 / 30 / 77."""

    id: str
    title: str
    id_scheme_version: str = ID_SCHEME_VERSION
    coptic_source_id: str = "dilley-thomas"
    coptic_corpus_id: str = "dilley-thomas-v6.2.0"
    fragment_corpus_id: str = "greek-thomas-fragments-v1"
    greek_nt_spine_source_id: str = Field(
        default="ntvmr-vaticanus-b",
        description="Sole Greek NT spine (Vaticanus B*); do not retarget",
    )
    sayings: list[PilotSaying]
    witness_alignments: list[WitnessAlignment] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _pilot_inventory(self) -> Self:
        if self.id_scheme_version != ID_SCHEME_VERSION:
            msg = (
                f"id_scheme_version {self.id_scheme_version!r} != "
                f"code {ID_SCHEME_VERSION!r}"
            )
            raise ValueError(msg)
        if self.greek_nt_spine_source_id != "ntvmr-vaticanus-b":
            msg = "greek_nt_spine_source_id must remain ntvmr-vaticanus-b (JAK-25)"
            raise ValueError(msg)
        logia = [s.logion for s in self.sayings]
        if tuple(logia) != PILOT_LOGIA:
            msg = f"pilot sayings must be exactly {PILOT_LOGIA} in order (got {logia})"
            raise ValueError(msg)
        if len({s.id for s in self.sayings}) != len(self.sayings):
            msg = "pilot saying ids must be unique"
            raise ValueError(msg)
        evidence_ids = [e.id for s in self.sayings for e in s.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            msg = "evidence record ids must be unique across the pilot set"
            raise ValueError(msg)
        for saying in self.sayings:
            if not saying.attested:
                msg = f"{saying.unit_id}: attested list must be non-empty"
                raise ValueError(msg)
            if not saying.missing:
                msg = f"{saying.unit_id}: missing list must be non-empty"
                raise ValueError(msg)
            types = {e.evidence_type for e in saying.evidence}
            required_types = (
                EvidenceType.G,
                EvidenceType.L,
                EvidenceType.N,
                EvidenceType.P,
            )
            for required in required_types:
                if required not in types:
                    msg = (
                        f"{saying.unit_id}: missing required evidence slot "
                        f"for type {required}"
                    )
                    raise ValueError(msg)
        return self

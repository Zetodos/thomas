# SPDX-License-Identifier: Apache-2.0
"""Typed models for Greek Thomas fragment transcriptions (evidence type G)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class EvidenceType(StrEnum):
    """PLAN.md §3.2 evidence typology.

    Fragment ingest (JAK-31) commits only ``G``. The Phase 1 pilot set (JAK-34)
    also uses ``L`` / ``N`` / ``P`` (and optional ``R``) slots.
    """

    G = "G"  # Direct Greek manuscripts of Thomas
    L = "L"  # Greek-derived words in Coptic
    N = "N"  # Greek NT / Septuagint parallels
    P = "P"  # Other early Greek literary attestations
    R = "R"  # Modern scholarly retrotranslation (not independent ancient evidence)


class CorpusRole(StrEnum):
    """How a corpus may be used relative to alignment training (JAK-28)."""

    EVALUATION_WITNESS = "evaluation-witness"
    TRAINING_OK = "training-ok"
    DISPLAY_ONLY = "display-only"


class FragmentLayerKind(StrEnum):
    """Recoverable text layers — never silently merge these."""

    SURVIVING = "surviving"
    LACUNA = "lacuna"
    RESTORATION = "restoration"


class FragmentToken(BaseModel):
    """One atomic span in a single recoverable layer."""

    kind: FragmentLayerKind
    text: str = ""
    reason: str | None = Field(
        default=None,
        description="TEI @reason or project reason (lost, omitted, abbreviation, …)",
    )
    quantity: str | None = None
    unit: str | None = None
    certainty: str | None = Field(
        default=None,
        description=(
            "Optional certainty hint (e.g. unclear for damaged surviving letters)"
        ),
    )


class FragmentLine(BaseModel):
    """One papyrus line (or column line) with layered tokens."""

    line_id: str
    n: str
    side_or_column: str | None = None
    tokens: list[FragmentToken]

    @property
    def surviving_text(self) -> str:
        return "".join(
            t.text for t in self.tokens if t.kind is FragmentLayerKind.SURVIVING
        )

    @property
    def restoration_text(self) -> str:
        return "".join(
            t.text for t in self.tokens if t.kind is FragmentLayerKind.RESTORATION
        )

    @property
    def lacuna_count(self) -> int:
        return sum(1 for t in self.tokens if t.kind is FragmentLayerKind.LACUNA)


class FragmentTranscription(BaseModel):
    """One manuscript transcription imported under a source manifest."""

    id: str
    source_id: str = Field(description="Must resolve to a SourceManifest.id")
    manuscript_id: str
    tm_id: str | None = None
    dclp_id: str | None = None
    evidence_type: EvidenceType = EvidenceType.G
    corpus_role: CorpusRole = CorpusRole.EVALUATION_WITNESS
    language: str = "grc"
    logia_coverage: str = ""
    lines: list[FragmentLine]
    notes: str = ""

    @field_validator("source_id")
    @classmethod
    def _normalize_source_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            msg = "source_id must be non-empty"
            raise ValueError(msg)
        return cleaned


class FragmentCorpus(BaseModel):
    """Bundled Greek Thomas fragment set (evidence type G)."""

    id: str
    title: str
    evidence_type: EvidenceType = EvidenceType.G
    corpus_role: CorpusRole = CorpusRole.EVALUATION_WITNESS
    transcriptions: list[FragmentTranscription]
    excluded_witnesses: list[str] = Field(
        default_factory=list,
        description="IDs/labels explicitly not Thomas manuscripts (e.g. P.Oxy. 5575)",
    )
    notes: str = ""

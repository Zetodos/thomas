# SPDX-License-Identifier: Apache-2.0
"""Typed source-manifest and imported text-unit models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class AvailabilityMode(StrEnum):
    """How a source may be obtained for offline work."""

    BUNDLED = "bundled"
    SETUP_DOWNLOAD = "setup-download"
    MANUAL_IMPORT = "manual-import"


# Hard exclusions from the default CLI/CI path (JAK-14 / JAK-15 / JAK-24).
GATED_DEFAULT_SOURCE_IDS: frozenset[str] = frozenset(
    {
        "sahidica",
        "copsahidica",
        "ntaeg",
        "bohairica",
        "wells-sahidica",
        "wells-ntaeg",
        "wells-bohairica",
    }
)


class SourceManifest(BaseModel):
    """Per-source rights, provenance, and availability record."""

    id: str = Field(description="Stable canonical source ID")
    title: str
    version: str
    url: str
    spdx_or_custom: str = Field(
        description="SPDX id or custom:<label> (e.g. custom:Public-Domain)"
    )
    attribution: str
    mode: AvailabilityMode
    checksum: str = Field(
        description="Algorithm-prefixed digest, e.g. sha256:…",
    )
    notes: str = ""
    editor: str | None = None
    language: str | None = None
    dialect: str | None = None
    license_url: str | None = None
    local_path: str | None = None
    redistributable: bool | None = None
    gate: str | None = Field(
        default=None,
        description="If gated: reason / Linear issue reference",
    )
    module_name: str | None = Field(
        default=None,
        description="CrossWire / upstream module identifier when applicable",
    )
    sword_version_date: str | None = None
    text_source: str | None = Field(
        default=None,
        description="Digitization provenance chain (e.g. Slavic Bible for Windows)",
    )
    counterpart_source_id: str | None = Field(
        default=None,
        description="Preferred parallel edition id (edition-independent pairing hint)",
    )
    manuscript_id: str | None = Field(
        default=None,
        description=(
            "Manuscript / papyrus shelfmark (required for fragment transcriptions)"
        ),
    )
    tm_id: str | None = Field(
        default=None,
        description="Trismegistos text id when applicable",
    )
    evidence_type: str | None = Field(
        default=None,
        description="PLAN.md evidence type (G for direct Greek Thomas manuscripts)",
    )
    corpus_role: str | None = Field(
        default=None,
        description=(
            "evaluation-witness | training-ok | display-only — fragment G texts "
            "default to evaluation-witness (JAK-28 leakage control)"
        ),
    )

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            msg = "source id must be non-empty"
            raise ValueError(msg)
        return cleaned

    @field_validator("checksum")
    @classmethod
    def _require_checksum_prefix(cls, value: str) -> str:
        if value == "":
            return value
        if ":" not in value:
            msg = "checksum must be algorithm-prefixed (e.g. sha256:…)"
            raise ValueError(msg)
        algorithm, digest = value.split(":", 1)
        if not algorithm or not digest:
            msg = "checksum must include both algorithm and digest"
            raise ValueError(msg)
        return value


class UnitKind(StrEnum):
    """What a ``TextUnit.unit_id`` denotes."""

    VERSE = "verse"
    LOGION = "logion"
    PROLOGUE = "prologue"
    SECTION = "section"


class TextUnit(BaseModel):
    """One verse/logion/section with original and normalized surface forms.

    ``unit_id`` is the stable handle for alignments (JAK-32). Optional token
    layers live in ``AlignmentDocument`` / ``ThomasCorpus`` (shared ``Token`` /
    ``Span`` in ``schemas.alignment``) rather than inline here so existing
    ``PilotCorpus`` JSON stays unchanged.

    Thomas prologue is ``unit_id=thomas.0``, ``chapter=0``, ``kind=prologue``.
    Logia use ``thomas.N`` with ``chapter=N`` (N ≥ 1) and ``kind=logion``.
    """

    source_id: str
    unit_id: str = Field(
        description="Stable unit id, e.g. john.1.1, thomas.0 (prologue), thomas.7",
    )
    book: str
    chapter: int = Field(
        ge=0,
        description="Chapter/logion number; 0 reserved for Thomas prologue",
    )
    verse: int = Field(ge=0)
    original_text: str
    normalized_text: str
    sequence: int = Field(default=0, ge=0)
    kind: UnitKind = Field(
        default=UnitKind.VERSE,
        description="Structural kind; default verse keeps JAK-24 pilot valid",
    )


class ParallelUnit(BaseModel):
    """Greek-Sahidic verse pair for pilot alignment."""

    unit_id: str
    greek: TextUnit
    sahidic: TextUnit
    notes: str = ""


class PilotCorpus(BaseModel):
    """Small redistributable Greek-Sahidic pilot excerpt."""

    id: str
    title: str
    greek_source_id: str
    sahidic_source_id: str
    units: list[ParallelUnit]
    notes: str = ""

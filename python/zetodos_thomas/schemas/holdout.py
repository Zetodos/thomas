# SPDX-License-Identifier: Apache-2.0
"""Benchmark split, omission-fixture, and fragment-divergence schemas (JAK-33)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class PassageRole(StrEnum):
    """How a whole passage/book may be used in Greek-Sahidic alignment work."""

    TRAIN = "train"
    DEV = "dev"
    HOLDOUT = "holdout"


class PassageRef(BaseModel):
    """Whole-passage or whole-book reference (never a random word sample)."""

    book: str
    chapter_from: int = Field(ge=1)
    verse_from: int = Field(ge=0)
    chapter_to: int = Field(ge=1)
    verse_to: int = Field(ge=0)
    unit_ids: list[str] = Field(
        default_factory=list,
        description="Explicit verse/logion unit ids covered by this passage",
    )
    role: PassageRole
    notes: str = ""

    @field_validator("book")
    @classmethod
    def _normalize_book(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            msg = "book must be non-empty"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def _range_order(self) -> PassageRef:
        start = (self.chapter_from, self.verse_from)
        end = (self.chapter_to, self.verse_to)
        if end < start:
            msg = "passage end must be at or after start"
            raise ValueError(msg)
        return self


class EvaluationWitnessExclusion(BaseModel):
    """A witness excluded from training mappings (leakage control for JAK-28)."""

    witness_id: str
    source_id: str
    manuscript_id: str = ""
    corpus_role: str = "evaluation-witness"
    unit_ids: list[str] = Field(
        default_factory=list,
        description="Thomas logion unit ids this witness is intended to evaluate",
    )
    notes: str = ""


class NtCorpusSplit(BaseModel):
    """Written train/dev/holdout split for Greek-Sahidic NT (JAK-33).

    Ownership: JAK-33 defines the policy artifact. JAK-28 consumes it for
    alignment benchmarks and must not silently reassign roles without an
    explicit schema bump. Secondary OT is recorded only when used.
    """

    id: str
    title: str
    schema_version: str = "1"
    greek_source_id: str = "ntvmr-vaticanus-b"
    sahidic_source_id: str = "copsahhorner"
    pilot_corpus_id: str = "horner-vaticanus-nt-pilot-v1"
    owned_by: str = "JAK-33"
    consumed_by: list[str] = Field(default_factory=lambda: ["JAK-28"])
    passages: list[PassageRef]
    evaluation_witness_exclusions: list[EvaluationWitnessExclusion] = Field(
        default_factory=list
    )
    secondary_ot_used: bool = False
    secondary_ot_notes: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _roles_and_overlap(self) -> NtCorpusSplit:
        if self.greek_source_id != "ntvmr-vaticanus-b":
            msg = "greek_source_id must be ntvmr-vaticanus-b (Vaticanus B* only)"
            raise ValueError(msg)
        if self.sahidic_source_id != "copsahhorner":
            msg = "sahidic_source_id must be copsahhorner"
            raise ValueError(msg)

        seen_units: dict[str, PassageRole] = {}
        roles_present: set[PassageRole] = set()
        for passage in self.passages:
            roles_present.add(passage.role)
            for unit_id in passage.unit_ids:
                prior = seen_units.get(unit_id)
                if prior is not None and prior != passage.role:
                    msg = (
                        f"unit_id {unit_id!r} assigned to both "
                        f"{prior} and {passage.role}"
                    )
                    raise ValueError(msg)
                seen_units[unit_id] = passage.role

        required = {PassageRole.TRAIN, PassageRole.DEV, PassageRole.HOLDOUT}
        missing = required - roles_present
        if missing:
            # Allow empty-role presence via explicit zero-unit passages.
            # Callers must still include a PassageRef per required role.
            msg = (
                "split must include train, dev, and holdout PassageRef entries "
                f"(missing roles: {sorted(r.value for r in missing)})"
            )
            raise ValueError(msg)

        if not any(p.role is PassageRole.TRAIN and p.unit_ids for p in self.passages):
            msg = "train must cover at least one unit_id"
            raise ValueError(msg)
        if not any(p.role is PassageRole.HOLDOUT and p.unit_ids for p in self.passages):
            msg = "holdout must cover at least one unit_id"
            raise ValueError(msg)

        for exclusion in self.evaluation_witness_exclusions:
            if exclusion.corpus_role != "evaluation-witness":
                msg = (
                    "evaluation_witness_exclusions must keep "
                    "corpus_role='evaluation-witness'"
                )
                raise ValueError(msg)
        return self


class OmissionFixtureKind(StrEnum):
    """Why a unit is an omission / empty fixture rather than a positive pair."""

    SHARED_ABSENCE = "shared-absence"
    GREEK_EMPTY = "greek-empty"
    SAHIDIC_EMPTY = "sahidic-empty"
    OTHER = "other"


class OmissionUnit(BaseModel):
    """One empty/omission unit listed separately from positive lexical pairs."""

    unit_id: str
    book: str
    chapter: int = Field(ge=1)
    verse: int = Field(ge=0)
    kind: OmissionFixtureKind
    greek_empty: bool = True
    sahidic_empty: bool = True
    notes: str = ""


class OmissionFixtureSet(BaseModel):
    """Negative/omission fixtures excluded from positive lexical scoring."""

    id: str
    title: str
    schema_version: str = "1"
    greek_source_id: str = "ntvmr-vaticanus-b"
    sahidic_source_id: str = "copsahhorner"
    pilot_corpus_id: str = "horner-vaticanus-nt-pilot-v1"
    owned_by: str = "JAK-33"
    consumed_by: list[str] = Field(default_factory=lambda: ["JAK-28"])
    units: list[OmissionUnit]
    notes: str = ""

    @model_validator(mode="after")
    def _nonempty_and_empty_sides(self) -> OmissionFixtureSet:
        if not self.units:
            msg = "omission fixture set must list at least one unit"
            raise ValueError(msg)
        for unit in self.units:
            if not (unit.greek_empty or unit.sahidic_empty):
                msg = (
                    f"omission unit {unit.unit_id!r} must mark greek_empty "
                    "and/or sahidic_empty"
                )
                raise ValueError(msg)
        return self


class DivergenceKind(StrEnum):
    """How a Greek fragment differs from the Coptic Thomas sequence/edition."""

    SEQUENCE = "sequence"
    DIVISION = "division"
    WORDING = "wording"
    COVERAGE = "coverage"
    NOT_THOMAS_MS = "not-thomas-manuscript"
    OTHER = "other"


class FragmentDivergence(BaseModel):
    """One documented divergence — separate witnesses; never a silent merge."""

    id: str
    kind: DivergenceKind
    greek_witness_id: str
    greek_source_id: str
    manuscript_id: str
    coptic_unit_ids: list[str] = Field(
        default_factory=list,
        description="Dilley / Coptic Thomas unit ids (e.g. thomas.30, thomas.77)",
    )
    greek_coverage_note: str = ""
    summary: str
    handling: str = (
        "Keep as separate witnesses; do not silently merge into Coptic sequence."
    )
    notes: str = ""


class FragmentDivergenceDocument(BaseModel):
    """Catalog of Greek Thomas fragment vs Coptic sequence/edition divergences."""

    id: str
    title: str
    schema_version: str = "1"
    owned_by: str = "JAK-33"
    consumed_by: list[str] = Field(default_factory=lambda: ["JAK-28", "JAK-34"])
    coptic_source_id: str = "dilley-thomas"
    fragment_corpus_id: str = "greek-thomas-fragments-v1"
    divergences: list[FragmentDivergence]
    excluded_witnesses: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _require_divergences(self) -> FragmentDivergenceDocument:
        if not self.divergences:
            msg = "divergence document must list at least one divergence"
            raise ValueError(msg)
        ids = [d.id for d in self.divergences]
        if len(ids) != len(set(ids)):
            msg = "divergence ids must be unique"
            raise ValueError(msg)
        return self

# SPDX-License-Identifier: Apache-2.0
"""Unified linguistic schemas for annotated Coptic, Greek, and parallel corpora."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.source import UnitKind


class CopticMorpheme(BaseModel):
    """One segmented Coptic morpheme with full SCRIPTORIUM / analyzer layers."""

    id: str = ""
    unit_id: str = ""
    seq: int = Field(default=-1, ge=-1)
    group_seq: int = Field(ge=0)
    orig: str = ""
    norm: str
    lemma: str
    pos: str
    morph: tuple[str, ...] = ()
    lang: str | None = None
    multiword: str | None = None
    head_seq: int | None = None
    dep_rel: str | None = None


class CopticUnit(BaseModel):
    """One Coptic verse, prologue, or logion with full morpheme annotations."""

    unit_id: str
    source_id: str
    book: str
    chapter: int = Field(ge=0)
    verse: int = Field(ge=0)
    kind: UnitKind = UnitKind.VERSE
    original_text: str
    normalized_text: str
    morphemes: list[CopticMorpheme] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_sequences(self) -> Self:
        for idx, m in enumerate(self.morphemes):
            if m.seq < 0:
                m.seq = idx
            elif m.seq != idx:
                msg = f"{self.unit_id}: morpheme seq {m.seq} != index {idx}"
                raise ValueError(msg)
            if not m.unit_id:
                m.unit_id = self.unit_id
            if not m.id:
                m.id = token_id(self.source_id, self.unit_id, idx)
            if not m.orig:
                m.orig = m.norm
        return self


class CopticCorpus(BaseModel):
    """Annotated Coptic corpus (e.g. Dilley Gospel of Thomas)."""

    id: str
    title: str
    source_id: str
    id_scheme_version: str = "1"
    annotation_level: str = "token-complete"
    units: list[CopticUnit]
    notes: str = ""


class GreekMorphFeatures(BaseModel):
    """MorphGNT-compatible 8-slot Koine Greek morphological analysis."""

    pos: str
    person: str | None = None
    tense: str | None = None
    voice: str | None = None
    mood: str | None = None
    case: str | None = None
    number: str | None = None
    gender: str | None = None
    degree: str | None = None

    @classmethod
    def from_morphgnt(cls, pos: str, parse_code: str) -> GreekMorphFeatures:
        """Build from MorphGNT 2-char POS and 8-char parsing code."""
        code = parse_code.ljust(8, "-")[:8]

        def _slot(idx: int) -> str | None:
            ch = code[idx]
            return None if ch == "-" else ch

        return cls(
            pos=pos.strip(),
            person=_slot(0),
            tense=_slot(1),
            voice=_slot(2),
            mood=_slot(3),
            case=_slot(4),
            number=_slot(5),
            gender=_slot(6),
            degree=_slot(7),
        )

    def signature(self) -> str:
        """Compact canonical slot string for lookup keys."""
        slots = "".join(
            ch or "-"
            for ch in (
                self.person,
                self.tense,
                self.voice,
                self.mood,
                self.case,
                self.number,
                self.gender,
                self.degree,
            )
        )
        return f"{self.pos}:{slots}"


class GreekToken(BaseModel):
    """One Koine Greek token with polytonic surface, lemma, and morphology."""

    id: str
    unit_id: str
    seq: int = Field(ge=0)
    surface: str
    normalized: str
    lemma: str
    morph: GreekMorphFeatures


class GreekUnit(BaseModel):
    """One Greek verse/unit with morphological token sequence."""

    unit_id: str
    source_id: str
    book: str
    chapter: int = Field(ge=0)
    verse: int = Field(ge=0)
    original_text: str
    normalized_text: str
    tokens: list[GreekToken] = Field(default_factory=list)


class ParallelVerse(BaseModel):
    """Paired Coptic and Greek verse with split role and domain label."""

    unit_id: str
    coptic: CopticUnit
    greek: GreekUnit
    split_role: str = "train"
    domain: str = "nt"

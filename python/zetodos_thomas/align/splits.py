# SPDX-License-Identifier: Apache-2.0
"""Passage-level train/dev/holdout/omission splits for the NT alignment pilot."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus


class SplitRole(StrEnum):
    """Whole-passage role; never mixed within a verse."""

    TRAIN = "train"
    DEV = "dev"
    HOLDOUT = "holdout"
    OMISSION = "omission"


class PassageRange(BaseModel):
    """Inclusive verse range labeled with a split role."""

    book: str
    chapter: int = Field(ge=1)
    verse_from: int = Field(ge=0)
    verse_to: int = Field(ge=0)
    role: SplitRole
    rationale: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> PassageRange:
        if self.verse_to < self.verse_from:
            msg = "verse_to must be >= verse_from"
            raise ValueError(msg)
        return self

    def contains(self, unit_id: str) -> bool:
        """Return True when ``unit_id`` (``book.chapter.verse``) is in range."""
        parts = unit_id.split(".")
        if len(parts) != 3:
            return False
        book, chapter_s, verse_s = parts
        try:
            chapter = int(chapter_s)
            verse = int(verse_s)
        except ValueError:
            return False
        return (
            book == self.book
            and chapter == self.chapter
            and self.verse_from <= verse <= self.verse_to
        )

    def unit_ids(self) -> list[str]:
        """Expand this range to unit_id strings."""
        return [
            f"{self.book}.{self.chapter}.{verse}"
            for verse in range(self.verse_from, self.verse_to + 1)
        ]


class PilotSplits(BaseModel):
    """Documented passage splits for JAK-28 (no random word holdouts)."""

    id: str
    title: str
    greek_source_id: str
    sahidic_source_id: str
    passages: list[PassageRange]
    notes: str = ""
    # Explicit non-inclusion of Thomas Greek fragments (JAK-33/34).
    excluded_from_training_stack: list[str] = Field(
        default_factory=list,
        description="Witnesses deliberately kept out of this NT training stack",
    )

    @model_validator(mode="after")
    def _reject_overlapping_ranges(self) -> Self:
        """Forbid a unit_id appearing in more than one passage range.

        Silent first-match resolution would let a custom splits file put the
        same verse in train and holdout (fit as train, report as holdout).
        """
        claimed: dict[str, SplitRole] = {}
        for passage in self.passages:
            for unit_id in passage.unit_ids():
                prior = claimed.get(unit_id)
                if prior is not None:
                    msg = (
                        f"overlapping split ranges for {unit_id}: "
                        f"{prior.value} and {passage.role.value}"
                    )
                    raise ValueError(msg)
                claimed[unit_id] = passage.role
        return self

    def role_for(self, unit_id: str) -> SplitRole | None:
        """Look up the split role for a unit_id."""
        for passage in self.passages:
            if passage.contains(unit_id):
                return passage.role
        return None

    def unit_ids_for(self, role: SplitRole) -> list[str]:
        """All unit_ids assigned to ``role`` (stable order)."""
        out: list[str] = []
        for passage in self.passages:
            if passage.role is role:
                out.extend(passage.unit_ids())
        return out


DEFAULT_SPLITS = PilotSplits(
    id="horner-vaticanus-pilot-splits-v1",
    title="Horner x Vaticanus B* alignment pilot passage splits (JAK-28)",
    greek_source_id="ntvmr-vaticanus-b",
    sahidic_source_id="copsahhorner",
    passages=[
        PassageRange(
            book="john",
            chapter=1,
            verse_from=1,
            verse_to=18,
            role=SplitRole.TRAIN,
            rationale="Prologue; loanwords/nomina sacra; sayings-like density",
        ),
        PassageRange(
            book="mark",
            chapter=1,
            verse_from=1,
            verse_to=8,
            role=SplitRole.TRAIN,
            rationale="Gospel opening; Greek loanwords; Mark 1:1 heading paratext",
        ),
        PassageRange(
            book="matthew",
            chapter=5,
            verse_from=14,
            verse_to=16,
            role=SplitRole.TRAIN,
            rationale="Light / city sayings (Thomas-adjacent imagery); loanwords",
        ),
        PassageRange(
            book="matthew",
            chapter=7,
            verse_from=7,
            verse_to=8,
            role=SplitRole.TRAIN,
            rationale="Imperatives ask/seek/knock (Thomas 2/92/94 parallel zone)",
        ),
        PassageRange(
            book="luke",
            chapter=6,
            verse_from=20,
            verse_to=21,
            role=SplitRole.DEV,
            rationale="Beatitudes; pronouns; clause variation (Thomas 54 zone)",
        ),
        PassageRange(
            book="matthew",
            chapter=13,
            verse_from=31,
            verse_to=32,
            role=SplitRole.HOLDOUT,
            rationale="Mustard-seed parable whole-passage holdout (Thomas 20 zone)",
        ),
        PassageRange(
            book="john",
            chapter=8,
            verse_from=1,
            verse_to=11,
            role=SplitRole.OMISSION,
            rationale="PA dual-empty omission fixture; never positive lexical scores",
        ),
    ],
    notes=(
        "Splits are whole passages/books, never random words. "
        "Omission units stay in the corpus for negative tests but are excluded "
        "from positive lexical/phrase denominators. Greek Thomas fragment "
        "witnesses (notably logia 7/17/30/77) are not part of this NT stack."
    ),
    excluded_from_training_stack=[
        "thomas-greek-fragments-poxy (JAK-31; logia 7/17/30/77 holdouts)",
        "coptic-thomas-dilley (JAK-30; reconstruction target, not NT train)",
    ],
)


def load_splits(path: Path) -> PilotSplits:
    """Load splits JSON."""
    return PilotSplits.model_validate_json(path.read_text(encoding="utf-8"))


def write_splits(splits: PilotSplits, path: Path) -> None:
    """Write splits JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(splits.model_dump_json(indent=2) + "\n", encoding="utf-8")


def partition_corpus(
    corpus: PilotCorpus,
    splits: PilotSplits,
) -> dict[SplitRole, list[ParallelUnit]]:
    """Bucket pilot units by split role; raise if any unit is unlabeled."""
    buckets: dict[SplitRole, list[ParallelUnit]] = {role: [] for role in SplitRole}
    unlabeled: list[str] = []
    for unit in corpus.units:
        role = splits.role_for(unit.unit_id)
        if role is None:
            unlabeled.append(unit.unit_id)
            continue
        buckets[role].append(unit)
    if unlabeled:
        msg = f"pilot units missing split labels: {unlabeled}"
        raise ValueError(msg)
    return buckets


def nonempty_positive_units(units: list[ParallelUnit]) -> list[ParallelUnit]:
    """Units with text on both sides (usable lexical pairs)."""
    return [
        u
        for u in units
        if u.greek.original_text.strip() and u.sahidic.original_text.strip()
    ]


def dumps_splits_summary(splits: PilotSplits) -> str:
    """Human-readable summary for CLI/docs regeneration helpers."""
    lines = [f"{splits.id}: {splits.title}"]
    for passage in splits.passages:
        n = passage.verse_to - passage.verse_from + 1
        lines.append(
            f"  {passage.role}: {passage.book} {passage.chapter}:"
            f"{passage.verse_from}-{passage.verse_to} ({n} units) — "
            f"{passage.rationale}"
        )
    return "\n".join(lines)


def splits_from_mapping(data: dict[str, object]) -> PilotSplits:
    """Validate a raw mapping (tests / callers)."""
    return PilotSplits.model_validate(data)


def read_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object from disk."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"expected JSON object in {path}"
        raise TypeError(msg)
    return raw

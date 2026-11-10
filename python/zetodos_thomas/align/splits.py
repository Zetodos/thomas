# SPDX-License-Identifier: Apache-2.0
"""JAK-28 consumer of the JAK-33 canonical NT train/dev/holdout split.

The authoritative artifact lives at
``data/training/splits/horner-vaticanus-nt-split-v2.json`` (plus the John 8
omission fixture catalog). This module adapts those schemas into the runtime
``PilotSplits`` view used by validate/benchmark — it does **not** define a
second competing split policy under ``data/sources/pilot/``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.ingest.holdouts import load_nt_split, load_omission_fixtures
from zetodos_thomas.schemas.holdout import NtCorpusSplit, OmissionFixtureSet
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus


class SplitRole(StrEnum):
    """Whole-passage role; never mixed within a verse."""

    TRAIN = "train"
    DEV = "dev"
    HOLDOUT = "holdout"
    OMISSION = "omission"


class PassageRange(BaseModel):
    """Inclusive same-chapter verse range labeled with a split role."""

    book: str
    chapter: int = Field(ge=1)
    verse_from: int = Field(ge=0)
    verse_to: int = Field(ge=0)
    role: SplitRole
    rationale: str = ""
    unit_ids: list[str] = Field(
        default_factory=list,
        description="Explicit unit ids from the JAK-33 artifact (preferred)",
    )

    @model_validator(mode="after")
    def _ordered(self) -> PassageRange:
        if self.verse_to < self.verse_from:
            msg = "verse_to must be >= verse_from"
            raise ValueError(msg)
        return self

    def contains(self, unit_id: str) -> bool:
        """Return True when ``unit_id`` is in this passage."""
        if self.unit_ids:
            return unit_id in self.unit_ids
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

    def listed_unit_ids(self) -> list[str]:
        """Return explicit unit ids, or expand the verse range when absent."""
        if self.unit_ids:
            return list(self.unit_ids)
        return [
            f"{self.book}.{self.chapter}.{verse}"
            for verse in range(self.verse_from, self.verse_to + 1)
        ]


class PilotSplits(BaseModel):
    """Runtime split view for JAK-28 (always derived from JAK-33 artifacts)."""

    id: str
    title: str
    greek_source_id: str
    sahidic_source_id: str
    pilot_corpus_id: str = ""
    owned_by: str = "JAK-33"
    passages: list[PassageRange]
    notes: str = ""
    excluded_from_training_stack: list[str] = Field(
        default_factory=list,
        description="Witnesses deliberately kept out of this NT training stack",
    )
    source_split_path: str = ""
    source_omission_path: str = ""

    @model_validator(mode="after")
    def _reject_overlapping_ranges(self) -> Self:
        """Forbid a unit_id appearing in more than one passage."""
        claimed: dict[str, SplitRole] = {}
        for passage in self.passages:
            for unit_id in passage.listed_unit_ids():
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
                out.extend(passage.listed_unit_ids())
        return out


def _portable_data_path(path: Path | None) -> str:
    """Prefer a checkout-relative ``data/...`` path for committed reports."""
    if path is None:
        return ""
    resolved = path.resolve()
    parts = resolved.parts
    if "data" in parts:
        idx = parts.index("data")
        return "/".join(parts[idx:])
    return resolved.name


def pilot_splits_from_artifacts(
    split: NtCorpusSplit,
    omissions: OmissionFixtureSet,
    *,
    split_path: Path | None = None,
    omission_path: Path | None = None,
) -> PilotSplits:
    """Adapt JAK-33 ``NtCorpusSplit`` + omission catalog into ``PilotSplits``."""
    if split.pilot_corpus_id != omissions.pilot_corpus_id:
        msg = (
            f"split pilot_corpus_id {split.pilot_corpus_id!r} != "
            f"omission pilot_corpus_id {omissions.pilot_corpus_id!r}"
        )
        raise ValueError(msg)

    passages: list[PassageRange] = []
    for passage in split.passages:
        if not passage.unit_ids:
            # Empty reserved role (historical); skip for runtime bucketing.
            continue
        if passage.chapter_from != passage.chapter_to:
            msg = (
                "JAK-28 runtime adapter requires same-chapter passages; got "
                f"{passage.book} {passage.chapter_from}-{passage.chapter_to}"
            )
            raise ValueError(msg)
        passages.append(
            PassageRange(
                book=passage.book,
                chapter=passage.chapter_from,
                verse_from=passage.verse_from,
                verse_to=passage.verse_to,
                role=SplitRole(passage.role.value),
                rationale=passage.notes,
                unit_ids=list(passage.unit_ids),
            )
        )

    omission_ids = [unit.unit_id for unit in omissions.units]
    if omission_ids:
        first = omissions.units[0]
        passages.append(
            PassageRange(
                book=first.book,
                chapter=first.chapter,
                verse_from=min(u.verse for u in omissions.units),
                verse_to=max(u.verse for u in omissions.units),
                role=SplitRole.OMISSION,
                rationale=omissions.notes or "JAK-33 omission fixture catalog",
                unit_ids=omission_ids,
            )
        )

    excluded = [
        (f"{ex.witness_id} ({ex.manuscript_id or ex.source_id}; {ex.corpus_role})")
        for ex in split.evaluation_witness_exclusions
    ]
    if any("poxy" in ex.witness_id for ex in split.evaluation_witness_exclusions):
        excluded.append(
            "thomas-greek-fragments-poxy (JAK-31; logia 7/17/30/77 holdouts)"
        )

    return PilotSplits(
        id=split.id,
        title=split.title,
        greek_source_id=split.greek_source_id,
        sahidic_source_id=split.sahidic_source_id,
        pilot_corpus_id=split.pilot_corpus_id,
        owned_by=split.owned_by,
        passages=passages,
        notes=split.notes,
        excluded_from_training_stack=excluded,
        source_split_path=_portable_data_path(split_path),
        source_omission_path=_portable_data_path(omission_path),
    )


def load_canonical_splits(
    split_path: Path,
    omission_path: Path,
) -> PilotSplits:
    """Load JAK-33 split + omission artifacts into the JAK-28 runtime view."""
    split = load_nt_split(split_path)
    omissions = load_omission_fixtures(omission_path)
    return pilot_splits_from_artifacts(
        split,
        omissions,
        split_path=split_path,
        omission_path=omission_path,
    )


def load_splits(path: Path, *, omission_path: Path | None = None) -> PilotSplits:
    """Load a JAK-33 ``NtCorpusSplit`` JSON (canonical training split)."""
    if omission_path is None:
        from zetodos_thomas.ingest.paths import resolve_omission_fixtures_path

        root: Path | None = None
        for candidate in (path.resolve(), *path.resolve().parents):
            if (candidate / "pyproject.toml").is_file():
                root = candidate
                break
        omission_path = resolve_omission_fixtures_path(root)
    return load_canonical_splits(path, omission_path)


def default_splits() -> PilotSplits:
    """Load packaged JAK-33 artifacts (wheel / offline fallback)."""
    from zetodos_thomas.bundled import (
        bundled_nt_split_path,
        bundled_omission_fixtures_path,
    )

    return load_canonical_splits(
        bundled_nt_split_path(),
        bundled_omission_fixtures_path(),
    )


# Packaged JAK-33 view for call sites that historically used a constant.
DEFAULT_SPLITS = default_splits()


def partition_corpus(
    corpus: PilotCorpus,
    splits: PilotSplits,
) -> dict[SplitRole, list[ParallelUnit]]:
    """Bucket pilot units by split role; raise if any unit is unlabeled."""
    if splits.pilot_corpus_id and splits.pilot_corpus_id != corpus.id:
        msg = (
            f"split pilot_corpus_id {splits.pilot_corpus_id!r} != "
            f"pilot id {corpus.id!r}"
        )
        raise ValueError(msg)
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
    lines = [f"{splits.id}: {splits.title} (owned_by={splits.owned_by})"]
    for passage in splits.passages:
        n = len(passage.listed_unit_ids())
        lines.append(
            f"  {passage.role}: {passage.book} {passage.chapter}:"
            f"{passage.verse_from}-{passage.verse_to} ({n} units) — "
            f"{passage.rationale}"
        )
    return "\n".join(lines)

# SPDX-License-Identifier: Apache-2.0
"""Token, span, and phrase-alignment models with referential integrity checks.

Extends the JAK-24/JAK-25 ``TextUnit`` / ``ParallelUnit`` / ``PilotCorpus``
verse-pairing layer (PLAN.md §6) with span-first IDs suitable for JAK-28
evaluation. Many-to-many Coptic↔Greek mappings are first-class; discontinuous
and empty spans are allowed when explicitly marked.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION
from zetodos_thomas.schemas.source import TextUnit


class SpanKind(StrEnum):
    """Structural shape of a span over a unit's token sequence."""

    CONTIGUOUS = "contiguous"
    DISCONTINUOUS = "discontinuous"
    EMPTY = "empty"


class ParallelRelation(StrEnum):
    """How left (usually Coptic/Sahidic/Thomas) and right (Greek) spans relate."""

    VERSE_PAIR = "verse-pair"
    PHRASE = "phrase"
    LEXICAL = "lexical"
    LOANWORD = "loanword"
    OTHER = "other"


class AgreementDegree(StrEnum):
    """Coarse agreement label for parallels (PLAN.md evidence types)."""

    LEXICAL = "lexical"
    VERBAL = "verbal"
    SEMANTIC = "semantic"
    SHARED_ABSENCE = "shared-absence"
    UNCERTAIN = "uncertain"
    NONE = "none"


class Token(BaseModel):
    """One ordered token with a stable id that survives presentation changes."""

    id: str = Field(description="Stable token id (see docs/id-scheme.md)")
    source_id: str
    unit_id: str
    sequence: int = Field(ge=0, description="0-based order within the unit")
    surface: str = Field(description="Surface form from original_text")
    normalized: str = Field(description="Normalized form for matching")
    char_start: int | None = Field(
        default=None,
        ge=0,
        description="Inclusive UTF-8 code-point offset into original_text",
    )
    char_end: int | None = Field(
        default=None,
        ge=0,
        description="Exclusive UTF-8 code-point offset into original_text",
    )
    lemma: str | None = None
    pos: str | None = None
    grammatical_group_id: str | None = Field(
        default=None,
        description="Optional parent morphological/syntactic group id",
    )

    @model_validator(mode="after")
    def _char_span_consistent(self) -> Self:
        start, end = self.char_start, self.char_end
        if (start is None) ^ (end is None):
            msg = "char_start and char_end must both be set or both omitted"
            raise ValueError(msg)
        if start is not None and end is not None and end < start:
            msg = "char_end must be >= char_start"
            raise ValueError(msg)
        return self


class Span(BaseModel):
    """Ordered token coverage within one text unit (may be empty or gappy)."""

    id: str = Field(description="Stable span id (see docs/id-scheme.md)")
    source_id: str
    unit_id: str
    kind: SpanKind = SpanKind.CONTIGUOUS
    token_ids: list[str] = Field(
        default_factory=list,
        description="Ordered token ids; empty only when kind is empty",
    )
    exclusive_group: str | None = Field(
        default=None,
        description=(
            "If set, spans that share this group and unit_id must not overlap "
            "in token membership (illegal overlap)."
        ),
    )
    notes: str = ""

    @model_validator(mode="after")
    def _kind_matches_tokens(self) -> Self:
        if self.kind is SpanKind.EMPTY:
            if self.token_ids:
                msg = "empty spans must not list token_ids"
                raise ValueError(msg)
            return self
        if not self.token_ids:
            msg = f"{self.kind} spans require at least one token_id"
            raise ValueError(msg)
        if len(set(self.token_ids)) != len(self.token_ids):
            msg = "span token_ids must be unique"
            raise ValueError(msg)
        return self


class Parallel(BaseModel):
    """Span-first alignment record (phrase/lexical/verse); many-to-many OK."""

    id: str = Field(description="Stable parallel id (see docs/id-scheme.md)")
    left_span_ids: list[str] = Field(
        description="Usually Coptic / Sahidic / Thomas side",
    )
    right_span_ids: list[str] = Field(
        description="Usually Greek side (Vaticanus B* for NT pilot)",
    )
    relation: ParallelRelation = ParallelRelation.PHRASE
    agreement: AgreementDegree = AgreementDegree.UNCERTAIN
    provenance: str = Field(
        default="",
        description="Human or tool provenance (never invent ancient citations)",
    )
    notes: str = ""

    @model_validator(mode="after")
    def _non_empty_sides_unless_shared_absence(self) -> Self:
        if self.agreement is AgreementDegree.SHARED_ABSENCE:
            return self
        if not self.left_span_ids and not self.right_span_ids:
            msg = "parallel needs at least one left or right span_id"
            raise ValueError(msg)
        return self


class AlignmentDocument(BaseModel):
    """Self-contained units + tokens + spans + parallels for offline validation.

    Verse-level NT pairing remains available via ``PilotCorpus`` /
    ``ParallelUnit``. This document adds token/span IDs and phrase-level
    ``Parallel`` rows without requiring string matching at link time.
    """

    id: str
    title: str
    id_scheme_version: str = Field(
        default=ID_SCHEME_VERSION,
        description="Must match docs/id-scheme.md; bump + rewrite on retokenize",
    )
    text_units: list[TextUnit]
    tokens: list[Token] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    parallels: list[Parallel] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _referential_integrity(self) -> Self:
        if self.id_scheme_version != ID_SCHEME_VERSION:
            msg = (
                f"id_scheme_version {self.id_scheme_version!r} does not match "
                f"code scheme {ID_SCHEME_VERSION!r}; migrate per docs/id-scheme.md"
            )
            raise ValueError(msg)

        unit_keys = {(u.source_id, u.unit_id) for u in self.text_units}
        if len(unit_keys) != len(self.text_units):
            msg = "duplicate TextUnit (source_id, unit_id) pairs"
            raise ValueError(msg)

        token_by_id: dict[str, Token] = {}
        seq_seen: set[tuple[str, str, int]] = set()
        for token in self.tokens:
            key = (token.source_id, token.unit_id)
            if key not in unit_keys:
                msg = (
                    f"dangling token {token.id}: unknown unit "
                    f"{token.source_id}/{token.unit_id}"
                )
                raise ValueError(msg)
            if token.id in token_by_id:
                msg = f"duplicate token id: {token.id}"
                raise ValueError(msg)
            seq_key = (token.source_id, token.unit_id, token.sequence)
            if seq_key in seq_seen:
                msg = (
                    f"duplicate token sequence {token.sequence} in "
                    f"{token.source_id}/{token.unit_id}"
                )
                raise ValueError(msg)
            seq_seen.add(seq_key)
            token_by_id[token.id] = token

        span_by_id: dict[str, Span] = {}
        # exclusive_group -> unit_key -> list of token-id sets
        exclusive: dict[tuple[str, str, str], list[set[str]]] = {}
        for span in self.spans:
            key = (span.source_id, span.unit_id)
            if key not in unit_keys:
                msg = (
                    f"dangling span {span.id}: unknown unit "
                    f"{span.source_id}/{span.unit_id}"
                )
                raise ValueError(msg)
            if span.id in span_by_id:
                msg = f"duplicate span id: {span.id}"
                raise ValueError(msg)
            span_by_id[span.id] = span

            sequences: list[int] = []
            for tid in span.token_ids:
                token = token_by_id.get(tid)
                if token is None:
                    msg = f"dangling token id {tid} in span {span.id}"
                    raise ValueError(msg)
                if token.source_id != span.source_id or token.unit_id != span.unit_id:
                    msg = (
                        f"token {tid} in span {span.id} belongs to "
                        f"{token.source_id}/{token.unit_id}, not "
                        f"{span.source_id}/{span.unit_id}"
                    )
                    raise ValueError(msg)
                sequences.append(token.sequence)

            if span.kind is SpanKind.CONTIGUOUS and sequences:
                lo, hi = min(sequences), max(sequences)
                expected = list(range(lo, hi + 1))
                if sorted(sequences) != expected:
                    msg = (
                        f"contiguous span {span.id} has non-contiguous "
                        f"token sequences {sequences}"
                    )
                    raise ValueError(msg)
            if span.kind is SpanKind.DISCONTINUOUS and sequences:
                lo, hi = min(sequences), max(sequences)
                if sorted(sequences) == list(range(lo, hi + 1)):
                    msg = (
                        f"discontinuous span {span.id} has no gap; use kind=contiguous"
                    )
                    raise ValueError(msg)

            if span.exclusive_group is not None:
                ekey = (span.exclusive_group, span.source_id, span.unit_id)
                exclusive.setdefault(ekey, []).append(set(span.token_ids))

        for ekey, groups in exclusive.items():
            for i, left in enumerate(groups):
                for right in groups[i + 1 :]:
                    overlap = left & right
                    if overlap:
                        group, source_id, unit_id = ekey
                        msg = (
                            f"illegal overlap in exclusive_group {group!r} "
                            f"for {source_id}/{unit_id}: {sorted(overlap)}"
                        )
                        raise ValueError(msg)

        parallel_ids: set[str] = set()
        for parallel in self.parallels:
            if parallel.id in parallel_ids:
                msg = f"duplicate parallel id: {parallel.id}"
                raise ValueError(msg)
            parallel_ids.add(parallel.id)
            for side, ids in (
                ("left", parallel.left_span_ids),
                ("right", parallel.right_span_ids),
            ):
                for sid in ids:
                    if sid not in span_by_id:
                        msg = f"dangling {side} span id {sid} in parallel {parallel.id}"
                        raise ValueError(msg)

        return self

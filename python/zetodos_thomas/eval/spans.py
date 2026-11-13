# SPDX-License-Identifier: Apache-2.0
"""Blinded-reconstruction span minting + overlap resolution (JAK-40).

Horner NT uses the same ``bound-group-v1`` / v2 token path as JAK-36/37/38
training. Dilley Thomas uses a documented whitespace-normalized path (unit-level
ingest has no SCRIPTORIUM per-token orig↔norm pairs yet).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from zetodos_thomas.align.paratext import filter_alignable_tokens
from zetodos_thomas.align.segment_coptic import (
    SEGMENTED_ID_SCHEME_VERSION,
    segmented_coptic_tokens,
)
from zetodos_thomas.schemas.alignment import Token
from zetodos_thomas.schemas.decode import DecodeSpanInput
from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION
from zetodos_thomas.schemas.ids import span_id as mint_v1_span_id
from zetodos_thomas.schemas.ids import token_id as mint_v1_token_id
from zetodos_thomas.schemas.source import TextUnit

DILLEY_SOURCE_ID = "dilley-thomas"
HORNER_SOURCE_ID = "copsahhorner"
DILLEY_SEGMENTATION = "dilley-normalized-whitespace-v1"
HORNER_SEGMENTATION = "bound-group-v1"


@dataclass(frozen=True, slots=True)
class SourceTokenStream:
    """Ordered Coptic tokens ready for span windowing."""

    source_id: str
    unit_id: str
    segmentation: str
    id_scheme_version: str
    tokens: list[Token]


def dilley_normalized_tokens(unit: TextUnit) -> SourceTokenStream:
    """Mint Dilley tokens from ``normalized_text`` whitespace (documented path).

    Dilley unit-level ingest is not orig↔norm zip-ready; offsets index
    ``normalized_text``. Source id stays ``dilley-thomas`` (never decode-input).
    """
    if unit.source_id != DILLEY_SOURCE_ID:
        msg = f"dilley_normalized_tokens requires dilley-thomas, got {unit.source_id!r}"
        raise ValueError(msg)
    tokens: list[Token] = []
    text = unit.normalized_text
    sequence = 0
    for match in re.finditer(r"\S+", text):
        surface = match.group(0)
        tokens.append(
            Token(
                id=mint_v1_token_id(DILLEY_SOURCE_ID, unit.unit_id, sequence),
                source_id=DILLEY_SOURCE_ID,
                unit_id=unit.unit_id,
                sequence=sequence,
                surface=surface,
                normalized=surface,
                char_start=match.start(),
                char_end=match.end(),
            )
        )
        sequence += 1
    return SourceTokenStream(
        source_id=DILLEY_SOURCE_ID,
        unit_id=unit.unit_id,
        segmentation=DILLEY_SEGMENTATION,
        id_scheme_version=ID_SCHEME_VERSION,
        tokens=tokens,
    )


def horner_bound_group_tokens(unit: TextUnit) -> SourceTokenStream:
    """Mint Horner tokens with train-parity ``bound-group-v1`` segmentation."""
    if unit.source_id != HORNER_SOURCE_ID:
        msg = f"horner_bound_group_tokens requires copsahhorner, got {unit.source_id!r}"
        raise ValueError(msg)
    tokens = filter_alignable_tokens(segmented_coptic_tokens(unit))
    return SourceTokenStream(
        source_id=HORNER_SOURCE_ID,
        unit_id=unit.unit_id,
        segmentation=HORNER_SEGMENTATION,
        id_scheme_version=SEGMENTED_ID_SCHEME_VERSION,
        tokens=tokens,
    )


def mint_span_id(scheme_version: str, source_id: str, unit_id: str, local: str) -> str:
    """Mint ``vN:{source}/{unit}/s/{local}`` for scheme N (1 or 2)."""
    if scheme_version == ID_SCHEME_VERSION:
        return mint_v1_span_id(source_id, unit_id, local)
    local_clean = local.strip()
    if not local_clean or "/" in local_clean or " " in local_clean:
        msg = "span local_id must be a non-empty slug without spaces or '/'"
        raise ValueError(msg)
    return f"v{scheme_version}:{source_id}/{unit_id}/s/{local_clean}"


@dataclass(frozen=True, slots=True)
class CandidateWindow:
    """One contiguous window over the source token stream."""

    start: int
    end: int  # exclusive
    surfaces: tuple[str, ...]
    token_ids: tuple[str, ...]
    construction_id: str | None
    kind: str  # "construction" | "singleton"


def find_construction_windows(
    stream: SourceTokenStream,
    patterns: dict[tuple[str, ...], str],
) -> list[CandidateWindow]:
    """All exact Coptic-pattern windows present in ``stream``."""
    norms = [t.normalized for t in stream.tokens]
    out: list[CandidateWindow] = []
    if not norms:
        return out
    max_n = max(len(p) for p in patterns) if patterns else 1
    for n in range(max_n, 0, -1):
        for i in range(0, len(norms) - n + 1):
            key = tuple(norms[i : i + n])
            cid = patterns.get(key)
            if cid is None and n > 1:
                continue
            if n == 1:
                # Singletons always exist; construction_id only when pattern hits.
                out.append(
                    CandidateWindow(
                        start=i,
                        end=i + 1,
                        surfaces=key,
                        token_ids=tuple(t.id for t in stream.tokens[i : i + 1]),
                        construction_id=cid,
                        kind="construction" if cid else "singleton",
                    )
                )
            elif cid is not None:
                out.append(
                    CandidateWindow(
                        start=i,
                        end=i + n,
                        surfaces=key,
                        token_ids=tuple(t.id for t in stream.tokens[i : i + n]),
                        construction_id=cid,
                        kind="construction",
                    )
                )
    return out


def resolve_covering_spans(
    stream: SourceTokenStream,
    patterns: dict[tuple[str, ...], str],
) -> list[CandidateWindow]:
    """Select a non-overlapping cover: prefer longer construction matches.

    Every source token appears in exactly one selected window. Multi-token
    JAK-38 constructions win over singletons when they match; overlapping
    constructions keep the leftmost-longest winner.
    """
    n = len(stream.tokens)
    if n == 0:
        return []
    windows = find_construction_windows(stream, patterns)
    multi = [w for w in windows if w.end - w.start > 1]
    # Leftmost-longest: sort by start asc, length desc.
    multi.sort(key=lambda w: (w.start, -(w.end - w.start), w.construction_id or ""))
    chosen: list[CandidateWindow] = []
    covered = [False] * n
    for w in multi:
        if any(covered[i] for i in range(w.start, w.end)):
            continue
        chosen.append(w)
        for i in range(w.start, w.end):
            covered[i] = True
    # Fill gaps with singletons (carry construction_id when a 1-token pattern hits).
    singles = {w.start: w for w in windows if w.end - w.start == 1}
    for i in range(n):
        if covered[i]:
            continue
        if i in singles:
            chosen.append(singles[i])
        else:
            tok = stream.tokens[i]
            chosen.append(
                CandidateWindow(
                    start=i,
                    end=i + 1,
                    surfaces=(tok.normalized,),
                    token_ids=(tok.id,),
                    construction_id=None,
                    kind="singleton",
                )
            )
        covered[i] = True
    chosen.sort(key=lambda w: w.start)
    return chosen


def windows_to_decode_spans(
    stream: SourceTokenStream,
    windows: list[CandidateWindow],
) -> list[DecodeSpanInput]:
    """Convert covering windows into JAK-39 ``DecodeSpanInput`` rows."""
    spans: list[DecodeSpanInput] = []
    for idx, window in enumerate(windows):
        if window.end > window.start + 1:
            local = f"w{window.start}-{window.end - 1}"
        else:
            local = f"t{window.start}"
        # Keep local unique even if windows somehow collide.
        local = f"{local}-{idx}"
        spans.append(
            DecodeSpanInput(
                span_id=mint_span_id(
                    stream.id_scheme_version,
                    stream.source_id,
                    stream.unit_id,
                    local,
                ),
                source_id=stream.source_id,
                unit_id=stream.unit_id,
                token_ids=list(window.token_ids),
                surfaces=list(window.surfaces),
                construction_id=window.construction_id,
            )
        )
    return spans


def construction_patterns_from_model(
    constructions_by_pattern: dict[tuple[str, ...], list[object]],
) -> dict[tuple[str, ...], str]:
    """Map Coptic pattern → first construction_id (stable lexicographic)."""
    out: dict[tuple[str, ...], str] = {}
    for pattern, rows in constructions_by_pattern.items():
        ids = sorted(getattr(row, "construction_id") for row in rows)  # noqa: B009
        if ids:
            out[pattern] = ids[0]
    return out

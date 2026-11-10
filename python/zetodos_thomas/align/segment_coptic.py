# SPDX-License-Identifier: Apache-2.0
"""Rule-based Sahidic bound-group segmentation for Horner NT alignment.

Horner CopSahHorner surfaces keep many proclitics fused (e.g. ``ⲛϭⲓⲡϣⲁϫⲉ``).
Whitespace tokenization alone is insufficient for statistical word alignment
(JAK-43 → JAK-36). This module peels common Sahidic prefixes from
**normalized** Coptic (NFC, overlines already stripped by
``normalize_coptic``), then maps piece boundaries back onto
``original_text`` for ``Token.surface`` / ``char_*`` (schema contract).

Scope and honesty:
- Designed for ``copsahhorner`` / Horner NT, not Dilley SCRIPTORIUM Thomas.
  Dilley already ships morpheme-oriented ``normalized_text``; mint tokens from
  SCRIPTORIUM per-token orig↔norm pairs instead of this peeler.
- Rule-based / lexicon-light — not a full morphological analyzer.
- Over-segmentation and under-segmentation both remain possible; review
  samples before treating boundaries as gold.
- Token ids use scheme **v2** (``SEGMENTED_ID_SCHEME_VERSION``) so they never
  collide with whitespace **v1** gold / AlignmentDocument ids
  (``docs/id-scheme.md`` retokenization rule).
"""

from __future__ import annotations

import re

from zetodos_thomas.ingest.normalize import (
    map_coptic_normalized_span_to_original,
    normalize_coptic,
)
from zetodos_thomas.schemas.alignment import Token
from zetodos_thomas.schemas.source import TextUnit

SEGMENTATION_VERSION = "bound-group-v1"
# Distinct from whitespace v1 minting; see docs/id-scheme.md.
SEGMENTED_ID_SCHEME_VERSION = "2"

_WHITESPACE_RE = re.compile(r"\S+")

# Longest-first Sahidic proclitic / converter / article+prep prefixes.
# Ordered to prefer longer matches. Exact closed forms listed here stay
# atomic when they appear as the whole remainder (see peel_bound_group).
_PREFIXES: tuple[str, ...] = (
    "ⲛϭⲓ",
    "ⲉⲧⲉⲣⲉ",
    "ⲛⲧⲉⲣⲉ",
    "ⲛⲧⲁⲣⲉ",
    "ⲉⲧⲁⲣⲉ",
    "ⲛⲧⲉⲣ",
    "ϩⲛⲧⲉ",
    "ϩⲙⲡⲉ",
    "ϩⲛⲧⲛ",
    "ϩⲙⲡⲛ",
    "ⲙⲡⲉⲛ",
    "ⲙⲡⲉⲧ",
    "ⲙⲡⲉϥ",
    "ⲙⲡⲉⲥ",
    "ⲙⲡⲉⲩ",
    "ⲙⲡⲉⲓ",
    "ⲛⲧⲉϥ",
    "ⲛⲧⲉⲥ",
    "ⲛⲧⲉⲩ",
    "ⲛⲧⲉⲓ",
    "ⲛⲧⲉⲛ",
    "ⲛⲧⲉⲧ",
    "ⲡⲉϥ",
    "ⲡⲉⲥ",
    "ⲡⲉⲩ",
    "ⲡⲉⲛ",
    "ⲡⲉⲧ",
    "ⲡⲉⲓ",
    "ⲧⲉϥ",
    "ⲧⲉⲥ",
    "ⲧⲉⲩ",
    "ⲧⲉⲛ",
    "ⲧⲉⲧ",
    "ⲧⲉⲓ",
    "ⲛⲉϥ",
    "ⲛⲉⲥ",
    "ⲛⲉⲩ",
    "ⲛⲉⲛ",
    "ⲛⲉⲧ",
    "ⲛⲉⲓ",
    "ⲉⲡⲉϥ",
    "ⲉⲧⲉϥ",
    "ⲉⲡⲉⲥ",
    "ⲉⲧⲉⲥ",
    "ⲉⲡⲉⲩ",
    "ⲉⲧⲉⲩ",
    "ϩⲙⲡ",
    "ϩⲛⲧ",
    "ϩⲛⲛ",
    "ϩⲓⲧⲛ",
    "ϩⲓⲧⲙ",
    "ⲁϫⲛ",
    "ⲉϫⲛ",
    "ⲉϫⲙ",
    "ⲙⲡⲉ",
    "ⲙⲡⲛ",
    "ⲙⲡⲧ",
    "ⲛⲧⲉ",
    "ⲛⲧⲁ",
    "ⲛⲧⲟ",
    "ⲉⲧⲉ",
    "ⲉⲡⲉ",
    "ⲉⲡⲛ",
    "ⲉⲧⲛ",
    "ⲉⲩⲛ",
    "ⲛⲁϥ",
    "ⲛⲁⲥ",
    "ⲛⲁⲩ",
    "ⲛⲁⲓ",
    "ⲛⲁⲛ",
    "ⲙⲡ",
    "ⲙⲧ",
    "ⲛⲡ",
    "ⲛⲧ",
    "ⲛⲛ",
    "ϩⲛ",
    "ϩⲙ",
    "ϩⲓ",
    # Avoid bare ⲁⲩ/ⲉϥ-class verbal prefixes: they over-segment closed forms
    # such as ⲁⲩⲱ. Prefer longer converter/possessive forms above.
    "ⲉⲡ",
    "ⲉⲧ",
    "ⲉⲛ",
    "ⲟⲩ",
    "ⲡⲉ",
    "ⲧⲉ",
    "ⲛⲉ",
    "ⲡ",
    "ⲧ",
    "ⲛ",
)

_CLOSED_FORMS = frozenset(_PREFIXES)


def segmented_token_id(source_id: str, unit_id: str, sequence: int) -> str:
    """Mint a v2 token id for the bound-group segmentation layer."""
    if sequence < 0:
        msg = "token sequence must be >= 0"
        raise ValueError(msg)
    return f"v{SEGMENTED_ID_SCHEME_VERSION}:{source_id}/{unit_id}/t/{sequence}"


def peel_bound_group(token: str) -> list[str]:
    """Greedily peel known prefixes from one whitespace token.

    Returns one or more pieces. When the whole remainder is a known closed
    form (exact prefix entry), it is emitted atomically and peeling stops —
    so ``ⲡⲉ`` / ``ⲛϭⲓ`` stay whole rather than falling through to ``ⲡ``+``ⲉ``.
    """
    if not token:
        return []
    pieces: list[str] = []
    rest = token
    while rest:
        if rest in _CLOSED_FORMS:
            pieces.append(rest)
            break
        matched = False
        for prefix in _PREFIXES:
            if rest.startswith(prefix) and len(rest) > len(prefix):
                pieces.append(prefix)
                rest = rest[len(prefix) :]
                matched = True
                break
        if not matched:
            pieces.append(rest)
            break
    return pieces


def segment_normalized_coptic(text: str) -> list[str]:
    """Whitespace-split then peel bound groups on normalized Coptic text."""
    out: list[str] = []
    for match in _WHITESPACE_RE.finditer(text):
        out.extend(peel_bound_group(match.group(0)))
    return out


def segmented_coptic_tokens(unit: TextUnit) -> list[Token]:
    """Mint aligner tokens from bound-group splits with original-text spans.

    ``Token.surface`` / ``char_start`` / ``char_end`` index into
    ``unit.original_text``. ``Token.normalized`` is the segmented normalized
    piece. Ids use scheme **v2** (``segmented_token_id``).
    """
    if unit.source_id not in {"copsahhorner", "dilley-thomas"}:
        msg = (
            f"segmented_coptic_tokens expected a Coptic source; got {unit.source_id!r}"
        )
        raise ValueError(msg)
    if unit.source_id == "dilley-thomas":
        msg = (
            "segmented_coptic_tokens is for Horner bound groups; "
            "Dilley/SCRIPTORIUM must use per-token orig↔norm pairs"
        )
        raise ValueError(msg)

    text = normalize_coptic(unit.original_text)
    tokens: list[Token] = []
    sequence = 0
    for match in _WHITESPACE_RE.finditer(text):
        group = match.group(0)
        start = match.start()
        pieces = peel_bound_group(group)
        offset = 0
        for piece in pieces:
            local = group.find(piece, offset)
            if local < 0:
                msg = (
                    f"internal segmentation inconsistency in {unit.unit_id}: "
                    f"piece {piece!r} not found in {group!r} after {offset}"
                )
                raise ValueError(msg)
            abs_start = start + local
            abs_end = abs_start + len(piece)
            o_start, o_end = map_coptic_normalized_span_to_original(
                unit.original_text, abs_start, abs_end
            )
            surface = unit.original_text[o_start:o_end]
            tokens.append(
                Token(
                    id=segmented_token_id(unit.source_id, unit.unit_id, sequence),
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    sequence=sequence,
                    surface=surface,
                    normalized=piece,
                    char_start=o_start,
                    char_end=o_end,
                )
            )
            sequence += 1
            offset = local + len(piece)
    return tokens


def segmentation_stats(unit: TextUnit) -> dict[str, int]:
    """Compare whitespace vs bound-group token counts for one unit."""
    ws = len(_WHITESPACE_RE.findall(unit.normalized_text))
    seg = len(segment_normalized_coptic(unit.normalized_text))
    return {
        "whitespace_tokens": ws,
        "segmented_tokens": seg,
        "extra_segments": seg - ws,
    }

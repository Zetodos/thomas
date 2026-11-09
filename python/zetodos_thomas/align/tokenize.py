# SPDX-License-Identifier: Apache-2.0
"""Whitespace tokenization helpers for stable token minting (schema fixtures).

This is intentionally dumb: Phase 1 uses space-separated surfaces from Horner
and Vaticanus diplomatic text. It is **not** a morphological analyzer and does
not train or invoke eflomal.
"""

from __future__ import annotations

import re

from zetodos_thomas.schemas.alignment import Token
from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.source import TextUnit

_TOKEN_RE = re.compile(r"\S+")


def whitespace_tokens(
    unit: TextUnit,
    *,
    use_normalized_surface: bool = False,
) -> list[Token]:
    """Mint ordered tokens from a text unit's surface string.

    Offsets always refer to ``unit.original_text`` when
    ``use_normalized_surface`` is false (default). When true, ``surface`` and
    offsets refer to ``normalized_text`` (useful when originals are empty but
    a normalized placeholder exists — rare for the NT pilot).
    """
    text = unit.normalized_text if use_normalized_surface else unit.original_text
    norm_source = unit.normalized_text
    norm_parts = _TOKEN_RE.findall(norm_source) if norm_source else []
    tokens: list[Token] = []
    for sequence, match in enumerate(_TOKEN_RE.finditer(text)):
        surface = match.group(0)
        normalized = norm_parts[sequence] if sequence < len(norm_parts) else surface
        tokens.append(
            Token(
                id=token_id(unit.source_id, unit.unit_id, sequence),
                source_id=unit.source_id,
                unit_id=unit.unit_id,
                sequence=sequence,
                surface=surface,
                normalized=normalized,
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return tokens

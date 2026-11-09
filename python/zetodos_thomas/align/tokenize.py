# SPDX-License-Identifier: Apache-2.0
"""Whitespace tokenization helpers for stable token minting (schema fixtures).

This helper is for sources where **whitespace token counts match** between
``original_text`` and ``normalized_text`` (Horner / Vaticanus B* pilot
surfaces). It is **not** valid for Dilley / SCRIPTORIUM Thomas: there
``original_text`` is a continuous concatenation of ``orig`` surfaces while
``normalized_text`` is independently morpheme-segmented. SCRIPTORIUM ingest
must mint tokens from per-token orig↔norm pairs (JAK-30), not this zip.
"""

from __future__ import annotations

import re

from zetodos_thomas.schemas.alignment import Token
from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.source import TextUnit

_TOKEN_RE = re.compile(r"\S+")


class TokenCountMismatchError(ValueError):
    """Raised when original and normalized whitespace token counts differ."""


def whitespace_token_counts(unit: TextUnit) -> tuple[int, int]:
    """Return ``(original_count, normalized_count)`` after whitespace split."""
    original_n = len(_TOKEN_RE.findall(unit.original_text))
    normalized_n = len(_TOKEN_RE.findall(unit.normalized_text))
    return original_n, normalized_n


def whitespace_tokens(unit: TextUnit) -> list[Token]:
    """Mint ordered tokens by whitespace-zipping original and normalized text.

    ``Token.char_start`` / ``char_end`` are always offsets into
    ``unit.original_text``. ``normalized`` comes from the same-index whitespace
    token in ``normalized_text``.

    Raises ``TokenCountMismatchError`` when the two layers disagree on token
    count (the Dilley/SCRIPTORIUM case). Empty units (both layers empty) yield
    an empty token list.
    """
    original_parts = list(_TOKEN_RE.finditer(unit.original_text))
    norm_parts = _TOKEN_RE.findall(unit.normalized_text)
    if len(original_parts) != len(norm_parts):
        msg = (
            f"whitespace token count mismatch for {unit.source_id}/{unit.unit_id}: "
            f"original={len(original_parts)} normalized={len(norm_parts)}. "
            "whitespace_tokens only supports matching counts (Horner/Vaticanus). "
            "Dilley/SCRIPTORIUM must use per-token orig↔norm pairs, not positional zip."
        )
        raise TokenCountMismatchError(msg)

    tokens: list[Token] = []
    for sequence, match in enumerate(original_parts):
        tokens.append(
            Token(
                id=token_id(unit.source_id, unit.unit_id, sequence),
                source_id=unit.source_id,
                unit_id=unit.unit_id,
                sequence=sequence,
                surface=match.group(0),
                normalized=norm_parts[sequence],
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return tokens

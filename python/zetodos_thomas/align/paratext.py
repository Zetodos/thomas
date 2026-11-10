# SPDX-License-Identifier: Apache-2.0
"""Paratext / editorial-mark separation helpers for alignment scoring."""

from __future__ import annotations

import re

from zetodos_thomas.schemas.alignment import Token
from zetodos_thomas.schemas.source import TextUnit

# Horner Mark 1:1 begins with the book heading before the verse proper.
MARK_1_1_HEADING = "ⲙⲁⲣⲕⲟⲥ"

_EDITORIAL_CHAR_RE = re.compile(r"[\[\]⟨⟩<>]")


def is_mark_1_1_heading_token(token: Token) -> bool:
    """True when this Sahidic Mark 1:1 token is the book-heading paratext."""
    if token.source_id != "copsahhorner" or token.unit_id != "mark.1.1":
        return False
    return token.sequence == 0 and token.normalized == MARK_1_1_HEADING


def unit_has_mark_heading(unit: TextUnit) -> bool:
    """Detect Horner Mark 1:1 book heading in original/normalized text."""
    if unit.source_id != "copsahhorner" or unit.unit_id != "mark.1.1":
        return False
    first = unit.normalized_text.split()[:1]
    return bool(first) and first[0] == MARK_1_1_HEADING


def has_editorial_marks(text: str) -> bool:
    """True if diplomatic text still carries restoration/uncertainty wrappers."""
    return bool(_EDITORIAL_CHAR_RE.search(text))


def filter_alignable_tokens(tokens: list[Token]) -> list[Token]:
    """Drop known paratext tokens that must not score as lexical alignments."""
    return [t for t in tokens if not is_mark_1_1_heading_token(t)]

# SPDX-License-Identifier: Apache-2.0
"""Project bound-group (v2) tokens onto parent whitespace (v1) via char spans."""

from __future__ import annotations

from zetodos_thomas.schemas.alignment import Token


def parent_whitespace_token(
    segmented: Token,
    whitespace: list[Token],
) -> Token | None:
    """Return the v1 whitespace token whose original span contains ``segmented``.

    Uses inclusive/exclusive ``char_start`` / ``char_end`` on ``original_text``.
    Returns None when spans are missing or no parent contains the piece.
    """
    if segmented.char_start is None or segmented.char_end is None:
        return None
    for parent in whitespace:
        if parent.char_start is None or parent.char_end is None:
            continue
        if (
            parent.char_start <= segmented.char_start
            and segmented.char_end <= parent.char_end
        ):
            return parent
    return None


def project_segmented_ids_to_parent_v1(
    segmented: list[Token],
    whitespace: list[Token],
) -> list[str | None]:
    """Map each segmented token to its parent v1 token id (or None)."""
    return [
        (parent.id if (parent := parent_whitespace_token(seg, whitespace)) else None)
        for seg in segmented
    ]

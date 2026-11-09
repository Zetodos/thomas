# SPDX-License-Identifier: Apache-2.0
"""Lightweight original→normalized surface forms for pilot ingest."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
# Coptic combining overline / character binders often used for nomina sacra.
_COPTIC_COMBINING = "".join(
    (
        "\u0304",  # combining macron
        "\u0305",  # combining overline
        "\u035e",  # combining double macron
        "\ufe26",  # combining conjoining macron
    )
)
_COPTIC_STRIP_RE = re.compile(f"[{re.escape(_COPTIC_COMBINING)}\u0361]")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_coptic(text: str) -> str:
    """NFC + whitespace collapse; strip combining overlines for matching.

    Original display text (with overlines / binders) is preserved separately.
    Normalized form is for machine-oriented token comparison only.
    """
    nfc = unicodedata.normalize("NFC", text)
    stripped = _COPTIC_STRIP_RE.sub("", nfc)
    return normalize_whitespace(stripped)


def normalize_greek(text: str) -> str:
    """NFC + whitespace collapse for unaccented Robinson WH surfaces."""
    nfc = unicodedata.normalize("NFC", text)
    return normalize_whitespace(nfc)

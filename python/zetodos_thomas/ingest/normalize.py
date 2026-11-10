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
# Diplomatic editorial wrappers: keep letters, drop brackets/angles.
_EDITORIAL_WRAP_RE = re.compile(r"[\[\]⟨⟩<>]")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_editorial_marks(text: str) -> str:
    """Remove diplomatic uncertainty/restoration wrappers; keep letters.

    Used so normalized matching is not polluted by ``[η]``, ``⟨…⟩``, or
    Horner ``<…>`` restoration notation. Original text keeps the marks.
    """
    return _EDITORIAL_WRAP_RE.sub("", text)


def normalize_coptic(text: str) -> str:
    """NFC + whitespace collapse; strip combining overlines for matching.

    Original display text (with overlines / binders) is preserved separately.
    Normalized form is for machine-oriented token comparison only.
    """
    nfc = unicodedata.normalize("NFC", text)
    stripped = _COPTIC_STRIP_RE.sub("", nfc)
    stripped = strip_editorial_marks(stripped)
    return normalize_whitespace(stripped)


def normalize_greek(text: str) -> str:
    """NFC + whitespace collapse; strip editorial wrappers for matching."""
    nfc = unicodedata.normalize("NFC", text)
    stripped = strip_editorial_marks(nfc)
    return normalize_whitespace(stripped)


def recompute_normalized(*, language: str, original_text: str) -> str:
    """Deterministically recompute ``normalized_text`` from ``original_text``.

    ``language`` is ``coptic`` / ``sahidic`` / ``greek`` (case-insensitive).
    """
    key = language.strip().lower()
    if key in {"coptic", "sahidic", "cop", "sah"}:
        return normalize_coptic(original_text)
    if key in {"greek", "grc"}:
        return normalize_greek(original_text)
    msg = f"unsupported language for normalization: {language!r}"
    raise ValueError(msg)

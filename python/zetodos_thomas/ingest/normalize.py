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


def map_coptic_normalized_span_to_original(
    original: str, norm_start: int, norm_end: int
) -> tuple[int, int]:
    """Map a ``normalize_coptic`` exclusive span back onto ``original``.

    Returns ``(char_start, char_end)`` exclusive offsets into ``original``
    suitable for ``Token.char_start`` / ``Token.char_end``. Combining
    overlines and editorial wrappers stripped by normalization are skipped
    in the normalized index but still owned by neighboring kept letters when
    they sit inside a peeled piece's original span.
    """
    orig_chars = list(original)
    kept: list[tuple[str, int, int]] = []
    i = 0
    while i < len(orig_chars):
        j = i + 1
        while j < len(orig_chars) and unicodedata.combining(orig_chars[j]):
            j += 1
        cluster = "".join(orig_chars[i:j])
        nfc_cluster = unicodedata.normalize("NFC", cluster)
        for k, nfc_ch in enumerate(nfc_cluster):
            if len(nfc_cluster) == 1:
                o_start, o_end = i, j
            else:
                o_start = i + k
                o_end = min(j, o_start + 1)
            if _COPTIC_STRIP_RE.match(nfc_ch) or _EDITORIAL_WRAP_RE.match(nfc_ch):
                continue
            kept.append((nfc_ch, o_start, o_end))
        i = j

    spans: list[tuple[int, int]] = []
    chars: list[str] = []
    pending_ws: tuple[int, int] | None = None
    for ch, o_start, o_end in kept:
        if ch.isspace():
            if pending_ws is None:
                pending_ws = (o_start, o_end)
            else:
                pending_ws = (pending_ws[0], o_end)
            continue
        if pending_ws is not None:
            if chars:
                chars.append(" ")
                spans.append(pending_ws)
            pending_ws = None
        chars.append(ch)
        spans.append((o_start, o_end))
    normalized = "".join(chars).strip()
    expected = normalize_coptic(original)
    if normalized != expected:
        msg = (
            f"normalize alignment mismatch: built={normalized!r} expected={expected!r}"
        )
        raise ValueError(msg)
    if len(spans) != len(normalized):
        msg = (
            f"span/char length mismatch: spans={len(spans)} "
            f"normalized={len(normalized)}"
        )
        raise ValueError(msg)
    if norm_start < 0 or norm_end < norm_start or norm_end > len(spans):
        msg = f"normalized span [{norm_start}:{norm_end}] out of range"
        raise ValueError(msg)
    if norm_start == norm_end:
        if norm_start < len(spans):
            pin = spans[norm_start][0]
            return pin, pin
        if spans:
            pin = spans[-1][1]
            return pin, pin
        return 0, 0
    return spans[norm_start][0], spans[norm_end - 1][1]

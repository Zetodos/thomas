# SPDX-License-Identifier: Apache-2.0
"""Parse Maurice Robinson Westcott-Hort text-only beta-code files."""

from __future__ import annotations

import re
from pathlib import Path

from zetodos_thomas.ingest.normalize import normalize_greek
from zetodos_thomas.schemas.source import TextUnit

# Robinson textonly convention (y=θ, c=χ, v=final ς). Unaccented.
_BETA_MAP = str.maketrans(
    {
        "a": "α",
        "b": "β",
        "g": "γ",
        "d": "δ",
        "e": "ε",
        "z": "ζ",
        "h": "η",
        "y": "θ",
        "i": "ι",
        "k": "κ",
        "l": "λ",
        "m": "μ",
        "n": "ν",
        "x": "ξ",
        "o": "ο",
        "p": "π",
        "r": "ρ",
        "s": "σ",
        "v": "ς",
        "t": "τ",
        "u": "υ",
        "f": "φ",
        "c": "χ",
        "q": "ψ",
        "w": "ω",
    }
)

# Allow reference-only empty verses (e.g. "     5:4" with no trailing text).
_VERSE_START_RE = re.compile(r"^\s*(\d+):(\d+)(?:\s+(.*))?$")

# Map project book ids → Robinson textonly filenames (stem).
WH_BOOK_FILES: dict[str, str] = {
    "matthew": "MT.WH",
    "mark": "MR.WH",
    "luke": "LU.WH",
    "john": "JOH.WH",
}


def beta_to_unicode(beta: str) -> str:
    """Convert Robinson textonly beta code to Greek Unicode letters."""
    return beta.lower().translate(_BETA_MAP)


def parse_wh_textonly(
    path: Path, *, book: str, source_id: str = "westcott-hort"
) -> list[TextUnit]:
    """Parse a Robinson ``*.WH`` textonly file into verse :class:`TextUnit`s."""
    raw = path.read_text(encoding="utf-8")
    units: list[TextUnit] = []
    current_chapter: int | None = None
    current_verse: int | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal current_chapter, current_verse, chunks
        if current_chapter is None or current_verse is None:
            chunks = []
            return
        beta = " ".join(part.strip() for part in chunks if part.strip())
        original = beta_to_unicode(beta)
        unit_id = f"{book}.{current_chapter}.{current_verse}"
        units.append(
            TextUnit(
                source_id=source_id,
                unit_id=unit_id,
                book=book,
                chapter=current_chapter,
                verse=current_verse,
                original_text=original,
                normalized_text=normalize_greek(original),
                sequence=len(units),
            )
        )
        chunks = []

    for line in raw.splitlines():
        match = _VERSE_START_RE.match(line)
        if match:
            flush()
            current_chapter = int(match.group(1))
            current_verse = int(match.group(2))
            rest = match.group(3)
            chunks = [rest] if rest else []
        elif current_verse is not None:
            chunks.append(line)
    flush()
    return units


def load_wh_verses(
    textonly_dir: Path,
    book: str,
    chapter: int,
    verse_from: int,
    verse_to: int,
    *,
    source_id: str = "westcott-hort",
) -> list[TextUnit]:
    """Load a contiguous verse range from a Robinson textonly book file."""
    filename = WH_BOOK_FILES.get(book)
    if filename is None:
        msg = f"unsupported WH book id: {book}"
        raise KeyError(msg)
    path = textonly_dir / filename
    all_units = parse_wh_textonly(path, book=book, source_id=source_id)
    selected = [
        unit
        for unit in all_units
        if unit.chapter == chapter and verse_from <= unit.verse <= verse_to
    ]
    return selected

# SPDX-License-Identifier: Apache-2.0
"""Read CrossWire CopSahHorner verses via pysword (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from zetodos_thomas.ingest.normalize import normalize_coptic
from zetodos_thomas.schemas.source import TextUnit

# pysword / SWORD English book names for the NT module.
SWORD_BOOK_NAMES: dict[str, str] = {
    "matthew": "Matthew",
    "mark": "Mark",
    "luke": "Luke",
    "john": "John",
    "acts": "Acts",
    "romans": "Romans",
    "1corinthians": "I Corinthians",
    "2corinthians": "II Corinthians",
    "galatians": "Galatians",
    "ephesians": "Ephesians",
    "philippians": "Philippians",
    "colossians": "Colossians",
    "1thessalonians": "I Thessalonians",
    "2thessalonians": "II Thessalonians",
    "1timothy": "I Timothy",
    "2timothy": "II Timothy",
    "titus": "Titus",
    "philemon": "Philemon",
    "hebrews": "Hebrews",
    "james": "James",
    "1peter": "I Peter",
    "2peter": "II Peter",
    "1john": "I John",
    "2john": "II John",
    "3john": "III John",
    "jude": "Jude",
    "revelation": "Revelation of John",
}

MODULE_NAME = "CopSahHorner"


class SwordBible(Protocol):
    """Minimal protocol for pysword bible objects used here."""

    def get(
        self,
        books: list[str] | None = None,
        chapters: list[int] | None = None,
        verses: list[int] | None = None,
    ) -> str: ...


def _import_sword_modules() -> type:
    try:
        from pysword.modules import SwordModules
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        msg = (
            "pysword is required to read CopSahHorner modules. "
            "Install with: uv sync --extra sword"
        )
        raise ImportError(msg) from exc
    return SwordModules


def open_module(sword_root: Path) -> SwordBible:
    """Parse a local SWORD tree and return the CopSahHorner bible object."""
    sword_modules_cls = _import_sword_modules()
    modules = sword_modules_cls(str(sword_root))
    found = modules.parse_modules()
    if MODULE_NAME not in found:
        msg = f"{MODULE_NAME} not found under {sword_root}"
        raise FileNotFoundError(msg)
    bible: SwordBible = modules.get_bible_from_module(MODULE_NAME)
    return bible


def read_verse(
    bible: SwordBible,
    book: str,
    chapter: int,
    verse: int,
    *,
    source_id: str = "copsahhorner",
    sequence: int = 0,
) -> TextUnit:
    """Read one verse; empty string when the module has no text for the slot."""
    sword_book = SWORD_BOOK_NAMES.get(book)
    if sword_book is None:
        msg = f"unsupported Coptic book id: {book}"
        raise KeyError(msg)
    raw = bible.get(books=[sword_book], chapters=[chapter], verses=[verse])
    original = raw.strip()
    return TextUnit(
        source_id=source_id,
        unit_id=f"{book}.{chapter}.{verse}",
        book=book,
        chapter=chapter,
        verse=verse,
        original_text=original,
        normalized_text=normalize_coptic(original),
        sequence=sequence,
    )


def load_coptic_verses(
    sword_root: Path,
    book: str,
    chapter: int,
    verse_from: int,
    verse_to: int,
    *,
    source_id: str = "copsahhorner",
) -> list[TextUnit]:
    """Load a contiguous verse range from a local CopSahHorner install."""
    bible = open_module(sword_root)
    units: list[TextUnit] = []
    for verse in range(verse_from, verse_to + 1):
        units.append(
            read_verse(
                bible,
                book,
                chapter,
                verse,
                source_id=source_id,
                sequence=len(units),
            )
        )
    return units

# SPDX-License-Identifier: Apache-2.0
"""Extract Codex Vaticanus B* (firsthand) Greek from NTVMR TEI."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from zetodos_thomas.ingest.normalize import normalize_greek
from zetodos_thomas.schemas.source import TextUnit

TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": TEI_NS}

# OSIS book names in NTVMR ``ab/@n`` → project book ids.
OSIS_BOOK_TO_ID: dict[str, str] = {
    "Matt": "matthew",
    "Mark": "mark",
    "Luke": "luke",
    "John": "john",
    "Acts": "acts",
    "Rom": "romans",
    "1Cor": "1corinthians",
    "2Cor": "2corinthians",
    "Gal": "galatians",
    "Eph": "ephesians",
    "Phil": "philippians",
    "Col": "colossians",
    "1Thess": "1thessalonians",
    "2Thess": "2thessalonians",
    "1Tim": "1timothy",
    "2Tim": "2timothy",
    "Titus": "titus",
    "Phlm": "philemon",
    "Heb": "hebrews",
    "Jas": "james",
    "1Pet": "1peter",
    "2Pet": "2peter",
    "1John": "1john",
    "2John": "2john",
    "3John": "3john",
    "Jude": "jude",
    "Rev": "revelation",
}

SOURCE_ID = "ntvmr-vaticanus-b"
DOC_ID = "20003"
GA_NUM = "03"

_WHITESPACE_RE = re.compile(r"\s+")
_AB_N_RE = re.compile(r"^([A-Za-z0-9]+)\.(\d+)\.(\d+)$")


def _local(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _text_pieces(node: ET.Element, *, mark_unclear: bool) -> list[str]:
    """Collect diplomatic text from a TEI subtree under B* rules."""
    parts: list[str] = []
    name = _local(node.tag)

    if name == "note":
        return parts
    if name == "app":
        # Only original firsthand reading; empty = omission (emit nothing).
        for child in list(node):
            if _local(child.tag) != "rdg":
                continue
            if child.get("type") == "orig" and child.get("hand") == "firsthand":
                parts.extend(_text_pieces(child, mark_unclear=mark_unclear))
            # Ignore corrector readings entirely.
        return parts
    if name == "rdg" and not (
        node.get("type") == "orig" and node.get("hand") == "firsthand"
    ):
        # Corrector (or other) reading reached outside expected walk.
        return parts
    if name == "lb":
        # Soft break inside a word when break="no"; otherwise whitespace.
        if node.get("break") == "no":
            return parts
        parts.append(" ")
        return parts
    if name == "unclear":
        unclear_text = _WHITESPACE_RE.sub("", "".join(node.itertext()))
        if unclear_text:
            parts.append(f"[{unclear_text}]" if mark_unclear else unclear_text)
        return parts

    if node.text:
        parts.append(node.text)

    for child in list(node):
        child_name = _local(child.tag)
        if child_name == "note":
            pass  # skip editorial notes; still keep trailing whitespace below
        elif child_name == "supplied":
            supplied = _WHITESPACE_RE.sub("", "".join(child.itertext()))
            if supplied:
                parts.append(f"⟨{supplied}⟩" if mark_unclear else supplied)
        else:
            parts.extend(_text_pieces(child, mark_unclear=mark_unclear))
        if child.tail:
            parts.append(child.tail)
    return parts


def extract_ab_text(ab: ET.Element, *, mark_unclear: bool = True) -> str:
    """Diplomatic B* string for one ``<ab>`` verse element."""
    raw = "".join(_text_pieces(ab, mark_unclear=mark_unclear))
    # Join word-internal soft breaks already handled; collapse whitespace.
    raw = raw.replace("\n", " ")
    return _WHITESPACE_RE.sub(" ", raw).strip()


def parse_ab_n(n_attr: str) -> tuple[str, int, int] | None:
    """Parse ``John.1.4`` → ``(john, 1, 4)``."""
    match = _AB_N_RE.match(n_attr.strip())
    if match is None:
        return None
    osis_book, chapter_s, verse_s = match.groups()
    book = OSIS_BOOK_TO_ID.get(osis_book)
    if book is None:
        return None
    return book, int(chapter_s), int(verse_s)


def parse_teiraw(
    path: Path,
    *,
    source_id: str = SOURCE_ID,
) -> list[TextUnit]:
    """Parse an NTVMR teiraw XML file into B* :class:`TextUnit`s."""
    tree = ET.parse(path)
    root = tree.getroot()
    units: list[TextUnit] = []
    for ab in root.iter(f"{{{TEI_NS}}}ab"):
        n_attr = ab.get("n")
        if not n_attr:
            continue
        parsed = parse_ab_n(n_attr)
        if parsed is None:
            continue
        book, chapter, verse = parsed
        original = extract_ab_text(ab, mark_unclear=True)
        # Normalized: drop uncertainty/restoration wrappers; keep letters.
        normalized = normalize_greek(original)
        units.append(
            TextUnit(
                source_id=source_id,
                unit_id=f"{book}.{chapter}.{verse}",
                book=book,
                chapter=chapter,
                verse=verse,
                original_text=original,
                normalized_text=normalized,
                sequence=len(units),
            )
        )
    return units


def load_vaticanus_verses(
    tei_paths: list[Path],
    book: str,
    chapter: int,
    verse_from: int,
    verse_to: int,
    *,
    source_id: str = SOURCE_ID,
) -> list[TextUnit]:
    """Load a verse range from one or more teiraw files covering that range."""
    by_verse: dict[int, TextUnit] = {}
    for path in tei_paths:
        for unit in parse_teiraw(path, source_id=source_id):
            if (
                unit.book == book
                and unit.chapter == chapter
                and verse_from <= unit.verse <= verse_to
            ):
                by_verse[unit.verse] = unit
    # Emit contiguous slots; missing TEI abs become empty units.
    units: list[TextUnit] = []
    for verse in range(verse_from, verse_to + 1):
        existing = by_verse.get(verse)
        if existing is not None:
            units.append(existing.model_copy(update={"sequence": len(units)}))
        else:
            units.append(
                TextUnit(
                    source_id=source_id,
                    unit_id=f"{book}.{chapter}.{verse}",
                    book=book,
                    chapter=chapter,
                    verse=verse,
                    original_text="",
                    normalized_text="",
                    sequence=len(units),
                )
            )
    return units


def teiraw_filename(osis_range: str) -> str:
    """Map ``John.1.1-18`` → ``John_1_1-18.teiraw.xml``."""
    return f"{osis_range.replace('.', '_')}.teiraw.xml"

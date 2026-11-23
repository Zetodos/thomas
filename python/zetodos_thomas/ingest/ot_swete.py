# SPDX-License-Identifier: Apache-2.0
"""Ingest SCRIPTORIUM Sahidic OT (.tt with full lemma/POS/morph) and Swete LXX."""

from __future__ import annotations

import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.linguistic import (
    CopticMorpheme,
    CopticUnit,
    GreekMorphFeatures,
    GreekToken,
    GreekUnit,
)
from zetodos_thomas.schemas.source import UnitKind

OT_SOURCE_ID = "scriptorium-sahidic-ot"
SWETE_SOURCE_ID = "swete-lxx"
MAX_TOKEN_RATIO = 4.0

BOOK_TO_SWETE: dict[str, str | tuple[str, ...]] = {
    "Genesis": "01.Genesis.txt",
    "Exodus": "02.Exodus.txt",
    "Leviticus": "03.Leviticus.txt",
    "Numbers": "04.Numeri.txt",
    "Deuteronomy": "05.Deuteronomium.txt",
    "Joshua": "06.Josue.txt",
    "Judges": "08.Judices.txt",
    "Ruth": "10.Ruth.txt",
    "I_Samuel": "11.Regnorum_I.txt",
    "II_Samuel": "12.Regnorum_II.txt",
    "I_Kings": "13.Regnorum_III.txt",
    "II_Kings": "14.Regnorum_IV.txt",
    "I_Chronicles": "15.Paralipomenon_I.txt",
    "II_Chronicles": "16.Paralipomenon_II.txt",
    "Esther": "19.Esther.txt",
    "Job": "32.Job.txt",
    "Psalms": "27.Psalmi.txt",
    "Proverbs": "29.Proverbia.txt",
    "Song_of_Solomon": "31.Canticum.txt",
    "Isaiah": "48.Isaias.txt",
    "Jeremiah": "49.Jeremias.txt",
    "Lamentations": "51.Threni_seu_Lamentationes.txt",
    "Ezekiel": "53.Ezechiel.txt",
    "Daniel": (
        "56.Daniel_translatio_Graeca.txt",
        "57.Daniel_Theodotionis_versio.txt",
    ),
    "Hosea": "36.Osee.txt",
    "Joel": "39.Joel.txt",
    "Amos": "37.Amos.txt",
    "Obadiah": "40.Abdias.txt",
    "Jonah": "41.Jonas.txt",
    "Micah": "38.Michaeas.txt",
    "Nahum": "42.Nahum.txt",
    "Habakkuk": "43.Habacuc.txt",
    "Zephaniah": "44.Sophonias.txt",
    "Haggai": "45.Aggaeus.txt",
    "Zechariah": "46.Zacharias.txt",
    "Tobit": "21.Tobias.txt",
    "Judith": "20.Judith.txt",
    "Wisdom": "33.Sapientia_Salomonis.txt",
    "Sirach": "34.Ecclesiasticus.txt",
    "Baruch": "50.Baruch.txt",
    "Epistle_of_Jeremiah": "52.Epistula_Jeremiae.txt",
    "Susanna": (
        "54.Susanna_translatio_Graeca.txt",
        "55.Susanna_Theodotionis_versio.txt",
    ),
    "Bel_and_the_Dragon": (
        "58.Bel_et_Draco_translatio_Graeca.txt",
        "59.Bel_et_Draco_Theodotionis_versio.txt",
    ),
    "II_Maccabees": "24.Machabaeorum_ii.txt",
}

_TT_NAME_RE = re.compile(r"^(\d+)_(.+)_(\d+)\.tt$")
_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:\-]*)="([^"]*)"')
_ORIG_RE = re.compile(r'<orig\s+orig="([^"]*)">')
_OT_TAG_RE = re.compile(
    r"<verse_n\b[^>]*>"
    r"|</verse_n>"
    r"|<orig_group\b[^>]*>"
    r"|</orig_group>"
    r"|<norm_group\b[^>]*>"
    r"|</norm_group>"
    r"|<multiword\b[^>]*>"
    r"|</multiword>"
    r"|<orig\b[^>]*>"
    r"|<norm\b[^>]*>"
    r"|</norm>"
    r"|<morph\b[^>]*>"
    r"|<lang\b[^>]*>"
)

# Latin lookalikes that Swete's OCR/digitization leaves inside Greek words.
_LATIN_LOOKALIKES = str.maketrans(
    {
        "A": "Α",
        "B": "Β",
        "E": "Ε",
        "H": "Η",
        "I": "Ι",
        "K": "Κ",
        "M": "Μ",
        "N": "Ν",
        "O": "Ο",
        "P": "Ρ",
        "T": "Τ",
        "X": "Χ",
        "Y": "Υ",
        "Z": "Ζ",
        "a": "α",
        "e": "ε",
        "i": "ι",
        "k": "κ",
        "m": "μ",
        "n": "ν",
        "o": "ο",
        "p": "ρ",
        "t": "τ",
        "x": "χ",
        "y": "υ",
        "z": "ζ",
    }
)
_GREEK_PUNCT_STRIP = re.compile(
    r"^[\s·.,;;:!?()\[\]\"'«»—\-]+|[\s·.,;;:!?()\[\]\"'«»—\-]+$"
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _parse_attrs(blob: str) -> dict[str, str]:
    return {k: v for k, v in _ATTR_RE.findall(blob)}


def clean_swete_word(raw_word: str) -> str:
    """Repair Latin lookalikes in Swete digitization while keeping polytonic accents."""
    word = _nfc(raw_word.translate(_LATIN_LOOKALIKES))
    return _GREEK_PUNCT_STRIP.sub("", word)


def _greek_lower(word: str) -> str:
    low = word.lower()
    if len(low) > 1 and low.endswith("σ"):
        low = low[:-1] + "ς"
    return low


@dataclass
class _RawOtToken:
    group_seq: int
    xml_id: str
    head_ref: str
    dep_rel: str | None
    orig: str
    norm: str
    lemma: str
    pos: str
    morph: tuple[str, ...]
    lang: str | None
    multiword: str | None


def parse_ot_chapter_tt(
    text: str,
    *,
    book: str,
    chapter: int,
) -> dict[int, CopticUnit]:
    """Parse one Sahidic OT chapter ``*.tt`` preserving all morpheme annotations."""
    verses: dict[int, CopticUnit] = {}
    current_verse: int | None = None
    orig_buf: list[str] = []
    norm_buf: list[str] = []
    group_seq = -1
    in_orig_group = False
    active_multiword: str | None = None
    pending_orig: str | None = None
    in_norm = False
    cur_attrs: dict[str, str] = {}
    cur_orig = ""
    cur_morph: list[str] = []
    cur_lang: str | None = None
    cur_multiword: str | None = None
    cur_group_seq = 0
    raw_tokens: list[_RawOtToken] = []

    def flush_norm() -> None:
        nonlocal in_norm, cur_attrs, cur_orig, cur_morph, cur_lang, cur_multiword
        if not in_norm or current_verse is None:
            in_norm = False
            return
        norm_val = _nfc(cur_attrs.get("norm", cur_orig))
        lemma_val = _nfc(cur_attrs.get("lemma", norm_val))
        pos_val = cur_attrs.get("pos", "UNK")
        raw_tokens.append(
            _RawOtToken(
                group_seq=cur_group_seq,
                xml_id=cur_attrs.get("xml:id", ""),
                head_ref=cur_attrs.get("head", ""),
                dep_rel=cur_attrs.get("func") or None,
                orig=cur_orig,
                norm=norm_val,
                lemma=lemma_val,
                pos=pos_val,
                morph=tuple(_nfc(m) for m in cur_morph if m),
                lang=cur_lang,
                multiword=cur_multiword,
            )
        )
        norm_buf.append(norm_val)
        in_norm = False
        cur_attrs = {}
        cur_orig = ""
        cur_morph = []
        cur_lang = None
        cur_multiword = None

    def flush_verse() -> None:
        nonlocal current_verse, orig_buf, norm_buf, group_seq
        nonlocal in_orig_group, active_multiword, pending_orig, raw_tokens
        flush_norm()
        if current_verse is None or not raw_tokens:
            current_verse = None
            orig_buf = []
            norm_buf = []
            group_seq = -1
            in_orig_group = False
            active_multiword = None
            pending_orig = None
            raw_tokens = []
            return
        unit_id = f"ot.{book}.{chapter}.{current_verse}"
        xml_to_seq: dict[str, int] = {
            item.xml_id: seq for seq, item in enumerate(raw_tokens) if item.xml_id
        }
        morphemes: list[CopticMorpheme] = []
        for seq, item in enumerate(raw_tokens):
            head_ref = item.head_ref.lstrip("#")
            head_seq = xml_to_seq.get(head_ref) if head_ref else None
            morphemes.append(
                CopticMorpheme(
                    id=token_id(OT_SOURCE_ID, unit_id, seq),
                    unit_id=unit_id,
                    seq=seq,
                    group_seq=item.group_seq,
                    orig=item.orig,
                    norm=item.norm,
                    lemma=item.lemma,
                    pos=item.pos,
                    morph=item.morph,
                    lang=item.lang,
                    multiword=item.multiword,
                    head_seq=head_seq,
                    dep_rel=item.dep_rel,
                )
            )
        verses[current_verse] = CopticUnit(
            unit_id=unit_id,
            source_id=OT_SOURCE_ID,
            book=f"ot.{book.lower()}",
            chapter=chapter,
            verse=current_verse,
            kind=UnitKind.VERSE,
            original_text=_nfc(" ".join(orig_buf)),
            normalized_text=_nfc(" ".join(norm_buf)),
            morphemes=morphemes,
        )
        current_verse = None
        orig_buf = []
        norm_buf = []
        group_seq = -1
        in_orig_group = False
        active_multiword = None
        pending_orig = None
        raw_tokens = []

    for match in _OT_TAG_RE.finditer(text):
        tag = match.group(0)
        if tag.startswith("<verse_n"):
            flush_verse()
            attrs = _parse_attrs(tag[8:-1])
            v_str = attrs.get("verse_n", "")
            current_verse = int(v_str) if v_str.isdigit() else None
            continue
        if tag.startswith("</verse_n"):
            flush_verse()
            continue
        if current_verse is None:
            continue
        if tag.startswith("<orig_group"):
            group_seq += 1
            in_orig_group = True
            attrs = _parse_attrs(tag[11:-1])
            grp = _nfc(attrs.get("orig_group", "") or attrs.get("norm_group", ""))
            if grp:
                orig_buf.append(grp)
            continue
        if tag.startswith("<norm_group"):
            if not in_orig_group:
                group_seq += 1
                in_orig_group = True
                attrs = _parse_attrs(tag[11:-1])
                grp = _nfc(attrs.get("orig_group", "") or attrs.get("norm_group", ""))
                if grp:
                    orig_buf.append(grp)
            continue
        if tag.startswith(("</orig_group", "</norm_group")):
            in_orig_group = False
            continue
        if tag.startswith("<multiword"):
            attrs = _parse_attrs(tag[10:-1])
            active_multiword = _nfc(attrs.get("multiword", "")) or None
            continue
        if tag.startswith("</multiword"):
            active_multiword = None
            continue
        if tag.startswith("<orig"):
            m_orig = _ORIG_RE.match(tag)
            if m_orig is not None:
                orig_val = m_orig.group(1)
                pending_orig = orig_val
                if not in_orig_group:
                    group_seq += 1
                    orig_buf.append(_nfc(orig_val))
            continue
        if tag.startswith("<norm"):
            cur_attrs = _parse_attrs(tag[5:-1])
            orig_attr = cur_attrs.get("orig") or cur_attrs.get("norm")
            if pending_orig is None and orig_attr is None:
                continue
            flush_norm()
            cur_attrs = _parse_attrs(tag[5:-1])
            in_norm = True
            cur_orig = _nfc(
                pending_orig
                if pending_orig is not None
                else cur_attrs.get("orig", cur_attrs.get("norm", ""))
            )
            if not in_orig_group:
                group_seq += 1
                orig_buf.append(cur_orig)
            cur_morph = [m for m in cur_attrs.get("morph", "").split(";") if m]
            cur_lang = cur_attrs.get("lang", "").strip() or None
            cur_multiword = active_multiword
            cur_group_seq = max(group_seq, 0)
            pending_orig = None
            continue
        if tag.startswith("</norm"):
            flush_norm()
            continue
        if tag.startswith("<morph") and in_norm:
            attrs = _parse_attrs(tag[6:-1])
            piece = attrs.get("morph", "")
            if piece:
                cur_morph.append(piece)
            continue
        if tag.startswith("<lang") and in_norm:
            attrs = _parse_attrs(tag[5:-1])
            lang_val = attrs.get("lang", "").strip()
            if lang_val:
                cur_lang = lang_val
            continue

    flush_verse()
    return verses


def load_sahidic_ot_chapters(
    tt_zip: Path,
) -> dict[str, dict[tuple[int, int], CopticUnit]]:
    """Load all annotated Sahidic OT units keyed by ``book -> (chapter, verse)``."""
    if not tt_zip.is_file():
        msg = f"missing Sahidic OT zip: {tt_zip}"
        raise FileNotFoundError(msg)
    by_book: dict[str, dict[tuple[int, int], CopticUnit]] = defaultdict(dict)
    with zipfile.ZipFile(tt_zip) as archive:
        for info in archive.infolist():
            match = _TT_NAME_RE.match(Path(info.filename).name)
            if match is None:
                continue
            book = match.group(2)
            chapter = int(match.group(3))
            if book not in BOOK_TO_SWETE:
                continue
            text = archive.read(info).decode("utf-8", errors="replace")
            for verse, unit in parse_ot_chapter_tt(
                text, book=book, chapter=chapter
            ).items():
                by_book[book][(chapter, verse)] = unit
    return dict(by_book)


def parse_swete_file(text: str) -> dict[tuple[int, int], list[str]]:
    """Parse one Swete LXX file into cleaned polytonic Greek tokens by (ch, vs)."""
    verses: dict[tuple[int, int], list[str]] = defaultdict(list)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        ref, _, word = stripped.partition(" ")
        parts = ref.split(".")
        if len(parts) != 3 or not word:
            continue
        try:
            chapter = int(parts[1])
            verse = int(parts[2])
        except ValueError:
            continue
        cleaned = clean_swete_word(word)
        if cleaned:
            verses[(chapter, verse)].append(cleaned)
    return dict(verses)


def _choose_swete_text(
    coptic: dict[tuple[int, int], CopticUnit],
    candidates: dict[str, dict[tuple[int, int], list[str]]],
) -> dict[tuple[int, int], list[str]]:
    best_name = ""
    best_key: tuple[int, float] = (-1, 1e9)
    best: dict[tuple[int, int], list[str]] = {}
    for name, greek in candidates.items():
        shared = [key for key in coptic if greek.get(key)]
        if not shared:
            continue
        ratios = sorted(
            max(len(coptic[k].morphemes), len(greek[k]))
            / max(min(len(coptic[k].morphemes), len(greek[k])), 1)
            for k in shared
        )
        median = ratios[len(ratios) // 2]
        rank = (len(shared), abs(median - 1.0))
        if rank[0] > best_key[0] or (rank[0] == best_key[0] and rank[1] < best_key[1]):
            best_key = rank
            best_name = name
            best = greek
    return best if best_name else {}


def load_ot_swete_pairs(
    *,
    tt_zip: Path,
    swete_dir: Path,
) -> list[tuple[CopticUnit, GreekUnit]]:
    """Pair Sahidic OT annotated verses with Swete LXX Greek verses."""
    if not swete_dir.is_dir():
        msg = f"missing Swete LXX directory: {swete_dir}"
        raise FileNotFoundError(msg)
    swete_data_dir = swete_dir / "data" if (swete_dir / "data").is_dir() else swete_dir
    coptic_by_book = load_sahidic_ot_chapters(tt_zip)
    swete_cache: dict[str, dict[tuple[int, int], list[str]]] = {}
    pairs: list[tuple[CopticUnit, GreekUnit]] = []
    for book, coptic in sorted(coptic_by_book.items()):
        spec = BOOK_TO_SWETE[book]
        names = (spec,) if isinstance(spec, str) else spec
        loaded: dict[str, dict[tuple[int, int], list[str]]] = {}
        for name in names:
            if name not in swete_cache:
                path = swete_data_dir / name
                if path.is_file():
                    swete_cache[name] = parse_swete_file(
                        path.read_text(encoding="utf-8")
                    )
            if name in swete_cache:
                loaded[name] = swete_cache[name]
        greek = (
            _choose_swete_text(coptic, loaded)
            if len(loaded) > 1
            else next(iter(loaded.values()), {})
        )
        if not greek:
            continue
        for key in sorted(set(coptic) & set(greek)):
            c_unit = coptic[key]
            g_words = greek[key]
            if not c_unit.morphemes or not g_words:
                continue
            ratio = max(len(c_unit.morphemes), len(g_words)) / min(
                len(c_unit.morphemes), len(g_words)
            )
            if ratio > MAX_TOKEN_RATIO:
                continue
            chapter, verse = key
            unit_id = c_unit.unit_id
            g_tokens = [
                GreekToken(
                    id=token_id(SWETE_SOURCE_ID, unit_id, seq),
                    unit_id=unit_id,
                    seq=seq,
                    surface=word,
                    normalized=_greek_lower(word),
                    lemma=_greek_lower(word),
                    morph=GreekMorphFeatures(pos="UNK"),
                )
                for seq, word in enumerate(g_words)
            ]
            g_unit = GreekUnit(
                unit_id=unit_id,
                source_id=SWETE_SOURCE_ID,
                book=c_unit.book,
                chapter=chapter,
                verse=verse,
                original_text=" ".join(g_words),
                normalized_text=" ".join(t.normalized for t in g_tokens),
                tokens=g_tokens,
            )
            pairs.append((c_unit, g_unit))
    return pairs

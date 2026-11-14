# SPDX-License-Identifier: Apache-2.0
"""Verse pairs from SCRIPTORIUM Sahidic OT and Swete LXX.

Both texts are CC BY-SA 4.0 and are already allowed as setup-download.
Verses are paired only when the book is the same work and the chapter and
verse numbers exist on both sides. A pair whose token counts differ by more
than a fixed ratio is dropped, because OT versification does not always match.
No English gloss and no Thomas text is read.
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from pathlib import Path

from zetodos_thomas.dictionary.build import DictionaryBuildError
from zetodos_thomas.dictionary.orthography import fold_coptic, fold_greek
from zetodos_thomas.integrations.aligners.types import BitextSentence

OT_TT_URL = (
    "https://raw.githubusercontent.com/CopticScriptorium/corpora/"
    "v6.2.0/sahidic.ot/sahidic.ot_TT.zip"
)
SWETE_REPO = "https://github.com/nathans/lxx-swete"
SWETE_COMMIT = "26bad3eb42bba98471d154c954e36a6f30a0279d"
OT_LICENSE = "CC-BY-SA-4.0"

# Same biblical book, not a translation. Numeric prefixes differ between
# the two editions, so the join is by work name.
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

# Drop a numbered pair when one side is much longer than the other.
MAX_TOKEN_RATIO = 5.0

_VERSE_OPEN = re.compile(r'<verse_n\b[^>]*\bverse_n="(\d+)"')
_NORM_ATTR = re.compile(r'\bnorm="([^"]*)"')
_TT_NAME = re.compile(r"^(\d+)_(.+)_(\d+)\.tt$")

# Latin letters that the Swete digitization sometimes leaves inside Greek words.
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


def normalize_swete_word(word: str) -> str:
    """Fold a Swete token, repairing Latin lookalikes in the digitization."""
    return fold_greek(word.translate(_LATIN_LOOKALIKES))


def parse_tt_chapter(text: str) -> dict[int, list[str]]:
    """Normalized Coptic tokens keyed by verse number."""
    opens = list(_VERSE_OPEN.finditer(text))
    verses: dict[int, list[str]] = {}
    for index, match in enumerate(opens):
        verse = int(match.group(1))
        end = opens[index + 1].start() if index + 1 < len(opens) else len(text)
        tokens: list[str] = []
        for norm in _NORM_ATTR.findall(text[match.end() : end]):
            folded = fold_coptic(norm)
            if folded:
                tokens.append(folded)
        if tokens:
            verses[verse] = tokens
    return verses


def parse_swete_file(text: str) -> dict[tuple[int, int], list[str]]:
    """Greek tokens keyed by ``(chapter, verse)``."""
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
        token = normalize_swete_word(word)
        if token:
            verses[(chapter, verse)].append(token)
    return dict(verses)


def _tt_name(name: str) -> tuple[str, int] | None:
    match = _TT_NAME.match(Path(name).name)
    if match is None:
        return None
    book = match.group(2)
    chapter = int(match.group(3))
    if book not in BOOK_TO_SWETE:
        return None
    return book, chapter


def _length_ok(coptic: list[str], greek: list[str]) -> bool:
    if not coptic or not greek:
        return False
    ratio = max(len(coptic), len(greek)) / min(len(coptic), len(greek))
    return ratio <= MAX_TOKEN_RATIO


def choose_swete_text(
    coptic: dict[tuple[int, int], list[str]],
    candidates: dict[str, dict[tuple[int, int], list[str]]],
) -> dict[tuple[int, int], list[str]]:
    """Pick the Swete edition that shares the most verse ids with Coptic.

    Ties break toward the edition whose shared verses have token counts
    closer to the Coptic side. No word gloss is consulted.
    """
    best_name = ""
    best_key: tuple[int, float] = (-1, 1e9)
    best: dict[tuple[int, int], list[str]] = {}
    for name, greek in candidates.items():
        shared = [key for key in coptic if key in greek and coptic[key] and greek[key]]
        if not shared:
            continue
        ratios = [
            max(len(coptic[key]), len(greek[key]))
            / min(len(coptic[key]), len(greek[key]))
            for key in shared
        ]
        ratios.sort()
        median = ratios[len(ratios) // 2]
        rank = (len(shared), abs(median - 1.0))
        if rank[0] > best_key[0] or (rank[0] == best_key[0] and rank[1] < best_key[1]):
            best_key = rank
            best_name = name
            best = greek
    if not best_name:
        return {}
    return best


def iter_ot_sentences(
    tt_zip: Path,
    swete_dir: Path,
) -> list[BitextSentence]:
    """Build train sentences. Unit ids are ``ot.*`` and never Thomas."""
    if not tt_zip.is_file():
        msg = f"missing Sahidic OT zip: {tt_zip}"
        raise FileNotFoundError(msg)
    if not swete_dir.is_dir():
        msg = f"missing Swete directory: {swete_dir}"
        raise FileNotFoundError(msg)
    swete_cache: dict[str, dict[tuple[int, int], list[str]]] = {}
    coptic_by_book: dict[str, dict[tuple[int, int], list[str]]] = defaultdict(dict)
    with zipfile.ZipFile(tt_zip) as archive:
        for info in archive.infolist():
            parsed = _tt_name(info.filename)
            if parsed is None:
                continue
            book, chapter = parsed
            text = archive.read(info).decode("utf-8", errors="replace")
            for verse, tokens in parse_tt_chapter(text).items():
                coptic_by_book[book][(chapter, verse)] = tokens
    sentences: list[BitextSentence] = []
    for book, coptic in sorted(coptic_by_book.items()):
        spec = BOOK_TO_SWETE[book]
        names = (spec,) if isinstance(spec, str) else spec
        loaded: dict[str, dict[tuple[int, int], list[str]]] = {}
        for name in names:
            if name not in swete_cache:
                path = swete_dir / name
                if not path.is_file():
                    continue
                swete_cache[name] = parse_swete_file(path.read_text(encoding="utf-8"))
            if name in swete_cache:
                loaded[name] = swete_cache[name]
        greek = (
            choose_swete_text(coptic, loaded)
            if len(loaded) > 1
            else next(iter(loaded.values()), {})
        )
        if not greek:
            continue
        for key in sorted(set(coptic) & set(greek)):
            c_tokens = coptic[key]
            g_tokens = greek[key]
            if not _length_ok(c_tokens, g_tokens):
                continue
            chapter, verse = key
            unit_id = f"ot.{book}.{chapter}.{verse}"
            _assert_ot_unit(unit_id)
            sentences.append(
                BitextSentence(
                    unit_id=unit_id,
                    source_tokens=tuple(c_tokens),
                    target_tokens=tuple(g_tokens),
                    source_token_ids=tuple(
                        f"{unit_id}/c/{i}" for i in range(len(c_tokens))
                    ),
                    target_token_ids=tuple(
                        f"{unit_id}/g/{i}" for i in range(len(g_tokens))
                    ),
                    split_role="train",
                )
            )
    return sentences


def _assert_ot_unit(unit_id: str) -> None:
    lowered = unit_id.lower()
    if "thomas" in lowered or "poxy" in lowered:
        msg = f"refusing non-OT unit in the OT bitext: {unit_id}"
        raise DictionaryBuildError(msg)

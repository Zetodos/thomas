# SPDX-License-Identifier: Apache-2.0
"""Ingest MorphGNT SBLGNT (all 27 NT books) and pair with CopSahHorner."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from zetodos_thomas.ingest.copsahhorner import open_module as open_horner
from zetodos_thomas.ingest.copsahhorner import read_verse
from zetodos_thomas.schemas.holdout import (
    EvaluationWitnessExclusion,
    NtCorpusSplit,
    PassageRef,
    PassageRole,
)
from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.linguistic import (
    GreekMorphFeatures,
    GreekToken,
    GreekUnit,
)
from zetodos_thomas.schemas.source import TextUnit

SOURCE_ID = "morphgnt-sblgnt"
SAHIDIC_SOURCE_ID = "copsahhorner"

# MorphGNT book numbers (61..87) -> lowercase book names matching CopSahHorner.
MORPHGNT_BOOKS: dict[str, tuple[str, str]] = {
    "61": ("61-Mt-morphgnt.txt", "matthew"),
    "62": ("62-Mk-morphgnt.txt", "mark"),
    "63": ("63-Lk-morphgnt.txt", "luke"),
    "64": ("64-Jn-morphgnt.txt", "john"),
    "65": ("65-Ac-morphgnt.txt", "acts"),
    "66": ("66-Ro-morphgnt.txt", "romans"),
    "67": ("67-1Co-morphgnt.txt", "1corinthians"),
    "68": ("68-2Co-morphgnt.txt", "2corinthians"),
    "69": ("69-Ga-morphgnt.txt", "galatians"),
    "70": ("70-Eph-morphgnt.txt", "ephesians"),
    "71": ("71-Php-morphgnt.txt", "philippians"),
    "72": ("72-Col-morphgnt.txt", "colossians"),
    "73": ("73-1Th-morphgnt.txt", "1thessalonians"),
    "74": ("74-2Th-morphgnt.txt", "2thessalonians"),
    "75": ("75-1Ti-morphgnt.txt", "1timothy"),
    "76": ("76-2Ti-morphgnt.txt", "2timothy"),
    "77": ("77-Tit-morphgnt.txt", "titus"),
    "78": ("78-Phm-morphgnt.txt", "philemon"),
    "79": ("79-Heb-morphgnt.txt", "hebrews"),
    "80": ("80-Jas-morphgnt.txt", "james"),
    "81": ("81-1Pe-morphgnt.txt", "1peter"),
    "82": ("82-2Pe-morphgnt.txt", "2peter"),
    "83": ("83-1Jn-morphgnt.txt", "1john"),
    "84": ("84-2Jn-morphgnt.txt", "2john"),
    "85": ("85-3Jn-morphgnt.txt", "3john"),
    "86": ("86-Jud-morphgnt.txt", "jude"),
    "87": ("87-Re-morphgnt.txt", "revelation"),
}

GOSPEL_BOOKS: frozenset[str] = frozenset({"matthew", "mark", "luke", "john"})

_GREEK_WS_RE = re.compile(r"\S+")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def parse_morphgnt_file(path: Path, book: str) -> dict[str, GreekUnit]:
    """Parse one ``*-morphgnt.txt`` file into ``GreekUnit`` keyed by ``unit_id``."""
    by_verse: dict[tuple[int, int], list[tuple[str, str, str, str, str, str]]] = (
        defaultdict(list)
    )
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            msg = f"unexpected MorphGNT line in {path.name}: {line!r}"
            raise ValueError(msg)
        bcv, pos, parse_code, text, word, norm_word, lemma = parts[:7]
        chapter = int(bcv[2:4])
        verse = int(bcv[4:6])
        by_verse[(chapter, verse)].append(
            (pos, parse_code, _nfc(text), _nfc(word), _nfc(norm_word), _nfc(lemma))
        )

    units: dict[str, GreekUnit] = {}
    for (chapter, verse), rows in sorted(by_verse.items()):
        unit_id = f"{book}.{chapter}.{verse}"
        tokens: list[GreekToken] = []
        orig_surfaces: list[str] = []
        norm_surfaces: list[str] = []
        for seq, (pos, parse_code, text, word, norm_word, lemma) in enumerate(rows):
            orig_surfaces.append(text)
            norm_surfaces.append(norm_word)
            tokens.append(
                GreekToken(
                    id=token_id(SOURCE_ID, unit_id, seq),
                    unit_id=unit_id,
                    seq=seq,
                    surface=word,
                    normalized=norm_word,
                    lemma=lemma,
                    morph=GreekMorphFeatures.from_morphgnt(pos, parse_code),
                )
            )
        units[unit_id] = GreekUnit(
            unit_id=unit_id,
            source_id=SOURCE_ID,
            book=book,
            chapter=chapter,
            verse=verse,
            original_text=" ".join(orig_surfaces),
            normalized_text=" ".join(norm_surfaces),
            tokens=tokens,
        )
    return units


def load_morphgnt_nt(morphgnt_dir: Path) -> dict[str, GreekUnit]:
    """Load all 27 NT books from ``morphgnt/sblgnt`` checkout."""
    all_units: dict[str, GreekUnit] = {}
    for _code, (filename, book) in sorted(MORPHGNT_BOOKS.items()):
        path = morphgnt_dir / filename
        if not path.is_file():
            msg = f"missing MorphGNT book file: {path}"
            raise FileNotFoundError(msg)
        all_units.update(parse_morphgnt_file(path, book))
    return all_units


def pair_horner_morphgnt(
    *,
    sword_root: Path,
    morphgnt_dir: Path,
) -> list[tuple[TextUnit, GreekUnit]]:
    """Pair every nonempty CopSahHorner NT verse with its MorphGNT Greek verse."""
    greek_units = load_morphgnt_nt(morphgnt_dir)
    bible = open_horner(sword_root)
    paired: list[tuple[TextUnit, GreekUnit]] = []
    for _code, (_filename, book) in sorted(MORPHGNT_BOOKS.items()):
        book_units = [u for u in greek_units.values() if u.book == book]
        book_units.sort(key=lambda u: (u.chapter, u.verse))
        for seq, g_unit in enumerate(book_units):
            try:
                c_unit = read_verse(
                    bible,
                    book,
                    g_unit.chapter,
                    g_unit.verse,
                    source_id=SAHIDIC_SOURCE_ID,
                    sequence=seq,
                )
            except (KeyError, IndexError, ValueError):
                continue
            if not c_unit.normalized_text.strip() or not g_unit.tokens:
                continue
            paired.append((c_unit, g_unit))
    return paired


def _greek_5grams(unit: GreekUnit) -> set[str]:
    words = [t.normalized for t in unit.tokens if t.normalized]
    if len(words) < 5:
        return set()
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def build_nt_split_v1(
    *,
    paired_verses: list[tuple[TextUnit, GreekUnit]],
    legacy_split_path: Path,
    omission_unit_ids: set[str],
) -> NtCorpusSplit:
    """Build full-NT split preserving the 41 holdout & 51 dev Gospel verses.

    Any remaining train-candidate verse sharing an exact Greek 5-gram with any
    holdout verse is excluded from train (placed into dev/excluded or omitted
    from train) so zero Synoptic 5-gram leakage exists across the full NT.
    """
    legacy = NtCorpusSplit.model_validate(
        json.loads(legacy_split_path.read_text(encoding="utf-8"))
    )
    holdout_ids = {
        uid
        for p in legacy.passages
        if p.role is PassageRole.HOLDOUT
        for uid in p.unit_ids
    }
    dev_ids = {
        uid for p in legacy.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    }

    by_id = {g.unit_id: (c, g) for c, g in paired_verses}
    holdout_5grams: set[str] = set()
    for uid in holdout_ids:
        pair = by_id.get(uid)
        if pair is not None:
            holdout_5grams.update(_greek_5grams(pair[1]))

    # Assign roles per unit; drop train units with exact 5-gram overlap with holdout.
    excluded_synoptic: list[str] = []
    passages_by_role_chapter: dict[tuple[PassageRole, str, int], list[int]] = (
        defaultdict(list)
    )
    for uid, (_c, g) in sorted(by_id.items()):
        if uid in omission_unit_ids:
            continue
        if uid in holdout_ids:
            role = PassageRole.HOLDOUT
        elif uid in dev_ids:
            role = PassageRole.DEV
        else:
            if _greek_5grams(g) & holdout_5grams:
                excluded_synoptic.append(uid)
                continue
            role = PassageRole.TRAIN
        passages_by_role_chapter[(role, g.book, g.chapter)].append(g.verse)

    passages: list[PassageRef] = []
    for (role, book, chapter), verses in sorted(
        passages_by_role_chapter.items(),
        key=lambda item: (item[0][0].value, item[0][1], item[0][2]),
    ):
        verses_sorted = sorted(verses)
        passages.append(
            PassageRef(
                book=book,
                chapter_from=chapter,
                verse_from=verses_sorted[0],
                chapter_to=chapter,
                verse_to=verses_sorted[-1],
                unit_ids=[f"{book}.{chapter}.{v}" for v in verses_sorted],
                role=role,
                notes=(
                    "Preserved v3 holdout/dev"
                    if role in {PassageRole.HOLDOUT, PassageRole.DEV}
                    else "Full-NT MorphGNT x CopSahHorner train chapter"
                ),
            )
        )

    return NtCorpusSplit(
        id="horner-morphgnt-nt-split-v1",
        title="Full 27-book Horner x MorphGNT NT split (41 holdout, 51 dev, leak-free)",
        schema_version="1",
        greek_source_id=SOURCE_ID,
        sahidic_source_id=SAHIDIC_SOURCE_ID,
        pilot_corpus_id="horner-morphgnt-nt-v1",
        owned_by="phase-b-reset",
        consumed_by=["morphology", "align", "lexicon", "reconstruct", "eval"],
        passages=passages,
        evaluation_witness_exclusions=[
            EvaluationWitnessExclusion(
                witness_id=ex.witness_id,
                source_id=ex.source_id,
                manuscript_id=ex.manuscript_id,
                corpus_role="evaluation-witness",
                unit_ids=list(ex.unit_ids),
                notes=ex.notes,
            )
            for ex in legacy.evaluation_witness_exclusions
        ],
        secondary_ot_used=True,
        secondary_ot_notes=(
            "Trained jointly with scriptorium-sahidic-ot x swete-lxx "
            f"(5-gram Synoptic exclusions from train: {excluded_synoptic})"
        ),
        notes=(
            "Preserves exact 41 holdout and 51 dev verses across all 27 NT books; "
            "omits John 8 PA and any train verse sharing a Greek 5-gram with holdouts."
        ),
    )

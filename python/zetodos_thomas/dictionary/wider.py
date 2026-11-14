# SPDX-License-Identifier: Apache-2.0
"""Re-fit the parallel dictionary on a wider Horner x Vaticanus bitext.

Four Gospels, verse-aligned. The JAK-43 dev, holdout, and John 8 omission
verses stay out of training. Thomas Greek is not read here.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from zetodos_thomas.align.bitext import build_bitext
from zetodos_thomas.align.splits import PassageRange, PilotSplits, SplitRole
from zetodos_thomas.align.tokenize import whitespace_token_counts
from zetodos_thomas.dictionary.build import (
    DictionaryBuildError,
    ObservedLink,
    ParallelDictionary,
    build_dictionary,
)
from zetodos_thomas.dictionary.diagnose import load_nt_splits, nt_dev_snap_diagnostic
from zetodos_thomas.dictionary.orthography import fold_coptic, fold_greek
from zetodos_thomas.dictionary.ot_bitext import OT_LICENSE, SWETE_COMMIT
from zetodos_thomas.dictionary.run import (
    VERSION as V1_VERSION,
)
from zetodos_thomas.dictionary.run import (
    _next_step,
    _sufficient,
    _token_row,
    _write_outputs,
)
from zetodos_thomas.ingest.copsahhorner import open_module as open_horner
from zetodos_thomas.ingest.copsahhorner import read_verse
from zetodos_thomas.ingest.download import DownloadError, fetch_ntvmr_teiraw
from zetodos_thomas.ingest.ntvmr_vaticanus import SOURCE_ID as GREEK_SOURCE_ID
from zetodos_thomas.ingest.ntvmr_vaticanus import parse_teiraw
from zetodos_thomas.integrations.aligners.types import AlignerName, AlignmentLinks
from zetodos_thomas.lexicon.candidates import align_train_sentences
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus, TextUnit

VERSION = "parallel-dict-v2"
CORPUS_ID = "horner-vaticanus-gospels-v1"
SAHIDIC_SOURCE_ID = "copsahhorner"

# Standard chapter lengths. A verse the manuscript lacks stays empty and is
# not a training pair. This is versification, not a gloss list.
GOSPEL_VERSE_COUNTS: dict[str, tuple[int, ...]] = {
    "matthew": (
        25,
        23,
        17,
        25,
        48,
        34,
        29,
        34,
        38,
        42,
        30,
        50,
        58,
        36,
        39,
        28,
        27,
        35,
        30,
        34,
        46,
        46,
        39,
        51,
        46,
        75,
        66,
        20,
    ),
    "mark": (
        45,
        28,
        35,
        41,
        43,
        56,
        37,
        38,
        50,
        52,
        33,
        44,
        37,
        72,
        47,
        20,
    ),
    "luke": (
        80,
        52,
        38,
        44,
        39,
        49,
        50,
        56,
        62,
        42,
        54,
        59,
        35,
        35,
        32,
        31,
        37,
        43,
        48,
        47,
        38,
        71,
        56,
        53,
    ),
    "john": (
        51,
        25,
        36,
        54,
        47,
        71,
        53,
        59,
        41,
        42,
        57,
        50,
        38,
        31,
        27,
        33,
        26,
        40,
        42,
        31,
        25,
    ),
}

GOSPEL_OSIS: dict[str, str] = {
    "matthew": "Matt",
    "mark": "Mark",
    "luke": "Luke",
    "john": "John",
}


def gospel_ranges() -> list[tuple[str, str, int, int]]:
    """``(book, osis_range, chapter, verse_count)`` for every Gospel chapter."""
    ranges: list[tuple[str, str, int, int]] = []
    for book, counts in GOSPEL_VERSE_COUNTS.items():
        osis = GOSPEL_OSIS[book]
        for index, count in enumerate(counts, start=1):
            ranges.append((book, f"{osis}.{index}.1-{count}", index, count))
    return ranges


def gospel_tei_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "downloads" / "ntvmr-vaticanus-gospels"


def download_gospel_tei(
    repo_root: Path,
    *,
    force: bool = False,
    throttle_seconds: float = 0.4,
) -> Path:
    """Fetch Vaticanus B* teiraw for each Gospel chapter. No images."""
    import time

    dest_dir = gospel_tei_dir(repo_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for index, (_book, osis_range, _chapter, _count) in enumerate(gospel_ranges()):
        dest = dest_dir / f"{osis_range.replace('.', '_')}.teiraw.xml"
        if dest.is_file() and not force and dest.stat().st_size > 200:
            continue
        if index > 0 and throttle_seconds > 0:
            time.sleep(throttle_seconds)
        fetch_ntvmr_teiraw(osis_range, dest, force=force)
    return dest_dir


def role_for_unit(unit_id: str, base: PilotSplits) -> SplitRole:
    """Keep a v3 dev/holdout/omission label; everything else is train."""
    lowered = unit_id.lower().replace(" ", "")
    if lowered.startswith("thomas.") or "poxy" in lowered or "p.oxy" in lowered:
        msg = f"refusing Thomas or fragment unit in the Gospel bitext: {unit_id}"
        raise DictionaryBuildError(msg)
    existing = base.role_for(unit_id)
    if existing is None:
        return SplitRole.TRAIN
    return existing


def wider_splits(
    unit_ids: list[str],
    base: PilotSplits,
    *,
    corpus_id: str = CORPUS_ID,
) -> PilotSplits:
    """Label Gospel verses, preserving the v3 dev, holdout, and omission sets."""
    claimed = {
        unit_id for passage in base.passages for unit_id in passage.listed_unit_ids()
    }
    extra_train = [unit_id for unit_id in unit_ids if unit_id not in claimed]
    for unit_id in unit_ids:
        role_for_unit(unit_id, base)
    passages = list(base.passages)
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for unit_id in extra_train:
        book, chapter_s, verse_s = unit_id.split(".")
        grouped[(book, int(chapter_s))].append(int(verse_s))
    for (book, chapter), verses in sorted(grouped.items()):
        verses_sorted = sorted(verses)
        passages.append(
            PassageRange(
                book=book,
                chapter=chapter,
                verse_from=verses_sorted[0],
                verse_to=verses_sorted[-1],
                role=SplitRole.TRAIN,
                rationale=(
                    "Gospel verse outside the v3 dev, holdout, and omission sets."
                ),
                unit_ids=[f"{book}.{chapter}.{verse}" for verse in verses_sorted],
            )
        )
    excluded = list(base.excluded_from_training_stack)
    if not any(
        "poxy-654" in item.lower() or "poxy654" in item.lower() for item in excluded
    ):
        excluded.append("poxy-654 evaluation-witness (not training)")
    return PilotSplits(
        id="horner-vaticanus-gospels-split-v1",
        title="Four Gospels train, with v3 dev/holdout/omission held out",
        greek_source_id=base.greek_source_id,
        sahidic_source_id=base.sahidic_source_id,
        pilot_corpus_id=corpus_id,
        owned_by="parallel-dict-v2",
        passages=passages,
        notes=(
            "New Gospel verses are train. v3 dev, holdout, and John 8 omission "
            "labels are unchanged. Thomas Greek is not in this split."
        ),
        excluded_from_training_stack=excluded,
    )


_TEI_REPAIRS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r'<w>([^<]*)<app><rdg type="orig" hand="firsthand">([^<]*)</w></rdg>'
        ),
        r'<w>\1</w> <app><rdg type="orig" hand="firsthand"><w>\2</w></rdg>',
    ),
    (re.compile(r"<w>([^<]*)</rdg>"), r"<w>\1</w></rdg>"),
    (re.compile(r"</app>([^<]+)</w>"), r"</app> <w>\1</w>"),
)


def _parse_chapter(path: Path) -> list[TextUnit]:
    """Parse one teiraw file, repairing known NTVMR well-formedness slips.

    The repairs only close or reopen a ``w`` tag around a corrector reading.
    Firsthand text is unchanged.
    """
    try:
        ET.parse(path)
    except ET.ParseError:
        raw = path.read_text(encoding="utf-8")
        fixed = raw
        for pattern, replacement in _TEI_REPAIRS:
            fixed = pattern.sub(replacement, fixed)
        with tempfile.NamedTemporaryFile(
            suffix=".xml",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as handle:
            handle.write(fixed)
            tmp = Path(handle.name)
        try:
            return parse_teiraw(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    return parse_teiraw(path)


def _blank_if_untokenizable(unit: TextUnit) -> TextUnit:
    """Drop a side whose normalized text cannot be zipped to the original.

    Horner sometimes prints a bracket placeholder, or a restoration whose
    normalized form has a different token count. Those slots are not pairs.
    """
    if not unit.original_text.strip() and not unit.normalized_text.strip():
        return unit
    original_n, normalized_n = whitespace_token_counts(unit)
    if original_n != normalized_n or not unit.normalized_text.strip():
        return unit.model_copy(update={"original_text": "", "normalized_text": ""})
    return unit


def _empty_unit(book: str, chapter: int, verse: int, *, source_id: str) -> TextUnit:
    return TextUnit(
        source_id=source_id,
        unit_id=f"{book}.{chapter}.{verse}",
        book=book,
        chapter=chapter,
        verse=verse,
        original_text="",
        normalized_text="",
        sequence=0,
    )


def build_nt_corpus(*, sword_root: Path, tei_dirs: list[Path]) -> PilotCorpus:
    """Pair Horner with every Vaticanus verse found in ``tei_dirs``.

    Gospel files and the rest of the NT can live in different folders.
    v3 dev, holdout, and omission labels are applied later.
    """
    bible = open_horner(sword_root)
    grouped: dict[tuple[str, int], dict[int, TextUnit]] = defaultdict(dict)
    for tei_dir in tei_dirs:
        for path in sorted(tei_dir.glob("*.teiraw.xml")):
            for unit in _parse_chapter(path):
                grouped[(unit.book, unit.chapter)][unit.verse] = unit
    units: list[ParallelUnit] = []
    for book, chapter in sorted(grouped):
        if book not in GOSPEL_OSIS and book not in _NT_OSIS_EXTRA:
            continue
        by_verse = grouped[(book, chapter)]
        if not by_verse:
            continue
        last = max(by_verse)
        for verse in range(1, last + 1):
            greek = by_verse.get(verse) or _empty_unit(
                book, chapter, verse, source_id=GREEK_SOURCE_ID
            )
            try:
                sahidic = read_verse(
                    bible,
                    book,
                    chapter,
                    verse,
                    source_id=SAHIDIC_SOURCE_ID,
                    sequence=len(units),
                )
            except (KeyError, IndexError, ValueError):
                sahidic = _empty_unit(book, chapter, verse, source_id=SAHIDIC_SOURCE_ID)
            sahidic = _blank_if_untokenizable(sahidic)
            greek = _blank_if_untokenizable(greek)
            units.append(
                ParallelUnit(
                    unit_id=f"{book}.{chapter}.{verse}",
                    greek=greek,
                    sahidic=sahidic,
                    notes="",
                )
            )
    return PilotCorpus(
        id="horner-vaticanus-nt-wide-v1",
        title="Horner Sahidic NT x Vaticanus B* (parallel-dict-v3)",
        greek_source_id=GREEK_SOURCE_ID,
        sahidic_source_id=SAHIDIC_SOURCE_ID,
        units=units,
        notes=(
            "Public-domain Horner and CC BY 4.0 Vaticanus B* firsthand, "
            "Gospels plus the rest of the NT. No Thomas Greek."
        ),
    )


# Books outside the four Gospels. Chapter counts are only for the downloader.
_NT_OSIS_EXTRA: dict[str, str] = {
    "acts": "Acts",
    "romans": "Rom",
    "1corinthians": "1Cor",
    "2corinthians": "2Cor",
    "galatians": "Gal",
    "ephesians": "Eph",
    "philippians": "Phil",
    "colossians": "Col",
    "1thessalonians": "1Thess",
    "2thessalonians": "2Thess",
    "1timothy": "1Tim",
    "2timothy": "2Tim",
    "titus": "Titus",
    "philemon": "Phlm",
    "hebrews": "Heb",
    "james": "Jas",
    "1peter": "1Pet",
    "2peter": "2Pet",
    "1john": "1John",
    "2john": "2John",
    "3john": "3John",
    "jude": "Jude",
    "revelation": "Rev",
}


def nt_tei_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "downloads" / "ntvmr-vaticanus-nt"


def build_gospel_corpus(*, sword_root: Path, tei_dir: Path) -> PilotCorpus:
    """Pair Horner and Vaticanus B* for every Gospel verse slot."""
    bible = open_horner(sword_root)
    units: list[ParallelUnit] = []
    for book, osis_range, chapter, count in gospel_ranges():
        path = tei_dir / f"{osis_range.replace('.', '_')}.teiraw.xml"
        if not path.is_file():
            msg = f"missing Vaticanus teiraw {path.name}; download Gospel TEI first"
            raise FileNotFoundError(msg)
        greek_by_verse = {
            unit.verse: unit
            for unit in _parse_chapter(path)
            if unit.book == book
            and unit.chapter == chapter
            and 1 <= unit.verse <= count
        }
        for verse in range(1, count + 1):
            greek = greek_by_verse.get(verse) or _empty_unit(
                book, chapter, verse, source_id=GREEK_SOURCE_ID
            )
            sahidic = read_verse(
                bible,
                book,
                chapter,
                verse,
                source_id=SAHIDIC_SOURCE_ID,
                sequence=len(units),
            )
            # Horner prints "[ ]" for a verse the module omits. That is not text.
            if not sahidic.normalized_text.strip():
                sahidic = sahidic.model_copy(
                    update={"original_text": "", "normalized_text": ""}
                )
            if not greek.normalized_text.strip():
                greek = greek.model_copy(
                    update={"original_text": "", "normalized_text": ""}
                )
            units.append(
                ParallelUnit(
                    unit_id=f"{book}.{chapter}.{verse}",
                    greek=greek,
                    sahidic=sahidic,
                    notes="",
                )
            )
    return PilotCorpus(
        id=CORPUS_ID,
        title="Horner Sahidic Gospels x NTVMR Vaticanus B* (parallel-dict-v2)",
        greek_source_id=GREEK_SOURCE_ID,
        sahidic_source_id=SAHIDIC_SOURCE_ID,
        units=units,
        notes=(
            "Public-domain Horner Sahidic and CC BY 4.0 Vaticanus B* firsthand. "
            "No Thomas Greek. v3 dev, holdout, and omission verses are not training."
        ),
    )


def observed_links_from_alignments(
    by_aligner: dict[str, AlignmentLinks],
) -> tuple[list[ObservedLink], dict[str, dict[str, dict[str, int]]]]:
    """Union token-link counts, plus previous-token counts for case choice."""
    if set(by_aligner) != {"eflomal", "fast_align"}:
        msg = f"keep-both requires eflomal and fast_align; got {sorted(by_aligner)}"
        raise DictionaryBuildError(msg)
    union: dict[tuple[str, str], set[tuple[str, int, int]]] = defaultdict(set)
    per_aligner: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"eflomal": 0, "fast_align": 0}
    )
    context: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for name, links in by_aligner.items():
        if len(links.sentences) != len(links.symmetrized):
            msg = f"{name} symmetrized links do not match the sentence list"
            raise DictionaryBuildError(msg)
        for sentence, aligned in zip(links.sentences, links.symmetrized, strict=True):
            seen: set[tuple[str, int, int]] = set()
            for src_index, trg_index in aligned.pairs:
                if src_index < 0 or trg_index < 0:
                    continue
                if src_index >= len(sentence.source_tokens) or trg_index >= len(
                    sentence.target_tokens
                ):
                    continue
                occurrence = (sentence.unit_id, src_index, trg_index)
                if occurrence in seen:
                    continue
                seen.add(occurrence)
                pair = (
                    sentence.source_tokens[src_index],
                    sentence.target_tokens[trg_index],
                )
                union[pair].add(occurrence)
                per_aligner[pair][name] += 1
                if src_index > 0:
                    previous = fold_coptic(sentence.source_tokens[src_index - 1])
                    coptic_key = fold_coptic(pair[0])
                    greek_key = fold_greek(pair[1])
                    if previous and coptic_key and greek_key:
                        context[coptic_key][greek_key][previous] += 1
                    if src_index > 1 and previous and coptic_key and greek_key:
                        earlier = fold_coptic(sentence.source_tokens[src_index - 2])
                        if earlier:
                            bigram = earlier + "+" + previous
                            context[coptic_key][greek_key][bigram] += 1
    observed: list[ObservedLink] = []
    for coptic, greek in sorted(union):
        observed.append(
            ObservedLink(
                coptic=coptic,
                greek=greek,
                observation_count=len(union[(coptic, greek)]),
                count_eflomal=per_aligner[(coptic, greek)]["eflomal"],
                count_fast_align=per_aligner[(coptic, greek)]["fast_align"],
            )
        )
    plain_context = {
        coptic: {greek: dict(prevs) for greek, prevs in greeks.items()}
        for coptic, greeks in context.items()
    }
    return observed, plain_context


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def gospel_tei_digest(tei_dir: Path) -> str:
    """Stable digest of the Gospel teiraw files that fed this fit."""
    digest = hashlib.sha256()
    for path in sorted(tei_dir.glob("*.teiraw.xml")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _dev_token_precision(
    sentences: list[Any],
    dictionary: ParallelDictionary,
    *,
    use_context: bool,
) -> float:
    """Share of dev emissions whose Greek string occurs in that verse."""
    from zetodos_thomas.dictionary.apply import apply_tokens
    from zetodos_thomas.dictionary.orthography import fold_greek

    saved = dictionary.use_context
    dictionary.use_context = use_context
    emitted = 0
    hits = 0
    try:
        for sentence in sentences:
            gold = {fold_greek(token) for token in sentence.target_tokens}
            gold.discard("")
            outputs = apply_tokens(
                list(sentence.source_tokens),
                dictionary,
                snap_enabled=False,
            )
            for item in outputs:
                if not item.greek:
                    continue
                emitted += 1
                if fold_greek(item.greek) in gold:
                    hits += 1
    finally:
        dictionary.use_context = saved
    if emitted == 0:
        return 0.0
    return hits / emitted


def run_gospel_dictionary(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    sword_root: Path | None = None,
    tei_dir: Path | None = None,
    download: bool = True,
    prepared: PilotCorpus | None = None,
    version: str | None = None,
    extra_train: list[Any] | None = None,
) -> dict[str, Any]:
    """Align a Horner x Vaticanus corpus with both aligners, then score."""
    root = repo_root
    fit_version = version or VERSION
    tei = tei_dir or gospel_tei_dir(root)
    if prepared is None and download:
        try:
            tei = download_gospel_tei(root)
        except DownloadError:
            if not any(tei.glob("*.teiraw.xml")):
                raise
    sword = sword_root or (root / "data" / "downloads" / "sword")
    if not (sword / "mods.d" / "copsahhorner.conf").is_file():
        msg = f"CopSahHorner SWORD tree not found at {sword}"
        raise FileNotFoundError(msg)
    corpus = prepared or build_gospel_corpus(sword_root=sword, tei_dir=tei)
    base = load_nt_splits(root)
    splits = wider_splits(
        [unit.unit_id for unit in corpus.units],
        base,
        corpus_id=corpus.id,
    )
    bundle = build_bitext(corpus, splits)
    train = list(bundle.sentences_for(SplitRole.TRAIN))
    for sentence in extra_train or []:
        if sentence.split_role != SplitRole.TRAIN.value or not str(
            sentence.unit_id
        ).startswith("ot."):
            msg = f"extra training sentence is not an OT train pair: {sentence.unit_id}"
            raise DictionaryBuildError(msg)
        train.append(sentence)
    holdout_ids = set(splits.unit_ids_for(SplitRole.HOLDOUT))
    omission_ids = set(splits.unit_ids_for(SplitRole.OMISSION))
    train_ids = [sentence.unit_id for sentence in train]
    if any(unit_id in holdout_ids or unit_id in omission_ids for unit_id in train_ids):
        msg = "v3 holdout or omission verse entered Gospel training"
        raise DictionaryBuildError(msg)
    alignments = align_train_sentences(
        train,
        aligners=(AlignerName.EFLOMAL, AlignerName.FAST_ALIGN),
        repo_root=root,
    )
    links, context_counts = observed_links_from_alignments(alignments)
    dictionary: ParallelDictionary = build_dictionary(
        links,
        unit_ids=sorted(set(train_ids)),
        aligners=["eflomal", "fast_align"],
        provenance={
            "source": corpus.id,
            "replaces": V1_VERSION,
            "pilot_corpus_id": corpus.id,
            "split_id": splits.id,
            "bitext_checksum_sha256": bundle.checksum_sha256,
            "gospel_tei_sha256": gospel_tei_digest(tei),
            "train_sentences": len(train),
            "dev_sentences": len(bundle.sentences_for(SplitRole.DEV)),
            "holdout_units_excluded": len(holdout_ids),
            "coptic_segmentation": bundle.coptic_segmentation,
            "greek_tokenization": bundle.greek_tokenization,
            "greek_source_id": GREEK_SOURCE_ID,
            "sahidic_source_id": SAHIDIC_SOURCE_ID,
            "ot_train_sentences": len(extra_train or []),
            "nt_train_sentences": len(train) - len(extra_train or []),
            "license": OT_LICENSE if extra_train else "CC-BY-4.0",
            "swete_commit": SWETE_COMMIT if extra_train else "",
        },
        version=fit_version,
    )
    dictionary.context_counts = context_counts
    dev_sentences = bundle.sentences_for(SplitRole.DEV)
    dev_without = _dev_token_precision(dev_sentences, dictionary, use_context=False)
    dev_with = _dev_token_precision(dev_sentences, dictionary, use_context=True)
    # Keep context case choice only when held-out NT exact precision does not drop.
    dictionary.use_context = dev_with >= dev_without
    dictionary.provenance["dev_token_precision_without_context"] = round(dev_without, 6)
    dictionary.provenance["dev_token_precision_with_context"] = round(dev_with, 6)
    dictionary.provenance["use_context"] = dictionary.use_context
    diagnostic = nt_dev_snap_diagnostic(corpus, splits, dictionary)
    dictionary.snap_enabled = bool(diagnostic["snap_enabled"])
    return _apply_and_write(
        dictionary,
        root,
        output_dir or (root / "data" / "reconstructions" / fit_version),
        diagnostic=diagnostic,
        train_sentences=len(train),
    )


def run_ot_dictionary(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit NT plus Sahidic OT and Swete. Both OT texts are CC BY-SA 4.0."""
    from zetodos_thomas.dictionary.ot_bitext import iter_ot_sentences

    root = repo_root
    ot_sentences = iter_ot_sentences(
        root / "data" / "downloads" / "sahidic-ot" / "sahidic.ot_TT.zip",
        root / "data" / "downloads" / "lxx-swete" / "repo" / "data",
    )
    if len(ot_sentences) < 1000:
        msg = f"OT verse pairs are too few to be a real parallel ({len(ot_sentences)})"
        raise DictionaryBuildError(msg)
    corpus = build_nt_corpus(
        sword_root=root / "data" / "downloads" / "sword",
        tei_dirs=[gospel_tei_dir(root), nt_tei_dir(root)],
    )
    return run_gospel_dictionary(
        root,
        output_dir=output_dir,
        download=False,
        prepared=corpus,
        version="parallel-dict-v4",
        extra_train=ot_sentences,
    )


def run_nt_dictionary(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Re-fit on the Gospels plus the rest of the Horner x Vaticanus NT."""
    root = repo_root
    sword = root / "data" / "downloads" / "sword"
    corpus = build_nt_corpus(
        sword_root=sword,
        tei_dirs=[gospel_tei_dir(root), nt_tei_dir(root)],
    )
    return run_gospel_dictionary(
        root,
        output_dir=output_dir,
        download=False,
        prepared=corpus,
        version="parallel-dict-v3",
    )


def _apply_and_write(
    dictionary: ParallelDictionary,
    repo_root: Path,
    output_dir: Path,
    *,
    diagnostic: dict[str, object],
    train_sentences: int,
) -> dict[str, Any]:
    from zetodos_thomas.dictionary.apply import apply_tokens
    from zetodos_thomas.dictionary.score import score_fragments
    from zetodos_thomas.ingest.dilley_thomas import (
        default_corpus_path,
        load_thomas_corpus,
    )
    from zetodos_thomas.ingest.fragments import (
        fragments_corpus_path,
        load_fragment_corpus,
    )

    thomas = load_thomas_corpus(default_corpus_path(repo_root))
    units: list[dict[str, Any]] = []
    decisions: dict[int, list[Any]] = {}
    for unit in thomas.units:
        outputs = apply_tokens(unit.normalized_text.split(), dictionary)
        decisions[unit.chapter] = outputs
        units.append(
            {
                "unit_id": unit.unit_id,
                "logion": unit.chapter,
                "tokens": [_token_row(item) for item in outputs],
            }
        )
    fragments = load_fragment_corpus(fragments_corpus_path(repo_root))
    summary = score_fragments(decisions, fragments)
    sufficient, reason = _sufficient(summary)
    metrics: dict[str, Any] = {
        "id": f"{VERSION}-metrics",
        "version": VERSION,
        "uses_thomas_greek_as_training": False,
        "uses_handwritten_glosses": False,
        "uses_verified_loanword_table": False,
        "frozen_before_fragment_score": True,
        "snap_enabled": dictionary.snap_enabled,
        "nt_dev_diagnostic": diagnostic,
        "train_sentences": train_sentences,
        "loanword_stem_match_rate": summary["loanword_stem_match_rate"],
        "loanword_exact_match_rate": summary["loanword_exact_match_rate"],
        "loanword_tokens": summary["loanword_tokens"],
        "content_type_recall": summary["content_type_recall"],
        "content_emission_precision": summary["content_emission_precision"],
        "fragment_type_recall": summary["fragment_type_recall"],
        "abstain_rate": summary["abstain_rate"],
        "approach_sufficient": sufficient,
        "approach_sufficient_reason": reason,
        "next_step": _next_step(sufficient),
        "micro": {
            key: value for key, value in summary.items() if key != "by_manuscript"
        },
        "by_manuscript": summary["by_manuscript"],
        "dictionary_entry_count": len(dictionary.emitted_entries()),
        "notes": [
            "Train text is the four Horner x Vaticanus Gospels.",
            "v3 dev, holdout, and John 8 omission verses are not training.",
            "Both eflomal and fast_align must support an emitted statistical pair.",
            "Thomas Greek was loaded only after the dictionary was frozen.",
            "No handwritten glosses and no verified-loanword table.",
        ],
    }
    _write_outputs(output_dir, dictionary, units, metrics)
    metrics["output_dir"] = str(output_dir)
    return metrics

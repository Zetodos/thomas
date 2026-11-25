# SPDX-License-Identifier: Apache-2.0
"""Ingest zero-leakage 1st/2nd-century Koine prose (Apostolic Fathers & First1KGreek).

Strictly loads only pinned CC-BY-SA-4.0 sources:
- ``jtauber/apostolic-fathers`` (``texts/*.txt``)
- ``OpenGreekAndLatin/First1KGreek`` Epictetus (``tlg0557``) & Philo (``tlg0018``)

Used for monolingual Koine LM training, span-denoising pretraining, and reverse
synthetic back-translation in ``model-v3``. Contains zero Gospel of Thomas or
P.Oxy. material.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

from zetodos_thomas.ingest.holdouts import assert_clean_training_unit
from zetodos_thomas.ingest.ot_swete import _LATIN_LOOKALIKES

APOSTOLIC_FATHERS_SOURCE_ID = "apostolic-fathers"
FIRST1K_SOURCE_ID = "first1k-koine-prose"

# Explicit TLG author allowlist (Epictetus + Philo of Alexandria)
ALLOWED_FIRST1K_PREFIXES: tuple[str, ...] = ("tlg0557.", "tlg0018.")

_SENT_SPLIT_RE = re.compile(r"(?<=[.;;·?!])\s+")
_NUM_PREFIX_RE = re.compile(r"^(?:[IVXLCDM]+|\d+(?:\.\d+)*)\.?\s+")
_NUM_TOK_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")
_GREEK_CHAR_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_STRIP_BRACKETS_RE = re.compile(r"[\[\]<>()†‡*«»\"“”„]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_SKIP_TEI_TAGS = frozenset({"note", "head", "bibl", "teiHeader", "front", "back"})


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean_koine_sentence(raw: str) -> str:
    """Clean editorial artifacts and verify pure Greek Koine sentence."""
    txt = _STRIP_BRACKETS_RE.sub(" ", raw)
    txt = _NUM_PREFIX_RE.sub("", txt.strip())
    txt = _MULTI_SPACE_RE.sub(" ", txt).strip()
    if not txt:
        return ""
    # Reject headings or Latin apparatus notes before lookalike repair
    n_greek = len(_GREEK_CHAR_RE.findall(txt))
    n_latin = len(_LATIN_CHAR_RE.findall(txt))
    if n_greek < 10 or n_latin > max(2, n_greek // 8):
        return ""
    # Repair occasional single-char Latin lookalikes inside Greek words
    tokens: list[str] = []
    for tok in txt.split():
        if _GREEK_CHAR_RE.search(tok):
            tok = tok.translate(_LATIN_LOOKALIKES)
        elif _LATIN_CHAR_RE.search(tok) or _NUM_TOK_RE.match(tok):
            continue
        tok = _nfc(tok)
        if tok:
            tokens.append(tok)
    return " ".join(tokens)


def _segment_koine_prose(
    text: str, *, min_words: int = 6, max_words: int = 38
) -> list[str]:
    """Split a Greek prose paragraph into sentence/clause-length Koine units."""
    out: list[str] = []
    for raw_sent in _SENT_SPLIT_RE.split(text):
        cleaned = _clean_koine_sentence(raw_sent)
        if not cleaned:
            continue
        words = cleaned.split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            out.append(cleaned)
            continue
        # Split long philosophical/patristic periods at commas or mid-clauses
        buf: list[str] = []
        for w in words:
            buf.append(w)
            at_comma = w.endswith((",", "·", ";", ";"))
            if (at_comma and len(buf) >= min_words) or len(buf) >= max_words:
                if len(buf) >= min_words:
                    out.append(" ".join(buf))
                buf = []
        if len(buf) >= min_words:
            out.append(" ".join(buf))
    return out


def load_apostolic_fathers_sentences(
    af_dir: Path, *, min_words: int = 6, max_words: int = 38
) -> list[tuple[str, str]]:
    """Load clean ``(unit_id, greek_sentence)`` pairs from ``apostolic-fathers``."""
    texts_dir = af_dir / "texts"
    if not texts_dir.is_dir():
        msg = f"missing apostolic-fathers texts directory: {texts_dir}"
        raise FileNotFoundError(msg)

    results: list[tuple[str, str]] = []
    for txt_path in sorted(texts_dir.glob("*.txt")):
        stem = txt_path.stem.lower()
        assert_clean_training_unit(
            f"af.{stem}.0", source_id=APOSTOLIC_FATHERS_SOURCE_ID
        )
        raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
        idx = 0
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip ALL-CAPS work titles and bare Roman chapter numerals
            if stripped.isupper() and len(stripped.split()) <= 8:
                continue
            for sent in _segment_koine_prose(
                stripped, min_words=min_words, max_words=max_words
            ):
                uid = f"af.{stem}.{idx}"
                assert_clean_training_unit(uid, source_id=APOSTOLIC_FATHERS_SOURCE_ID)
                results.append((uid, sent))
                idx += 1
    return results


def _iter_tei_body_paragraphs(xml_path: Path) -> list[str]:
    """Extract clean paragraph strings from a First1KGreek TEI-XML body."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    def _collect_text(elem: ET.Element, buf: list[str]) -> None:
        loc = _local(elem.tag)
        if loc in _SKIP_TEI_TAGS:
            return
        if elem.text:
            buf.append(elem.text)
        for child in elem:
            _collect_text(child, buf)
            if child.tail:
                buf.append(child.tail)

    paragraphs: list[str] = []
    for elem in root.iter():
        if _local(elem.tag) == "p":
            buf: list[str] = []
            _collect_text(elem, buf)
            joined = _MULTI_SPACE_RE.sub(" ", "".join(buf)).strip()
            if joined:
                paragraphs.append(joined)
    return paragraphs


def load_first1k_koine_sentences(
    first1k_dir: Path, *, min_words: int = 6, max_words: int = 38
) -> list[tuple[str, str]]:
    """Load clean ``(unit_id, greek_sentence)`` pairs from pinned First1KGreek XMLs."""
    if not first1k_dir.is_dir():
        msg = f"missing first1k-koine-prose directory: {first1k_dir}"
        raise FileNotFoundError(msg)

    results: list[tuple[str, str]] = []
    for xml_path in sorted(first1k_dir.glob("*.xml")):
        fname = xml_path.name
        if not fname.startswith(ALLOWED_FIRST1K_PREFIXES):
            msg = f"non-allowlisted First1KGreek file rejected: {fname}"
            raise ValueError(msg)
        stem = xml_path.stem.lower()
        idx = 0
        for para in _iter_tei_body_paragraphs(xml_path):
            for sent in _segment_koine_prose(
                para, min_words=min_words, max_words=max_words
            ):
                uid = f"f1k.{stem}.{idx}"
                assert_clean_training_unit(uid, source_id=FIRST1K_SOURCE_ID)
                results.append((uid, sent))
                idx += 1
    return results


def load_all_patristic_koine_sentences(
    downloads_dir: Path, *, min_words: int = 6, max_words: int = 38
) -> list[tuple[str, str]]:
    """Load deduplicated patristic & 1st-century Koine Greek sentences if present."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    af_dir = downloads_dir / "apostolic-fathers"
    if (af_dir / "texts").is_dir():
        for uid, sent in load_apostolic_fathers_sentences(
            af_dir, min_words=min_words, max_words=max_words
        ):
            if sent not in seen:
                seen.add(sent)
                out.append((uid, sent))
    f1k_dir = downloads_dir / "first1k-koine-prose"
    if f1k_dir.is_dir():
        for uid, sent in load_first1k_koine_sentences(
            f1k_dir, min_words=min_words, max_words=max_words
        ):
            if sent not in seen:
                seen.add(sent)
                out.append((uid, sent))
    return out

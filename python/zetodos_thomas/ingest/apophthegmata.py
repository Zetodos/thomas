# SPDX-License-Identifier: Apache-2.0
"""Ingest Coptic SCRIPTORIUM Apophthegmata Patrum (.tt) with English translations.

Strictly loads the pinned CC-BY-4.0 ``AP/apophthegmata.patrum_TT/*.tt`` sayings
for non-biblical Sahidic span-denoising pretraining and ``cop -> eng`` multi-task
pivot training in ``model-v3``. Contains zero Gospel of Thomas or P.Oxy. material.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.ingest.holdouts import assert_clean_training_unit
from zetodos_thomas.schemas.ids import token_id
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit
from zetodos_thomas.schemas.source import UnitKind

APOPHTHEGMATA_SOURCE_ID = "scriptorium-apophthegmata"

_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:\-]*)="([^"]*)"')
_ORIG_RE = re.compile(r'<orig\s+orig="([^"]*)">')
_AP_TAG_RE = re.compile(
    r"<chapter_n\b[^>]*>"
    r"|<verse_n\b[^>]*>"
    r"|</verse_n>"
    r"|<translation\b[^>]*>"
    r"|</translation>"
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
_MULTI_SPACE_RE = re.compile(r"\s+")
_ENG_CLEAN_RE = re.compile(r"[\[\]<>]+")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _parse_attrs(blob: str) -> dict[str, str]:
    return {k: html.unescape(v) for k, v in _ATTR_RE.findall(blob)}


def _clean_english_translation(raw: str) -> str:
    txt = _ENG_CLEAN_RE.sub("", html.unescape(raw))
    txt = txt.replace("…", " ").replace("...", " ")
    txt = _MULTI_SPACE_RE.sub(" ", txt).strip()
    # Reject placeholder / untranslated annotations
    if len(txt.split()) < 3:
        return ""
    return txt


@dataclass(frozen=True)
class ApophthegmataUnit:
    """One Sahidic saying verse from Apophthegmata Patrum + optional English gloss."""

    coptic: CopticUnit
    english: str | None = None


@dataclass
class _RawApToken:
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


def parse_apophthegmata_tt(text: str, *, doc_id: str) -> list[ApophthegmataUnit]:
    """Parse one Coptic SCRIPTORIUM Apophthegmata Patrum ``*.tt`` document."""
    units: list[ApophthegmataUnit] = []
    cur_chapter = 1
    cur_verse: int | None = None
    cur_translation = ""
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
    raw_tokens: list[_RawApToken] = []
    fallback_seq = 0

    def flush_norm() -> None:
        nonlocal in_norm, cur_attrs, cur_orig, cur_morph, cur_lang, cur_multiword
        if not in_norm or cur_verse is None:
            in_norm = False
            return
        norm_val = _nfc(cur_attrs.get("norm", cur_orig))
        lemma_val = _nfc(cur_attrs.get("lemma", norm_val))
        pos_val = cur_attrs.get("pos", "UNK")
        raw_tokens.append(
            _RawApToken(
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
        nonlocal cur_verse, cur_translation, orig_buf, norm_buf, group_seq
        nonlocal in_orig_group, active_multiword, pending_orig, raw_tokens
        flush_norm()
        if cur_verse is None or not raw_tokens:
            cur_verse = None
            cur_translation = ""
            orig_buf = []
            norm_buf = []
            group_seq = -1
            in_orig_group = False
            active_multiword = None
            pending_orig = None
            raw_tokens = []
            return
        unit_id = f"ap.{doc_id}.{cur_chapter}.{cur_verse}"
        assert_clean_training_unit(unit_id, source_id=APOPHTHEGMATA_SOURCE_ID)
        xml_to_seq = {
            item.xml_id: seq for seq, item in enumerate(raw_tokens) if item.xml_id
        }
        morphemes: list[CopticMorpheme] = []
        for seq, item in enumerate(raw_tokens):
            head_ref = item.head_ref.lstrip("#")
            head_seq = xml_to_seq.get(head_ref) if head_ref else None
            morphemes.append(
                CopticMorpheme(
                    id=token_id(APOPHTHEGMATA_SOURCE_ID, unit_id, seq),
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
        c_unit = CopticUnit(
            unit_id=unit_id,
            source_id=APOPHTHEGMATA_SOURCE_ID,
            book=f"ap.{doc_id.lower()}",
            chapter=cur_chapter,
            verse=cur_verse,
            kind=UnitKind.VERSE,
            original_text=_nfc(" ".join(orig_buf)),
            normalized_text=_nfc(" ".join(norm_buf)),
            morphemes=morphemes,
        )
        eng = _clean_english_translation(cur_translation) or None
        units.append(ApophthegmataUnit(coptic=c_unit, english=eng))
        cur_verse = None
        cur_translation = ""
        orig_buf = []
        norm_buf = []
        group_seq = -1
        in_orig_group = False
        active_multiword = None
        pending_orig = None
        raw_tokens = []

    for match in _AP_TAG_RE.finditer(text):
        tag = match.group(0)
        if tag.startswith("<chapter_n"):
            attrs = _parse_attrs(tag[10:-1])
            ch_s = attrs.get("chapter_n", "")
            if ch_s.isdigit():
                cur_chapter = int(ch_s)
            continue
        if tag.startswith("<verse_n"):
            flush_verse()
            attrs = _parse_attrs(tag[8:-1])
            v_str = attrs.get("verse_n", "")
            if v_str.isdigit():
                cur_verse = int(v_str)
            else:
                fallback_seq += 1
                cur_verse = fallback_seq
            continue
        if tag.startswith("</verse_n"):
            flush_verse()
            continue
        if tag.startswith("<translation"):
            attrs = _parse_attrs(tag[12:-1])
            tr = attrs.get("translation", "").strip()
            if cur_verse is None:
                # Some AP files wrap sentences in <translation> without <verse_n>
                fallback_seq += 1
                cur_verse = fallback_seq
            if tr:
                cur_translation = (
                    f"{cur_translation} {tr}".strip() if cur_translation else tr
                )
            continue
        if tag.startswith("</translation"):
            continue
        if cur_verse is None:
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
    return units


def load_apophthegmata_corpus(ap_root: Path) -> list[ApophthegmataUnit]:
    """Load all Sahidic Apophthegmata Patrum units + English translations."""
    tt_dir = ap_root / "AP" / "apophthegmata.patrum_TT"
    if not tt_dir.is_dir():
        tt_dir = ap_root
    if not tt_dir.is_dir():
        msg = f"missing Apophthegmata Patrum TT directory: {tt_dir}"
        raise FileNotFoundError(msg)

    units: list[ApophthegmataUnit] = []
    for tt_path in sorted(tt_dir.rglob("*.tt")):
        doc_id = tt_path.stem.replace(" ", "_")
        text = tt_path.read_text(encoding="utf-8", errors="replace")
        units.extend(parse_apophthegmata_tt(text, doc_id=doc_id))
    return units

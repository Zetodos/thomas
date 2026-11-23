# SPDX-License-Identifier: Apache-2.0
"""Ingest Dilley / SCRIPTORIUM annotated Coptic Gospel of Thomas (TreeTagger SGML).

Preserves full per-morpheme annotations across ``thomas.0``-``thomas.114``:
``orig_group``/``norm_group`` boundaries, ``orig``, ``norm``, ``lemma``, ``pos``,
``<morph>`` sub-pieces, ``<lang>`` loanword tags, ``<multiword>`` idioms, and
Universal Dependencies ``head`` / ``func`` relations.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION, token_id
from zetodos_thomas.schemas.linguistic import CopticCorpus, CopticMorpheme, CopticUnit
from zetodos_thomas.schemas.source import UnitKind

SOURCE_ID = "dilley-thomas"
CORPUS_ID = "dilley-thomas-v6.2.0"
BOOK = "thomas"
EXPECTED_LOGIA = 114  # sayings 1-114; unit 0 is the prologue
EXPECTED_UNITS = EXPECTED_LOGIA + 1
INGEST_LEVEL = "token-complete"

TT_FILENAME = "thomas_gospel.tt"
TT_RELATIVE_PATH = "thomas-gospel/thomas.gospel_TT/thomas_gospel.tt"
TT_TAG_URL = (
    "https://raw.githubusercontent.com/CopticScriptorium/corpora/"
    f"v6.2.0/{TT_RELATIVE_PATH}"
)

_CHAPTER_OPEN_RE = re.compile(r'<chapter_n\s+chapter_n="([^"]*)">')
_ORIG_RE = re.compile(r'<orig\s+orig="([^"]*)">')
_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:\-]*)="([^"]*)"')
_TAG_RE = re.compile(
    r"</?chapter_n\b[^>]*>"
    r"|<orig_group\b[^>]*>"
    r"|</orig_group>"
    r"|<multiword\b[^>]*>"
    r"|</multiword>"
    r"|<orig\b[^>]*>"
    r"|<norm\b[^>]*>"
    r"|</norm>"
    r"|<morph\b[^>]*>"
    r"|<lang\b[^>]*>"
)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _parse_attrs(attr_blob: str) -> dict[str, str]:
    return {key: value for key, value in _ATTR_RE.findall(attr_blob)}


@dataclass
class _RawToken:
    xml_id: str
    raw_head: str
    group_seq: int
    orig: str
    norm: str
    lemma: str
    pos: str
    morph: tuple[str, ...]
    lang: str | None
    multiword: str | None
    dep_rel: str | None


def expected_unit_ids() -> list[str]:
    """Return the canonical ordered unit ids ``thomas.0`` … ``thomas.114``."""
    return [f"{BOOK}.{n}" for n in range(EXPECTED_UNITS)]


def validate_thomas_unit_sequence(units: list[CopticUnit]) -> None:
    """Require exactly ``thomas.0`` through ``thomas.114`` in order."""
    actual_ids = [u.unit_id for u in units]
    expected = expected_unit_ids()
    if actual_ids != expected:
        msg = (
            "Thomas corpus must contain exactly "
            f"{expected[0]} through {expected[-1]} in order "
            f"(got {len(actual_ids)} units"
        )
        if actual_ids:
            msg += f"; first={actual_ids[0]!r} last={actual_ids[-1]!r}"
        msg += ")"
        raise ValueError(msg)
    for sequence, unit in enumerate(units):
        if unit.chapter != sequence:
            msg = (
                f"unit {unit.unit_id} chapter must equal sequence index "
                f"{sequence} (found {unit.chapter})"
            )
            raise ValueError(msg)
        if unit.book != BOOK:
            msg = f"unit {unit.unit_id} book must be {BOOK!r}"
            raise ValueError(msg)
        if sequence == 0 and unit.kind is not UnitKind.PROLOGUE:
            msg = f"unit {unit.unit_id} kind must be prologue"
            raise ValueError(msg)
        if sequence > 0 and unit.kind is not UnitKind.LOGION:
            msg = f"unit {unit.unit_id} kind must be logion"
            raise ValueError(msg)
        if not unit.morphemes:
            msg = f"unit {unit.unit_id} must carry annotated morphemes"
            raise ValueError(msg)


def parse_tt(
    path: Path,
    *,
    source_id: str = SOURCE_ID,
    expect_complete: bool = True,
) -> CopticCorpus:
    """Parse Dilley SCRIPTORIUM ``*.tt`` into a token-complete ``CopticCorpus``."""
    raw = path.read_text(encoding="utf-8")
    body_start = raw.find("<pb_xml_id")
    body = raw if body_start < 0 else raw[body_start:]

    units: list[CopticUnit] = []
    current_logion: int | None = None
    skip_chapter = False
    orig_buf: list[str] = []
    norm_buf: list[str] = []
    group_seq = -1
    in_orig_group = False
    active_multiword: str | None = None
    pending_orig: str | None = None

    # Active <norm> accumulators
    in_norm = False
    cur_attrs: dict[str, str] = {}
    cur_orig: str = ""
    cur_morph: list[str] = []
    cur_lang: str | None = None
    cur_multiword: str | None = None
    cur_group_seq: int = 0
    raw_tokens: list[_RawToken] = []

    def flush_norm() -> None:
        nonlocal in_norm, cur_attrs, cur_orig, cur_morph, cur_lang, cur_multiword
        if not in_norm or current_logion is None:
            in_norm = False
            return
        norm_val = _nfc(cur_attrs.get("norm", cur_orig))
        lemma_val = _nfc(cur_attrs.get("lemma", norm_val))
        pos_val = cur_attrs.get("pos", "UNK")
        xml_id = cur_attrs.get("xml:id", "")
        raw_head = cur_attrs.get("head", "")
        dep_rel = cur_attrs.get("func") or None
        raw_tokens.append(
            _RawToken(
                xml_id=xml_id,
                raw_head=raw_head,
                group_seq=cur_group_seq,
                orig=cur_orig,
                norm=norm_val,
                lemma=lemma_val,
                pos=pos_val,
                morph=tuple(_nfc(m) for m in cur_morph if m),
                lang=cur_lang,
                multiword=cur_multiword,
                dep_rel=dep_rel,
            )
        )
        norm_buf.append(norm_val)
        in_norm = False
        cur_attrs = {}
        cur_orig = ""
        cur_morph = []
        cur_lang = None
        cur_multiword = None

    def flush_chapter() -> None:
        nonlocal current_logion, skip_chapter, orig_buf, norm_buf, group_seq
        nonlocal in_orig_group, active_multiword, pending_orig, raw_tokens
        flush_norm()
        if current_logion is None:
            skip_chapter = False
            pending_orig = None
            raw_tokens = []
            return
        unit_id = f"{BOOK}.{current_logion}"
        kind = UnitKind.PROLOGUE if current_logion == 0 else UnitKind.LOGION
        xml_to_seq: dict[str, int] = {}
        for seq, item in enumerate(raw_tokens):
            if item.xml_id:
                xml_to_seq[item.xml_id] = seq

        morphemes: list[CopticMorpheme] = []
        for seq, item in enumerate(raw_tokens):
            head_seq: int | None = None
            if item.raw_head.startswith("#"):
                head_seq = xml_to_seq.get(item.raw_head[1:])
            morphemes.append(
                CopticMorpheme(
                    id=token_id(source_id, unit_id, seq),
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

        units.append(
            CopticUnit(
                unit_id=unit_id,
                source_id=source_id,
                book=BOOK,
                chapter=current_logion,
                verse=0,
                kind=kind,
                original_text=_nfc(" ".join(orig_buf)),
                normalized_text=_nfc(" ".join(norm_buf)),
                morphemes=morphemes,
            )
        )
        current_logion = None
        skip_chapter = False
        orig_buf = []
        norm_buf = []
        group_seq = -1
        in_orig_group = False
        active_multiword = None
        pending_orig = None
        raw_tokens = []

    for match in _TAG_RE.finditer(body):
        tag = match.group(0)
        if tag.startswith("<chapter_n"):
            flush_chapter()
            m_ch = _CHAPTER_OPEN_RE.match(tag)
            if m_ch is None:
                msg = f"unparseable chapter tag: {tag}"
                raise ValueError(msg)
            label = m_ch.group(1)
            if label.isdigit():
                current_logion = int(label)
                skip_chapter = False
            else:
                current_logion = None
                skip_chapter = True
            continue
        if tag.startswith("</chapter_n"):
            flush_chapter()
            continue
        if skip_chapter or current_logion is None:
            continue
        if tag.startswith("<orig_group"):
            group_seq += 1
            in_orig_group = True
            attrs = _parse_attrs(tag[11:-1])
            grp = _nfc(attrs.get("orig_group", ""))
            if grp:
                orig_buf.append(grp)
            continue
        if tag.startswith("</orig_group"):
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
            if m_orig is None:
                msg = f"unparseable orig tag: {tag}"
                raise ValueError(msg)
            orig_val = m_orig.group(1)
            pending_orig = orig_val
            if not in_orig_group:
                group_seq += 1
                orig_buf.append(_nfc(orig_val))
            continue
        if tag.startswith("<norm"):
            if pending_orig is None:
                msg = "norm token without preceding orig"
                raise ValueError(msg)
            flush_norm()
            in_norm = True
            cur_attrs = _parse_attrs(tag[5:-1])
            cur_orig = _nfc(pending_orig)
            cur_morph = []
            cur_lang = None
            cur_multiword = active_multiword
            cur_group_seq = max(group_seq, 0)
            pending_orig = None
            continue
        if tag.startswith("</norm"):
            flush_norm()
            continue
        if tag.startswith("<morph"):
            if in_norm:
                attrs = _parse_attrs(tag[6:-1])
                piece = attrs.get("morph", "")
                if piece:
                    cur_morph.append(piece)
            continue
        if tag.startswith("<lang"):
            if in_norm:
                attrs = _parse_attrs(tag[5:-1])
                lang_val = attrs.get("lang", "").strip()
                if lang_val:
                    cur_lang = lang_val
            continue

    flush_chapter()

    if expect_complete:
        validate_thomas_unit_sequence(units)
    elif not units:
        msg = f"no logion units parsed from {path}"
        raise ValueError(msg)

    return CopticCorpus(
        id=CORPUS_ID,
        title=(
            "Dilley / Coptic SCRIPTORIUM Gospel of Thomas "
            "(NH Codex II; annotated CC BY 4.0) — token-complete ingest"
        ),
        source_id=source_id,
        id_scheme_version=ID_SCHEME_VERSION,
        annotation_level=INGEST_LEVEL,
        units=units,
        notes=(
            f"Token-complete SCRIPTORIUM ingest ({INGEST_LEVEL}): preserves "
            "orig_group, orig, norm, lemma, pos, morph, lang, multiword, and UD "
            "head_seq/dep_rel across thomas.0-thomas.114 (colophon subtitle omitted)."
        ),
    )


def write_thomas_corpus(corpus: CopticCorpus, path: Path) -> None:
    """Serialize annotated Thomas corpus as compact UTF-8 JSON (<1024 KiB)."""
    payload = corpus.model_dump(mode="json", exclude_defaults=True)
    for unit in payload.get("units", []):
        for morph in unit.get("morphemes", []):
            # Deterministically re-hydrated on load by CopticUnit._check_sequences.
            morph.pop("id", None)
            morph.pop("unit_id", None)
            morph.pop("seq", None)
            if morph.get("orig") == morph.get("norm"):
                morph.pop("orig", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_thomas_corpus(path: Path) -> CopticCorpus:
    """Load a committed Thomas corpus JSON and enforce token-complete sequence."""
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus = CopticCorpus.model_validate(data)
    validate_thomas_unit_sequence(corpus.units)
    return corpus


def default_corpus_path(repo_root: Path) -> Path:
    """Checkout path for the bundled Dilley Thomas import."""
    return repo_root / "data" / "sources" / "thomas" / f"{CORPUS_ID}.json"


def default_tt_path(repo_root: Path) -> Path:
    """Expected local path of the downloaded TreeTagger SGML after setup."""
    return repo_root / "data" / "downloads" / "dilley-thomas" / TT_FILENAME

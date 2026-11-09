# SPDX-License-Identifier: Apache-2.0
"""Ingest Dilley / SCRIPTORIUM annotated Coptic Gospel of Thomas (TreeTagger SGML).

Produces a **unit-level** ``ThomasCorpus`` (shared JAK-32 model) with recoverable
original vs normalized TextUnit layers for ``thomas.0``-``thomas.114``.

Token/span lists stay empty here: this import is **not** token-complete and
**not** alignment-ready. Full SCRIPTORIUM per-token orig↔norm pairs (and
orig_group relationships) must be minted later with JAK-32 ``token_id`` /
``span_id`` helpers — never via ``whitespace_tokens`` (orig is continuous while
norm is morpheme-segmented).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from zetodos_thomas.schemas.alignment import ThomasCorpus
from zetodos_thomas.schemas.ids import ID_SCHEME_VERSION
from zetodos_thomas.schemas.source import TextUnit, UnitKind

SOURCE_ID = "dilley-thomas"
CORPUS_ID = "dilley-thomas-v6.2.0"
BOOK = "thomas"
EXPECTED_LOGIA = 114  # sayings 1-114; unit 0 is the prologue
EXPECTED_UNITS = EXPECTED_LOGIA + 1  # prologue + 114 logia
INGEST_LEVEL = "unit-level"
UNIT_LEVEL_LABEL = "UNIT-LEVEL INGEST ONLY — not token-complete and not alignment-ready"

# SCRIPTORIUM TreeTagger SGML (most complete per-token orig/norm layers upstream).
TT_FILENAME = "thomas_gospel.tt"
TT_RELATIVE_PATH = "thomas-gospel/thomas.gospel_TT/thomas_gospel.tt"
TT_TAG_URL = (
    "https://raw.githubusercontent.com/CopticScriptorium/corpora/"
    f"v6.2.0/{TT_RELATIVE_PATH}"
)

_CHAPTER_OPEN_RE = re.compile(r'<chapter_n\s+chapter_n="([^"]*)">')
_ORIG_RE = re.compile(r'<orig\s+orig="([^"]*)">')
_ATTR_RE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:\-]*)="([^"]*)"')


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _parse_attrs(attr_blob: str) -> dict[str, str]:
    return {key: value for key, value in _ATTR_RE.findall(attr_blob)}


def expected_unit_ids() -> list[str]:
    """Return the canonical ordered unit ids ``thomas.0`` … ``thomas.114``."""
    return [f"{BOOK}.{n}" for n in range(EXPECTED_UNITS)]


def validate_thomas_unit_sequence(units: list[TextUnit]) -> None:
    """Require exactly ``thomas.0`` through ``thomas.114`` in order.

    Rejects missing prologue, missing/duplicated logia, or reordered chapters.
    """
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
        if sequence == 0:
            if unit.kind is not UnitKind.PROLOGUE:
                msg = f"unit {unit.unit_id} kind must be prologue"
                raise ValueError(msg)
        elif unit.kind is not UnitKind.LOGION:
            msg = f"unit {unit.unit_id} kind must be logion"
            raise ValueError(msg)


def parse_tt(
    path: Path,
    *,
    source_id: str = SOURCE_ID,
    expect_complete: bool = True,
) -> ThomasCorpus:
    """Parse a SCRIPTORIUM ``*.tt`` into unit-level ``ThomasCorpus`` (no tokens).

    ``original_text`` concatenates manuscript-oriented ``orig`` surfaces
    (diplomatic stream). ``normalized_text`` joins Dilley's ``norm`` analysis
    forms with spaces. These are recoverable layers of one edition — not
    competing Coptic texts.

    ``tokens`` / ``spans`` are left empty on purpose (unit-level ingest). Do not
    call ``whitespace_tokens`` on these units.

    When ``expect_complete`` is true (default), require ``thomas.0``-``thomas.114``.
    Fixtures may set it false.
    """
    raw = path.read_text(encoding="utf-8")
    # Drop the leading <meta ...> line; it embeds long notes with angle brackets.
    body_start = raw.find("<pb_xml_id")
    body = raw if body_start < 0 else raw[body_start:]

    units: list[TextUnit] = []
    current_logion: int | None = None
    skip_chapter = False
    orig_buf: list[str] = []
    norm_buf: list[str] = []
    pending_orig: str | None = None

    def flush() -> None:
        nonlocal current_logion, skip_chapter, orig_buf, norm_buf, pending_orig
        if current_logion is None:
            skip_chapter = False
            pending_orig = None
            return
        unit_id = f"{BOOK}.{current_logion}"
        kind = UnitKind.PROLOGUE if current_logion == 0 else UnitKind.LOGION
        units.append(
            TextUnit(
                source_id=source_id,
                unit_id=unit_id,
                book=BOOK,
                chapter=current_logion,
                verse=0,
                original_text=_nfc("".join(orig_buf)),
                normalized_text=_nfc(" ".join(norm_buf)),
                sequence=len(units),
                kind=kind,
            )
        )
        current_logion = None
        skip_chapter = False
        orig_buf = []
        norm_buf = []
        pending_orig = None

    for match in re.finditer(
        r"</?chapter_n\b[^>]*>|<orig\b[^>]*>|<norm\b[^>]*>",
        body,
    ):
        tag = match.group(0)
        if tag.startswith("<chapter_n"):
            flush()
            m_ch = _CHAPTER_OPEN_RE.match(tag)
            if m_ch is None:
                msg = f"unparseable chapter tag: {tag}"
                raise ValueError(msg)
            label = m_ch.group(1)
            # SCRIPTORIUM uses a non-numeric "subtitle" chapter for the colophon;
            # keep units thomas.0-thomas.114 only (JAK-30).
            if label.isdigit():
                current_logion = int(label)
                skip_chapter = False
            else:
                current_logion = None
                skip_chapter = True
            continue
        if tag.startswith("</chapter_n"):
            flush()
            continue
        if tag.startswith("<orig"):
            if skip_chapter or current_logion is None:
                pending_orig = None
                continue
            m_orig = _ORIG_RE.match(tag)
            if m_orig is None:
                msg = f"unparseable orig tag: {tag}"
                raise ValueError(msg)
            pending_orig = m_orig.group(1)
            continue
        if tag.startswith("<norm"):
            if skip_chapter or current_logion is None:
                pending_orig = None
                continue
            if pending_orig is None:
                msg = "norm token without preceding orig"
                raise ValueError(msg)
            attrs = _parse_attrs(tag[5:-1])
            surface = _nfc(pending_orig)
            normalized = _nfc(attrs.get("norm", surface))
            orig_buf.append(surface)
            norm_buf.append(normalized)
            pending_orig = None

    flush()

    if expect_complete:
        validate_thomas_unit_sequence(units)
    elif not units:
        msg = f"no logion units parsed from {path}"
        raise ValueError(msg)

    return ThomasCorpus(
        id=CORPUS_ID,
        title=(
            "Dilley / Coptic SCRIPTORIUM Gospel of Thomas "
            "(NH Codex II; annotated CC BY 4.0) — unit-level ingest"
        ),
        source_id=source_id,
        id_scheme_version=ID_SCHEME_VERSION,
        units=units,
        tokens=[],
        spans=[],
        notes=(
            f"{UNIT_LEVEL_LABEL} (JAK-30 / ingest_level={INGEST_LEVEL}). "
            "Original = SCRIPTORIUM orig surfaces (manuscript-oriented); "
            "normalized = Dilley/SCRIPTORIUM norm analysis forms. "
            "Units are exactly thomas.0 (kind=prologue) through thomas.114 "
            "(kind=logion) in order. Non-numeric SCRIPTORIUM "
            "chapter_n=subtitle (colophon) is omitted. "
            "tokens[]/spans[] are empty by design — do not invent "
            "thomas.N.stub0 or other non-v1 ids, and do not use "
            "whitespace_tokens on these units. Full SCRIPTORIUM token/group "
            "relationships (~6.8k annotated tokens, orig_group boundaries, "
            "POS/lemma) remain for a later ingest that mints "
            "v1:dilley-thomas/thomas.N/t/k from per-token orig↔norm pairs "
            "(JAK-32). Do not substitute Layton/Attridge or Bethge continuous "
            "critical Greek."
        ),
    )


def write_thomas_corpus(corpus: ThomasCorpus, path: Path) -> None:
    """Serialize a Thomas corpus as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        corpus.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_thomas_corpus(path: Path) -> ThomasCorpus:
    """Load a committed Thomas corpus JSON and enforce the unit sequence."""
    data = json.loads(path.read_text(encoding="utf-8"))
    corpus = ThomasCorpus.model_validate(data)
    validate_thomas_unit_sequence(corpus.units)
    if INGEST_LEVEL not in corpus.notes or "not alignment-ready" not in corpus.notes:
        msg = (
            "Thomas corpus notes must label unit-level ingest as "
            "not alignment-ready (JAK-30)"
        )
        raise ValueError(msg)
    if corpus.tokens or corpus.spans:
        msg = (
            "unit-level Thomas corpus must leave tokens[] and spans[] empty; "
            "full SCRIPTORIUM token ingest comes later under JAK-32 IDs"
        )
        raise ValueError(msg)
    return corpus


def default_corpus_path(repo_root: Path) -> Path:
    """Checkout path for the bundled Dilley Thomas import."""
    return repo_root / "data" / "sources" / "thomas" / f"{CORPUS_ID}.json"


def default_tt_path(repo_root: Path) -> Path:
    """Expected local path of the downloaded TreeTagger SGML after setup."""
    return repo_root / "data" / "downloads" / "dilley-thomas" / TT_FILENAME

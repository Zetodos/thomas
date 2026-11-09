# SPDX-License-Identifier: Apache-2.0
"""Offline tests for token/span/parallel IDs and AlignmentDocument validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from zetodos_thomas.align.tokenize import (
    TokenCountMismatchError,
    whitespace_token_counts,
    whitespace_tokens,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.schemas.alignment import (
    AgreementDegree,
    AlignmentDocument,
    Parallel,
    ParallelRelation,
    Span,
    SpanKind,
    ThomasCorpus,
    Token,
)
from zetodos_thomas.schemas.ids import (
    ID_SCHEME_VERSION,
    parallel_id,
    parse_token_id,
    scheme_version_of,
    span_id,
    token_id,
)
from zetodos_thomas.schemas.source import TextUnit, UnitKind

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
FIXTURES = Path(__file__).parent / "fixtures" / "alignment"
JOHN_PHRASE = FIXTURES / "horner-vaticanus-john1.1-phrase.json"
THOMAS_STUB = FIXTURES / "thomas-7-unit-stub.json"


def test_id_helpers_embed_scheme_version() -> None:
    tid = token_id("ntvmr-vaticanus-b", "john.1.1", 0)
    sid = span_id("copsahhorner", "john.1.1", "whole")
    pid = parallel_id("john.1.1-logos")
    assert tid.startswith(f"v{ID_SCHEME_VERSION}:")
    assert scheme_version_of(tid) == ID_SCHEME_VERSION
    assert scheme_version_of(sid) == ID_SCHEME_VERSION
    assert scheme_version_of(pid) == ID_SCHEME_VERSION
    parsed = parse_token_id(tid)
    assert parsed.sequence == 0
    assert parsed.source_id == "ntvmr-vaticanus-b"
    with pytest.raises(ValueError):
        span_id("copsahhorner", "john.1.1", "has space")


def test_pilot_verse_pairing_still_loads() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    assert corpus.greek_source_id == "ntvmr-vaticanus-b"
    assert len(corpus.units) == 37
    john = next(u for u in corpus.units if u.unit_id == "john.1.1")
    assert john.greek.kind is UnitKind.VERSE
    assert "λογος" in john.greek.original_text


def test_whitespace_tokens_from_pilot_john_1_1() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    pair = next(u for u in corpus.units if u.unit_id == "john.1.1")
    greek = whitespace_tokens(pair.greek)
    sahidic = whitespace_tokens(pair.sahidic)
    assert greek[0].surface == "εν"
    assert greek[0].id == token_id("ntvmr-vaticanus-b", "john.1.1", 0)
    assert greek[0].char_start == 0
    assert any(t.normalized == "λογος" for t in greek)
    assert any("ϣⲁϫⲉ" in t.normalized for t in sahidic)
    empty = next(u for u in corpus.units if u.unit_id == "john.8.1")
    assert whitespace_tokens(empty.greek) == []
    assert whitespace_tokens(empty.sahidic) == []


def test_whitespace_tokens_rejects_dilley_style_mismatch() -> None:
    """SCRIPTORIUM-like: continuous orig concat vs segmented norm must not zip."""
    unit = TextUnit(
        source_id="dilley-thomas",
        unit_id="thomas.0",
        book="thomas",
        chapter=0,
        verse=0,
        original_text="ⲛⲁⲉⲓⲛⲉⲛϣⲁϫⲉ",  # continuous concat (no spaces)
        normalized_text="ⲛⲁⲓ ⲛⲉ ⲛ ϣⲁϫⲉ",  # morpheme-segmented
        kind=UnitKind.PROLOGUE,
        sequence=0,
    )
    orig_n, norm_n = whitespace_token_counts(unit)
    assert orig_n == 1
    assert norm_n == 4
    with pytest.raises(TokenCountMismatchError, match="mismatch"):
        whitespace_tokens(unit)


def test_thomas_prologue_unit_allowed() -> None:
    unit = TextUnit(
        source_id="dilley-thomas",
        unit_id="thomas.0",
        book="thomas",
        chapter=0,
        verse=0,
        original_text="",
        normalized_text="",
        kind=UnitKind.PROLOGUE,
        sequence=0,
    )
    corpus = ThomasCorpus(
        id="thomas-prologue-smoke",
        title="prologue smoke",
        source_id="dilley-thomas",
        units=[unit],
    )
    assert corpus.units[0].chapter == 0
    assert corpus.units[0].kind is UnitKind.PROLOGUE


def test_horner_vaticanus_phrase_fixture() -> None:
    doc = AlignmentDocument.model_validate_json(JOHN_PHRASE.read_text(encoding="utf-8"))
    assert doc.id_scheme_version == ID_SCHEME_VERSION
    assert doc.tokens
    logos = next(s for s in doc.spans if s.id.endswith("/s/logos-all"))
    assert logos.kind is SpanKind.DISCONTINUOUS
    assert len(logos.token_ids) >= 2
    phrase = next(p for p in doc.parallels if p.relation is ParallelRelation.PHRASE)
    assert len(phrase.left_span_ids) == 1
    assert len(phrase.right_span_ids) == 1
    empty = next(
        p for p in doc.parallels if p.agreement is AgreementDegree.SHARED_ABSENCE
    )
    assert empty.left_span_ids and empty.right_span_ids


def test_thomas_unit_stub_fixture() -> None:
    doc = AlignmentDocument.model_validate_json(THOMAS_STUB.read_text(encoding="utf-8"))
    assert {u.kind for u in doc.text_units} == {UnitKind.LOGION}
    assert any(u.unit_id == "thomas.7" for u in doc.text_units)
    assert any(p.relation is ParallelRelation.LOANWORD for p in doc.parallels)
    rights = [sid for p in doc.parallels for sid in p.right_span_ids]
    assert len(rights) >= 2
    assert len(set(rights)) < len(rights)


def test_rejects_dangling_token_in_span() -> None:
    unit = TextUnit(
        source_id="ntvmr-vaticanus-b",
        unit_id="john.1.1",
        book="john",
        chapter=1,
        verse=1,
        original_text="εν αρχη",
        normalized_text="εν αρχη",
    )
    tokens = whitespace_tokens(unit)
    span = Span(
        id=span_id(unit.source_id, unit.unit_id, "bad"),
        source_id=unit.source_id,
        unit_id=unit.unit_id,
        kind=SpanKind.CONTIGUOUS,
        token_ids=[tokens[0].id, "v1:ntvmr-vaticanus-b/john.1.1/t/99"],
    )
    with pytest.raises(ValidationError, match="dangling token"):
        AlignmentDocument(
            id="bad",
            title="bad",
            text_units=[unit],
            tokens=tokens,
            spans=[span],
        )


def test_rejects_mismatched_token_id_scheme_or_sequence() -> None:
    unit = TextUnit(
        source_id="ntvmr-vaticanus-b",
        unit_id="john.1.1",
        book="john",
        chapter=1,
        verse=1,
        original_text="εν",
        normalized_text="εν",
    )
    with pytest.raises(ValidationError, match="scheme"):
        AlignmentDocument(
            id="v0",
            title="v0",
            text_units=[unit],
            tokens=[
                Token(
                    id="v0:ntvmr-vaticanus-b/john.1.1/t/0",
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    sequence=0,
                    surface="εν",
                    normalized="εν",
                )
            ],
        )
    with pytest.raises(ValidationError, match="sequence"):
        AlignmentDocument(
            id="seq",
            title="seq",
            text_units=[unit],
            tokens=[
                Token(
                    id=token_id(unit.source_id, unit.unit_id, 0),
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    sequence=1,  # disagrees with id …/t/0
                    surface="εν",
                    normalized="εν",
                )
            ],
        )


def test_rejects_out_of_order_span_token_ids() -> None:
    unit = TextUnit(
        source_id="ntvmr-vaticanus-b",
        unit_id="john.1.1",
        book="john",
        chapter=1,
        verse=1,
        original_text="α β γ",
        normalized_text="α β γ",
    )
    tokens = whitespace_tokens(unit)
    # Contiguous membership but wrong stored order: [t/2, t/0, t/1]
    with pytest.raises(ValidationError, match="out of order"):
        AlignmentDocument(
            id="order",
            title="order",
            text_units=[unit],
            tokens=tokens,
            spans=[
                Span(
                    id=span_id(unit.source_id, unit.unit_id, "scrambled"),
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    kind=SpanKind.CONTIGUOUS,
                    token_ids=[tokens[2].id, tokens[0].id, tokens[1].id],
                )
            ],
        )
    # Reversed discontinuous span
    with pytest.raises(ValidationError, match="out of order"):
        AlignmentDocument(
            id="rev",
            title="rev",
            text_units=[unit],
            tokens=tokens,
            spans=[
                Span(
                    id=span_id(unit.source_id, unit.unit_id, "reversed"),
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    kind=SpanKind.DISCONTINUOUS,
                    token_ids=[tokens[2].id, tokens[0].id],
                )
            ],
        )


def test_rejects_illegal_exclusive_overlap() -> None:
    unit = TextUnit(
        source_id="copsahhorner",
        unit_id="john.1.1",
        book="john",
        chapter=1,
        verse=1,
        original_text="ⲁ ⲃ ⲅ",
        normalized_text="ⲁ ⲃ ⲅ",
    )
    tokens = whitespace_tokens(unit)
    spans = [
        Span(
            id=span_id(unit.source_id, unit.unit_id, "a"),
            source_id=unit.source_id,
            unit_id=unit.unit_id,
            kind=SpanKind.CONTIGUOUS,
            token_ids=[tokens[0].id, tokens[1].id],
            exclusive_group="cover",
        ),
        Span(
            id=span_id(unit.source_id, unit.unit_id, "b"),
            source_id=unit.source_id,
            unit_id=unit.unit_id,
            kind=SpanKind.CONTIGUOUS,
            token_ids=[tokens[1].id, tokens[2].id],
            exclusive_group="cover",
        ),
    ]
    with pytest.raises(ValidationError, match="illegal overlap"):
        AlignmentDocument(
            id="overlap",
            title="overlap",
            text_units=[unit],
            tokens=tokens,
            spans=spans,
        )


def test_rejects_contiguous_with_gap() -> None:
    unit = TextUnit(
        source_id="ntvmr-vaticanus-b",
        unit_id="john.1.1",
        book="john",
        chapter=1,
        verse=1,
        original_text="α β γ",
        normalized_text="α β γ",
    )
    tokens = whitespace_tokens(unit)
    span = Span(
        id=span_id(unit.source_id, unit.unit_id, "gap"),
        source_id=unit.source_id,
        unit_id=unit.unit_id,
        kind=SpanKind.CONTIGUOUS,
        token_ids=[tokens[0].id, tokens[2].id],
    )
    with pytest.raises(ValidationError, match="non-contiguous"):
        AlignmentDocument(
            id="gap",
            title="gap",
            text_units=[unit],
            tokens=tokens,
            spans=[span],
        )


def test_allows_discontinuous_and_empty() -> None:
    unit = TextUnit(
        source_id="ntvmr-vaticanus-b",
        unit_id="john.1.1",
        book="john",
        chapter=1,
        verse=1,
        original_text="α β γ",
        normalized_text="α β γ",
    )
    empty_unit = TextUnit(
        source_id="ntvmr-vaticanus-b",
        unit_id="john.8.1",
        book="john",
        chapter=8,
        verse=1,
        original_text="",
        normalized_text="",
    )
    tokens = whitespace_tokens(unit)
    doc = AlignmentDocument(
        id="ok",
        title="ok",
        text_units=[unit, empty_unit],
        tokens=tokens,
        spans=[
            Span(
                id=span_id(unit.source_id, unit.unit_id, "disc"),
                source_id=unit.source_id,
                unit_id=unit.unit_id,
                kind=SpanKind.DISCONTINUOUS,
                token_ids=[tokens[0].id, tokens[2].id],
            ),
            Span(
                id=span_id(empty_unit.source_id, empty_unit.unit_id, "empty"),
                source_id=empty_unit.source_id,
                unit_id=empty_unit.unit_id,
                kind=SpanKind.EMPTY,
                token_ids=[],
            ),
        ],
        parallels=[
            Parallel(
                id=parallel_id("multi"),
                left_span_ids=[
                    span_id(unit.source_id, unit.unit_id, "disc"),
                    span_id(empty_unit.source_id, empty_unit.unit_id, "empty"),
                ],
                right_span_ids=[span_id(unit.source_id, unit.unit_id, "disc")],
                relation=ParallelRelation.PHRASE,
                agreement=AgreementDegree.UNCERTAIN,
                provenance="unit test",
            )
        ],
    )
    assert len(doc.parallels[0].left_span_ids) == 2


def test_rejects_wrong_scheme_version() -> None:
    unit = TextUnit(
        source_id="x",
        unit_id="u",
        book="thomas",
        chapter=1,
        verse=0,
        original_text="",
        normalized_text="",
        kind=UnitKind.LOGION,
    )
    with pytest.raises(ValidationError, match="id_scheme_version"):
        AlignmentDocument(
            id="old",
            title="old",
            id_scheme_version="0",
            text_units=[unit],
        )


def test_rejects_dangling_parallel_span() -> None:
    unit = TextUnit(
        source_id="x",
        unit_id="u",
        book="john",
        chapter=1,
        verse=1,
        original_text="α",
        normalized_text="α",
    )
    tokens = whitespace_tokens(unit)
    span = Span(
        id=span_id("x", "u", "a"),
        source_id="x",
        unit_id="u",
        kind=SpanKind.CONTIGUOUS,
        token_ids=[tokens[0].id],
    )
    with pytest.raises(ValidationError, match="dangling left span"):
        AlignmentDocument(
            id="p",
            title="p",
            text_units=[unit],
            tokens=tokens,
            spans=[span],
            parallels=[
                Parallel(
                    id=parallel_id("missing"),
                    left_span_ids=["v1:x/u/s/missing"],
                    right_span_ids=[span.id],
                    relation=ParallelRelation.LEXICAL,
                )
            ],
        )


def test_token_char_span_roundtrip_json() -> None:
    token = Token(
        id=token_id("ntvmr-vaticanus-b", "john.1.1", 0),
        source_id="ntvmr-vaticanus-b",
        unit_id="john.1.1",
        sequence=0,
        surface="εν",
        normalized="εν",
        char_start=0,
        char_end=2,
    )
    data = json.loads(token.model_dump_json())
    assert Token.model_validate(data).char_end == 2

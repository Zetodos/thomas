# SPDX-License-Identifier: Apache-2.0
"""Tests for clause chunking, morphosyntactic transfer, and offline beam decoding."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from zetodos_thomas.align.lemma_aligner import LexicalCandidate, LexicalTranslationModel
from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon, LoanwordEntry
from zetodos_thomas.lexicon.idioms import CopticIdiomRule, IdiomAndValencyTable
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.reconstruct.chunker import chunk_unit
from zetodos_thomas.reconstruct.decoder import DecoderWeights, KoineBeamDecoder
from zetodos_thomas.reconstruct.koine_lm import KoineNgramLM, morph_lm_key
from zetodos_thomas.schemas.linguistic import (
    CopticMorpheme,
    CopticUnit,
    GreekMorphFeatures,
)
from zetodos_thomas.schemas.source import UnitKind

MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "mappings" / "model-v1"


def _synthetic_decoder() -> KoineBeamDecoder:
    lex = LexicalTranslationModel(
        by_lemma_pos={
            ("ϣⲁϫⲉ", "N"): [
                LexicalCandidate("λόγος", "N-", 1.5, 0.9, 0.9, "N", 50, 10)
            ],
            ("ⲙⲟⲩ", "N"): [
                LexicalCandidate("θάνατος", "N-", 1.5, 0.9, 0.9, "N", 40, 10)
            ],
            ("ϣⲓⲛⲉ", "V"): [
                LexicalCandidate("ζητέω", "V-", 1.4, 0.85, 0.85, "N", 40, 10)
            ],
            ("ⲛⲟⲩⲧⲉ", "N"): [
                LexicalCandidate("θεός", "N-", 1.6, 0.95, 0.95, "N", 90, 50)
            ],
            ("ⲛⲁⲩ", "V"): [LexicalCandidate("βλέπω", "V-", 1.6, 0.9, 0.9, "N", 60, 20)],
            ("ⲣⲣⲣⲟ", "V"): [
                LexicalCandidate("βασιλεύω", "V-", 1.6, 0.9, 0.9, "N", 20, 10)
            ],
        },
        by_lemma={},
    )
    ccl = CclLexicon(
        loanwords={
            "ϩⲉⲣⲙⲏⲛⲉⲓⲁ": [LoanwordEntry("ϩⲉⲣⲙⲏⲛⲉⲓⲁ", "ἑρμηνεία", "N-", gender="F")]
        },
        itacism_index={},
        greek_genders={"ἑρμηνεία": "F", "λόγος": "M", "θάνατος": "M", "θεός": "M"},
    )
    inf = GreekInflector(
        paradigms={
            (
                "λέγω",
                GreekMorphFeatures(
                    pos="V-", person="3", tense="P", voice="A", mood="I", number="S"
                ).signature(),
            ): "λέγει",
            (
                "λέγω",
                GreekMorphFeatures(
                    pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
                ).signature(),
            ): "εἶπεν",
            (
                "γεύομαι",
                GreekMorphFeatures(
                    pos="V-", person="3", tense="A", voice="M", mood="S", number="S"
                ).signature(),
            ): "γεύσηται",
            (
                "εὑρίσκω",
                GreekMorphFeatures(
                    pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
                ).signature(),
            ): "εὗρεν",
            (
                "ζητέω",
                GreekMorphFeatures(
                    pos="V-",
                    tense="P",
                    voice="A",
                    mood="P",
                    case="N",
                    number="S",
                    gender="M",
                ).signature(),
            ): "ζητῶν",
            (
                "ὁ",
                GreekMorphFeatures(
                    pos="RA", case="N", number="S", gender="M"
                ).signature(),
            ): "ὁ",
            (
                "ὁ",
                GreekMorphFeatures(
                    pos="RA", case="A", number="S", gender="F"
                ).signature(),
            ): "τήν",
            (
                "ὁ",
                GreekMorphFeatures(
                    pos="RA", case="G", number="S", gender="M"
                ).signature(),
            ): "τοῦ",
            (
                "αὐτός",
                GreekMorphFeatures(
                    pos="RP", person="3", case="D", number="P", gender="M"
                ).signature(),
            ): "αὐτοῖς",
            (
                "αὐτός",
                GreekMorphFeatures(
                    pos="RP", case="A", number="S", gender="M"
                ).signature(),
            ): "αὐτόν",
            (
                "σύ",
                GreekMorphFeatures(pos="RP", case="G", number="P").signature(),
            ): "ὑμῶν",
            (
                "βλέπω",
                GreekMorphFeatures(
                    pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
                ).signature(),
            ): "ἔβλεψεν",
        },
        coarse_paradigms={},
        lemma_genders={"λόγος": "M", "θάνατος": "M", "ἑρμηνεία": "F", "θεός": "M"},
        surface_analyses={},
    )
    lm = KoineNgramLM(
        unigram_logp={
            morph_lm_key(
                "γεύομαι",
                GreekMorphFeatures(
                    pos="V-", person="3", tense="A", voice="M", mood="S", number="S"
                ),
            ): -2.0
        },
        bigram_logp={
            (
                "<s>",
                morph_lm_key(
                    "λέγω",
                    GreekMorphFeatures(
                        pos="V-", person="3", tense="P", voice="A", mood="I", number="S"
                    ),
                ),
            ): -1.2
        },
        trigram_logp={},
        surface_bi_logp={},
    )
    synthetic_idioms = IdiomAndValencyTable(
        idioms=(
            CopticIdiomRule(("ϩⲉ", "ⲉ"), "εὑρίσκω", "V-", required_case="A"),
            CopticIdiomRule(("ϫⲓ", "ϯⲡⲉ"), "γεύομαι", "V-", required_case="G"),
            CopticIdiomRule(("ϭⲱⲗⲡ", "ⲉⲃⲟⲗ"), "ἀποκαλύπτω", "V-", required_case="A"),
            CopticIdiomRule(("ⲃⲱⲕ", "ⲉϩⲟⲩⲛ"), "εἰσέρχομαι", "V-"),
        ),
        verb_cases={"γεύομαι": "G", "εὑρίσκω": "A"},
        prep_cases={"ἐν": "D", "εἰς": "A"},
    )
    return KoineBeamDecoder(
        lexicon=lex,
        ccl_lexicon=ccl,
        idioms=synthetic_idioms,
        inflector=inf,
        lm=lm,
        weights=DecoderWeights(),
    )


def test_transfer_rules_negative_subjunctive_genitive_and_gender_reprojection() -> None:
    dec = _synthetic_decoder()
    # Coptic: "ⲡⲉϫⲁ=ϥ ⲛⲁ=ⲩ ϫⲉ ⲛⲛⲉ=ϥ ϫⲓϯⲡⲉ ⲙ=ⲡ=ⲙⲟⲩ" + "ϩⲉ ⲉ=ⲧ=ϩⲉⲣⲙⲏⲛⲉⲓⲁ"
    unit = CopticUnit(
        unit_id="thomas.test.1",
        source_id="dilley-thomas",
        book="thomas",
        chapter=1,
        verse=1,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲡⲉϫⲁ", lemma="ⲡⲉϫⲉ", pos="VBD"),
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERO"),
            CopticMorpheme(group_seq=1, norm="ⲛⲁ", lemma="ⲛⲁ", pos="PREP"),
            CopticMorpheme(group_seq=1, norm="ⲩ", lemma="ⲛⲧⲟⲟⲩ", pos="PPERO"),
            CopticMorpheme(group_seq=2, norm="ⲛⲛⲉ", lemma="ⲛⲛⲉ", pos="ANEGJUS"),
            CopticMorpheme(group_seq=2, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=2, norm="ϫⲓ", lemma="ϫⲓ", pos="V"),
            CopticMorpheme(group_seq=2, norm="ϯⲡⲉ", lemma="ϯⲡⲉ", pos="N"),
            CopticMorpheme(group_seq=3, norm="ⲙ", lemma="ⲛ", pos="PREP"),
            CopticMorpheme(group_seq=3, norm="ⲡ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(group_seq=3, norm="ⲙⲟⲩ", lemma="ⲙⲟⲩ", pos="N"),
        ],
    )
    gen = dec.decode_unit(unit)
    lemmas = [t.lemma for t in gen.tokens]
    assert "λέγω" in lemmas
    assert "οὐ" in lemmas and "μή" in lemmas
    assert "γεύομαι" in lemmas
    assert "θάνατος" in lemmas
    assert "θανάτου" in gen.greek_text  # Genitive governed by γεύομαι!
    assert "null" not in gen.greek_text.lower()

    # Check articular participle (`ⲡ+ⲉⲧ+ϣⲓⲛⲉ`) + feminine loanword (`ⲧ+ϩⲉⲣⲙⲏⲛⲉⲓⲁ`)
    u2 = CopticUnit(
        unit_id="thomas.test.2",
        source_id="dilley-thomas",
        book="thomas",
        chapter=2,
        verse=2,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲡ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(group_seq=0, norm="ⲉⲧ", lemma="ⲉⲧⲉⲣⲉ", pos="CREL"),
            CopticMorpheme(group_seq=0, norm="ϣⲓⲛⲉ", lemma="ϣⲓⲛⲉ", pos="V"),
            CopticMorpheme(group_seq=1, norm="ⲁ", lemma="ⲁ", pos="APST"),
            CopticMorpheme(group_seq=1, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=1, norm="ϩⲉ", lemma="ϩⲉ", pos="V"),
            CopticMorpheme(group_seq=2, norm="ⲉ", lemma="ⲉ", pos="PREP"),
            CopticMorpheme(group_seq=2, norm="ⲧ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(
                group_seq=2, norm="ϩⲉⲣⲙⲏⲛⲉⲓⲁ", lemma="ϩⲉⲣⲙⲏⲛⲉⲓⲁ", pos="N", lang="Greek"
            ),
        ],
    )
    gen2 = dec.decode_unit(u2)
    assert "ζητῶν" in gen2.greek_text
    assert "εὗρ" in gen2.greek_text
    assert "τὴν ἑρμηνείαν" in gen2.greek_text


def test_chunk_unit_and_offline_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    dec = _synthetic_decoder()
    unit = CopticUnit(
        unit_id="thomas.test.det",
        source_id="dilley-thomas",
        book="thomas",
        chapter=1,
        verse=1,
        kind=UnitKind.LOGION,
        original_text="ⲡⲉϫⲁϥ",
        normalized_text="ⲡⲉϫⲁ ϥ",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲡⲉϫⲁ", lemma="ⲡⲉϫⲉ", pos="VBD"),
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERO"),
            CopticMorpheme(group_seq=1, norm="ⲇⲉ", lemma="ⲇⲉ", pos="PTC"),
            CopticMorpheme(group_seq=2, norm="ϫⲉ", lemma="ϫⲉ", pos="CONJ"),
            CopticMorpheme(group_seq=3, norm="ⲡ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(group_seq=3, norm="ϣⲁϫⲉ", lemma="ϣⲁϫⲉ", pos="N"),
        ],
    )
    clauses = chunk_unit(unit)
    assert clauses and clauses[0].postpositive is not None

    # Block network sockets strictly during decoding
    def _no_net(*_a: object, **_kw: object) -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(socket, "socket", _no_net)
    r1 = dec.decode_unit(unit).model_dump_json()
    r2 = dec.decode_unit(unit).model_dump_json()
    assert r1 == r2
    assert "δέ" in r1
    assert dec.decode_unit(unit).greek_text.split()[1] == "δὲ"


def test_possessed_locative_and_perception_oblique_e() -> None:
    dec = _synthetic_decoder()
    unit = CopticUnit(
        unit_id="thomas.test.s3",
        source_id="dilley-thomas",
        book="thomas",
        chapter=3,
        verse=3,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲁ", lemma="ⲁ", pos="APST"),
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=0, norm="ⲛⲁⲩ", lemma="ⲛⲁⲩ", pos="V"),
            CopticMorpheme(group_seq=1, norm="ⲉⲣⲟ", lemma="ⲉ", pos="PREP"),
            CopticMorpheme(group_seq=1, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERO"),
            CopticMorpheme(group_seq=2, norm="ⲙ", lemma="ⲛ", pos="PREP"),
            CopticMorpheme(group_seq=2, norm="ⲡⲉⲧⲛ", lemma="ⲡⲉⲧⲛ", pos="PPOS"),
            CopticMorpheme(group_seq=2, norm="ϩⲟⲩⲛ", lemma="ϩⲟⲩⲛ", pos="N"),
        ],
    )
    gen = dec.decode_unit(unit)
    assert "εἰς" not in gen.greek_text
    assert "αὐτὸν" in gen.greek_text
    assert "ἐντὸς ὑμῶν" in gen.greek_text


def test_tree_guided_postverbal_nmod_genitive() -> None:
    dec = _synthetic_decoder()
    dec.inflector._paradigms[
        (
            "θεός",
            GreekMorphFeatures(pos="N-", case="G", number="S", gender="M").signature(),
        )
    ] = "θεοῦ"
    dec.inflector._paradigms[
        (
            "λόγος",
            GreekMorphFeatures(pos="N-", case="A", number="S", gender="M").signature(),
        )
    ] = "λόγον"
    dec.inflector._paradigms[
        (
            "ὁ",
            GreekMorphFeatures(pos="RA", case="A", number="S", gender="M").signature(),
        )
    ] = "τόν"
    # "ⲁ ϥ ⲛⲁⲩ ⲙ ⲡ ϣⲁϫⲉ ⲙ ⲡ ⲛⲟⲩⲧⲉ" (saw the word [obj] of God [nmod]!)
    unit = CopticUnit(
        unit_id="thomas.test.s5",
        source_id="dilley-thomas",
        book="thomas",
        chapter=5,
        verse=5,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(
                group_seq=0, norm="ⲁ", lemma="ⲁ", pos="APST", head_seq=2, dep_rel="aux"
            ),
            CopticMorpheme(
                group_seq=0,
                norm="ϥ",
                lemma="ⲛⲧⲟϥ",
                pos="PPERS",
                head_seq=2,
                dep_rel="nsubj",
            ),
            CopticMorpheme(
                group_seq=0,
                norm="ⲛⲁⲩ",
                lemma="ⲛⲁⲩ",
                pos="V",
                head_seq=None,
                dep_rel="root",
            ),
            CopticMorpheme(
                group_seq=1, norm="ⲙ", lemma="ⲛ", pos="PREP", head_seq=5, dep_rel="case"
            ),
            CopticMorpheme(
                group_seq=1, norm="ⲡ", lemma="ⲡ", pos="ART", head_seq=5, dep_rel="det"
            ),
            CopticMorpheme(
                group_seq=1,
                norm="ϣⲁϫⲉ",
                lemma="ϣⲁϫⲉ",
                pos="N",
                head_seq=2,
                dep_rel="obj",
            ),
            CopticMorpheme(
                group_seq=2, norm="ⲙ", lemma="ⲛ", pos="PREP", head_seq=8, dep_rel="case"
            ),
            CopticMorpheme(
                group_seq=2, norm="ⲡ", lemma="ⲡ", pos="ART", head_seq=8, dep_rel="det"
            ),
            CopticMorpheme(
                group_seq=2,
                norm="ⲛⲟⲩⲧⲉ",
                lemma="ⲛⲟⲩⲧⲉ",
                pos="N",
                head_seq=5,
                dep_rel="nmod",
            ),
        ],
    )
    gen = dec.decode_unit(unit)
    assert "τὸν λόγον τοῦ θεοῦ" in gen.greek_text


def test_stage6_stroke_norm_pdem_and_numerals() -> None:
    dec = _synthetic_decoder()
    dec.lexicon._by_lemma_pos[("ϩⲟⲟⲩ", "N")] = [
        LexicalCandidate("ἡμέρα", "N-", 1.6, 0.9, 0.9, "N", 80, 40)
    ]
    dec.inflector._paradigms[
        (
            "ὁ",
            GreekMorphFeatures(pos="RA", case="G", number="P", gender="M").signature(),
        )
    ] = "τῶν"
    dec.inflector._paradigms[
        (
            "λόγος",
            GreekMorphFeatures(pos="N-", case="G", number="P", gender="M").signature(),
        )
    ] = "λόγων"
    dec.inflector._paradigms[
        (
            "οὗτος",
            GreekMorphFeatures(pos="RD", case="G", number="P", gender="M").signature(),
        )
    ] = "τούτων"
    unit = CopticUnit(
        unit_id="thomas.test.s6",
        source_id="dilley-thomas",
        book="thomas",
        chapter=6,
        verse=6,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲁ", lemma="ⲁ", pos="APST"),
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=0, norm="ⲣⲣⲣⲟ", lemma="ⲣ̄ⲣⲣⲟ", pos="V"),
            CopticMorpheme(group_seq=1, norm="ⲛ", lemma="ⲛ", pos="PREP"),
            CopticMorpheme(group_seq=1, norm="ⲛⲉⲓ", lemma="ⲡⲉⲓ", pos="PDEM"),
            CopticMorpheme(
                group_seq=1, norm="ϣⲁϫⲉ", lemma="ϣⲁϫⲉ", pos="N", dep_rel="nmod"
            ),
            CopticMorpheme(group_seq=2, norm="ϩⲛ", lemma="ϩⲛ", pos="PREP"),
            CopticMorpheme(group_seq=2, norm="ⲥⲁϣϥ̅", lemma="ⲥⲁϣϥ̅", pos="NUM"),
            CopticMorpheme(group_seq=3, norm="ⲛ", lemma="ⲛ", pos="PREP"),
            CopticMorpheme(group_seq=3, norm="ϩⲟⲟⲩ", lemma="ϩⲟⲟⲩ", pos="N"),
        ],
    )
    gen = dec.decode_unit(unit)
    lemmas = [t.lemma for t in gen.tokens]
    assert "βασιλεύω" in lemmas
    assert "τῶν λόγων τούτων" in gen.greek_text
    assert "ἐν ἑπτὰ ἡμέραις" in gen.greek_text


def test_stage7_inf_idioms_and_labile_passives() -> None:
    dec = _synthetic_decoder()
    dec.lexicon._by_lemma_pos[("ϭⲱⲗⲡ", "V")] = [
        LexicalCandidate("ἀποκαλύπτω", "V-", 1.5, 0.9, 0.9, "N", 80, 40)
    ]
    unit = CopticUnit(
        unit_id="thomas.test.s7",
        source_id="dilley-thomas",
        book="thomas",
        chapter=7,
        verse=7,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=0, norm="ⲛⲁ", lemma="ⲛⲁ", pos="FUT"),
            CopticMorpheme(group_seq=0, norm="ϭⲱⲗⲡ", lemma="ϭⲱⲗⲡ", pos="V"),
            CopticMorpheme(group_seq=0, norm="ⲉⲃⲟⲗ", lemma="ⲉⲃⲟⲗ", pos="ADV"),
            CopticMorpheme(group_seq=1, norm="·", lemma="·", pos="PUNCT"),
            CopticMorpheme(group_seq=2, norm="ⲉ", lemma="ⲉ", pos="PREP"),
            CopticMorpheme(group_seq=2, norm="ⲃⲱⲕ", lemma="ⲃⲱⲕ", pos="V"),
            CopticMorpheme(group_seq=2, norm="ⲉϩⲟⲩⲛ", lemma="ⲉϩⲟⲩⲛ", pos="ADV"),
        ],
    )
    gen = dec.decode_unit(unit)
    assert "ἀποκαλυφθήσεται" in gen.greek_text.lower()
    assert "εἰσελθεῖν" in gen.greek_text.lower()
    assert not any(t.lemma == "εἰς" for t in gen.tokens)


def test_stage8_clauses_imperatives_modal_and_existential() -> None:
    dec = _synthetic_decoder()
    dec.lexicon._by_lemma_pos[("ϩⲱⲡ", "V")] = [
        LexicalCandidate("κρύπτω", "V-", 1.6, 0.9, 0.9, "N", 80, 40)
    ]
    dec.lexicon._by_lemma_pos[("ϩⲉ", "V")] = [
        LexicalCandidate("πίπτω", "V-", 1.6, 0.9, 0.9, "N", 80, 40)
    ]
    dec.lexicon._by_lemma_pos[("ⲡⲣⲟⲫⲏⲧⲏⲥ", "N")] = [
        LexicalCandidate("προφήτης", "N-", 1.8, 0.95, 0.95, "N", 80, 40)
    ]
    dec.lexicon._by_lemma_pos[("ϣⲱⲡⲉ", "V")] = [
        LexicalCandidate("γίνομαι", "V-", 1.6, 0.9, 0.9, "N", 80, 40)
    ]
    dec.lexicon._by_lemma_pos[("ⲉⲓⲣⲏⲛⲏ", "N")] = [
        LexicalCandidate("εἰρήνη", "N-", 1.8, 0.95, 0.95, "N", 80, 40)
    ]
    dec.lexicon._by_lemma_pos[("ϣⲏⲡ", "V")] = [
        LexicalCandidate("δεκτός", "A-", 1.8, 0.95, 0.95, "N", 10, 5)
    ]
    dec.ccl_lexicon._greek_genders.update({"προφήτης": "M", "εἰρήνη": "F"})
    dec.inflector._paradigms[
        (
            "εἰμί",
            GreekMorphFeatures(
                pos="V-", person="1", tense="P", voice="A", mood="I", number="S"
            ).signature(),
        )
    ] = "εἰμί"
    dec.inflector._paradigms[
        (
            "εἰμί",
            GreekMorphFeatures(
                pos="V-", person="3", tense="P", voice="A", mood="I", number="S"
            ).signature(),
        )
    ] = "ἐστίν"
    dec.inflector._paradigms[
        (
            "δύναμαι",
            GreekMorphFeatures(
                pos="V-", person="3", tense="P", voice="M", mood="I", number="S"
            ).signature(),
        )
    ] = "δύναται"
    dec.inflector._paradigms[
        (
            "πίπτω",
            GreekMorphFeatures(pos="V-", tense="A", voice="A", mood="N").signature(),
        )
    ] = "πεσεῖν"
    dec.inflector._paradigms[
        (
            "κρύπτω",
            GreekMorphFeatures(
                pos="V-",
                tense="X",
                voice="P",
                mood="P",
                case="N",
                number="S",
                gender="N",
            ).signature(),
        )
    ] = "κεκρυμμένον"
    dec.inflector._paradigms[
        (
            "σύ",
            GreekMorphFeatures(pos="RP", case="N", number="P").signature(),
        )
    ] = "ὑμεῖς"
    dec.inflector._paradigms[
        (
            "ἐγώ",
            GreekMorphFeatures(pos="RP", case="N", number="S").signature(),
        )
    ] = "ἐγώ"
    dec.inflector._paradigms[
        (
            "ὁ",
            GreekMorphFeatures(pos="RA", case="N", number="S", gender="N").signature(),
        )
    ] = "τό"

    unit = CopticUnit(
        unit_id="thomas.test.s8",
        source_id="dilley-thomas",
        book="thomas",
        chapter=8,
        verse=8,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲁ", lemma="ⲁ", pos="APST"),
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=0, norm="ϫⲟⲟ", lemma="ϫⲱ", pos="V"),
            CopticMorpheme(group_seq=0, norm="ⲥ", lemma="ⲛⲧⲟⲥ", pos="PPERO"),
            CopticMorpheme(group_seq=1, norm="ϫⲉ", lemma="ϫⲉ", pos="CONJ"),
            CopticMorpheme(group_seq=2, norm="ⲁⲛⲟⲕ", lemma="ⲁⲛⲟⲕ", pos="PPERI"),
            CopticMorpheme(group_seq=3, norm="ϯ", lemma="ⲁⲛⲟⲕ", pos="PPERS"),
            CopticMorpheme(group_seq=4, norm="ϩⲛ", lemma="ϩⲛ", pos="PREP"),
            CopticMorpheme(group_seq=4, norm="ⲉⲓⲣⲏⲛⲏ", lemma="ⲉⲓⲣⲏⲛⲏ", pos="N"),
            CopticMorpheme(group_seq=5, norm="·", lemma="·", pos="PUNCT"),
            CopticMorpheme(group_seq=6, norm="ⲛⲧⲱⲧⲛ", lemma="ⲛⲧⲱⲧⲛ", pos="PPERI"),
            CopticMorpheme(group_seq=7, norm="ⲇⲉ", lemma="ⲇⲉ", pos="PTC"),
            CopticMorpheme(group_seq=8, norm="ϣⲱⲡⲉ", lemma="ϣⲱⲡⲉ", pos="V"),
            CopticMorpheme(group_seq=9, norm="·", lemma="·", pos="PUNCT"),
            CopticMorpheme(group_seq=10, norm="ⲡ", lemma="ⲡ", pos="ART"),
            CopticMorpheme(group_seq=10, norm="ⲉⲧ", lemma="ⲉⲧⲉⲣⲉ", pos="CREL"),
            CopticMorpheme(group_seq=10, norm="ϩⲏⲡ", lemma="ϩⲱⲡ", pos="VSTAT"),
            CopticMorpheme(group_seq=11, norm="·", lemma="·", pos="PUNCT"),
            CopticMorpheme(group_seq=12, norm="ⲙⲛ", lemma="ⲙⲛ", pos="EXIST"),
            CopticMorpheme(group_seq=13, norm="ϭⲟⲙ", lemma="ϭⲟⲙ", pos="N"),
            CopticMorpheme(group_seq=14, norm="ⲛ", lemma="ⲛⲧⲉ", pos="ACONJ"),
            CopticMorpheme(group_seq=14, norm="ⲥ", lemma="ⲛⲧⲟⲥ", pos="PPERS"),
            CopticMorpheme(group_seq=14, norm="ϩⲉ", lemma="ϩⲉ", pos="V"),
            CopticMorpheme(group_seq=15, norm="·", lemma="·", pos="PUNCT"),
            CopticMorpheme(group_seq=16, norm="ⲙⲛ", lemma="ⲙⲛ", pos="PREP"),
            CopticMorpheme(group_seq=17, norm="ⲡⲣⲟⲫⲏⲧⲏⲥ", lemma="ⲡⲣⲟⲫⲏⲧⲏⲥ", pos="N"),
            CopticMorpheme(group_seq=18, norm="ϣⲏⲡ", lemma="ϣⲏⲡ", pos="VSTAT"),
        ],
    )
    gen = dec.decode_unit(unit)
    txt = gen.greek_text.lower()
    assert "ἐγὼ εἰμὶ ἐν εἰρήνῃ" in txt
    assert "ὑμεῖς δὲ γίνεσθε" in txt
    assert "τὸ κεκρυμμένον" in txt
    assert "οὐ δύναται πεσεῖν" in txt
    assert "προφήτης δεκτός" in txt
    assert any(
        t.lemma == "εἰμί" and t.morph.tense == "P" and t.morph.person == "1"
        for t in gen.tokens
    )
    assert any(t.lemma == "προφήτης" and t.morph.case == "N" for t in gen.tokens)
    vbd_unit = CopticUnit(
        unit_id="thomas.test.s8.vbd",
        source_id="dilley-thomas",
        book="thomas",
        chapter=8,
        verse=9,
        kind=UnitKind.LOGION,
        original_text="test",
        normalized_text="test",
        morphemes=[
            CopticMorpheme(group_seq=0, norm="ⲡⲉϫⲁ", lemma="ⲡⲉϫⲉ", pos="VBD"),
            CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
            CopticMorpheme(group_seq=1, norm="ⲛⲁ", lemma="ⲛⲁ", pos="PREP"),
            CopticMorpheme(group_seq=1, norm="ⲩ", lemma="ⲛⲧⲟⲟⲩ", pos="PPERO"),
        ],
    )
    vbd_gen = dec.decode_unit(vbd_unit)
    assert not any(t.lemma == "εἰμί" for t in vbd_gen.tokens)

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lacuna-aware P.Oxy. fragment scoring & Koine itacism normalization."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.eval.benchmark import (
    _AttestedGreekLemmatiser,
    koine_itacism_fold,
    score_fragment_span_lacuna_aware,
    strip_greek_diacritics,
)
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.evidence import FragmentClauseSpan
from zetodos_thomas.schemas.fragment import FragmentWord
from zetodos_thomas.schemas.generated import GeneratedToken
from zetodos_thomas.schemas.linguistic import GreekMorphFeatures


def _tok(seq: int, surf: str, lemma: str, pos: str) -> GeneratedToken:
    return GeneratedToken(
        seq=seq,
        surface=surf,
        lemma=lemma,
        morph=GreekMorphFeatures(pos=pos),
        coptic_seqs=[seq],
        provenance="N",
        confidence=1.0,
    )


def test_itacism_and_diacritic_normalization() -> None:
    assert strip_greek_diacritics("Θαλάσσης") == "θαλασσησ"
    assert koine_itacism_fold("νηστεύσηται") == koine_itacism_fold("νηστεύσητε")
    assert koine_itacism_fold("γεινώσκοντας") == koine_itacism_fold("γινώσκοντας")
    assert koine_itacism_fold("εἰλικίαν") == koine_itacism_fold("ἡλικίαν")


def test_lacuna_only_words_excluded_and_truncated_prefix_scored_fairly() -> None:
    lemmatiser = _AttestedGreekLemmatiser(
        exact_map={
            "θαλασσησ": ("θάλασσα", "N-"),
            "ιχθυεσ": ("ἰχθύς", "N-"),
            "προφθασαντεσ": ("προφθάνω", "V-"),
        },
        ita_map={},
    )
    # Word 0 is complete lacuna (`surviving_text == ""`), word 1 truncated `θαλά[σσης]`,
    # word 2 is complete lacuna `[προφθάσαντες]`
    words = [
        FragmentWord(
            word_index=0,
            line_id="main.14",
            surviving_text="",
            restored_text="ἰχθύες",
            full_editorial_word="ἰχθύες",
            is_truncated=False,
        ),
        FragmentWord(
            word_index=1,
            line_id="main.14",
            surviving_text="θαλά",
            restored_text="σσης",
            full_editorial_word="θαλάσσης",
            is_truncated=True,
        ),
        FragmentWord(
            word_index=2,
            line_id="main.14",
            surviving_text="",
            restored_text="προφθάσαντες",
            full_editorial_word="προφθάσαντες",
            is_truncated=False,
        ),
    ]
    assert lemmatiser.resolve_word(words[0]) is None
    assert lemmatiser.resolve_word(words[1]) == ("θάλασσα", "N-")
    assert lemmatiser.resolve_word(words[2]) is None

    span = FragmentClauseSpan(
        span_id="test.span",
        unit_id="thomas.3",
        transcription_id="poxy-654-dclp",
        coptic_start=0,
        coptic_end=3,
        frag_word_start=0,
        frag_word_end=3,
    )
    preds = [
        _tok(0, "ἰχθύες", "ἰχθύς", "N-"),
        _tok(1, "θαλάσσης", "θάλασσα", "N-"),
        _tok(2, "προάξουσιν", "προάγω", "V-"),
    ]
    counts, summary = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=words,
        pred_tokens=preds,
        lemmatiser=lemmatiser,
        loanword_lemmas=[],
    )
    # Only the 1 surviving word (θαλά[σσης]) has gold lemma `θάλασσα`, matching oracle!
    assert counts["cl_gold"] == 1
    assert counts["cl_matched"] == 1
    # Truncated prefix `θαλά` scores 4/4 = 1.000 char F1 against predicted `θαλάσσης`
    assert summary.surviving_char_f1_itacism == 1.0
    assert summary.gold_content_lemmas == ["θάλασσα"]


def test_stage9_attested_lemmatiser_poxy_morphology() -> None:
    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={"ὄψις": "F"},
        surface_analyses={},
        known_lemmas={
            "γινωσκω": ("γινώσκω", "V-"),
            "οικοδομεω": ("οἰκοδομέω", "V-"),
            "στηριζω": ("στηρίζω", "V-"),
            "νηστευω": ("νηστεύω", "V-"),
            "παυω": ("παύω", "V-"),
            "ελκω": ("ἕλκω", "V-"),
            "θαπτω": ("θάπτω", "V-"),
            "οψις": ("ὄψις", "N-"),
            "προστιθημι": ("προστίθημι", "V-"),
        },
    )
    lem = _AttestedGreekLemmatiser(exact_map={}, ita_map={}, inflector=inf)
    cases = (
        ("γεινώσκοντας", ("γινώσκω", "V-")),
        ("οἰκοδομημένη", ("οἰκοδομέω", "V-")),
        ("ἐστηριγμένη", ("στηρίζω", "V-")),
        ("νηστεύσομεν", ("νηστεύω", "V-")),
        ("παυσάσθω", ("παύω", "V-")),
        ("ἕλκοντες", ("ἕλκω", "V-")),
        ("θεθαμμένον", ("θάπτω", "V-")),
        ("ὄψεώς", ("ὄψις", "N-")),
        ("προσθείη", ("προστίθημι", "V-")),
    )
    for surface, expected in cases:
        fw = FragmentWord(
            word_index=0,
            line_id="main.1",
            surviving_text=surface,
            restored_text="",
            full_editorial_word=surface,
            is_truncated=False,
        )
        assert lem.resolve_word(fw) == expected

    morphgnt_dir = (
        Path(__file__).resolve().parents[1] / "data" / "downloads" / "morphgnt-sblgnt"
    )
    if morphgnt_dir.is_dir():
        real_lem = _AttestedGreekLemmatiser.from_morphgnt(morphgnt_dir)
        for surface, expected in cases:
            fw = FragmentWord(
                word_index=0,
                line_id="main.1",
                surviving_text=surface,
                restored_text="",
                full_editorial_word=surface,
                is_truncated=False,
            )
            assert real_lem.resolve_word(fw) == expected

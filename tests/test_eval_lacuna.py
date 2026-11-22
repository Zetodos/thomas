# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lacuna-aware P.Oxy. fragment scoring & Koine itacism normalization."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.eval.benchmark import (
    _aggregate_frag_stratum,
    _AttestedGreekLemmatiser,
    _lcs_len,
    _monotone_surviving_char_overlap,
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


def test_order_sensitive_lcs_surviving_only_and_zero_loanword_none() -> None:
    # 1. Char overlap inside aligned words is order-preserving LCS, not bag-of-chars
    fw_trunc = FragmentWord(
        word_index=0,
        line_id="main.1",
        surviving_text="θαλά",
        restored_text="σσης",
        full_editorial_word="θαλάσσης",
        is_truncated=True,
    )
    m_exact, _, tot_exact, _ = _monotone_surviving_char_overlap(
        [fw_trunc], ["θαλάσσης"], normalizer=strip_greek_diacritics
    )
    assert m_exact == 4 and tot_exact == 4
    m_anagram, _, tot_anagram, _ = _monotone_surviving_char_overlap(
        [fw_trunc], ["αθαλσσης"], normalizer=strip_greek_diacritics
    )
    assert m_anagram == 3 and tot_anagram == 4 and m_anagram < tot_anagram

    # 2. Sequence LCS distinguishes reversed lemma order vs unordered multiset
    assert _lcs_len(["λόγος", "θεός"], ["λόγος", "θεός"]) == 2
    assert _lcs_len(["θεός", "λόγος"], ["λόγος", "θεός"]) == 1

    # 3. surviving_only excludes restored words; zero loanwords yields None (not 1.0)
    lemmatiser = _AttestedGreekLemmatiser(
        exact_map={
            "θαλασσησ": ("θάλασσα", "N-"),
            "λογοσ": ("λόγος", "N-"),
        },
        ita_map={},
    )
    fw_intact = FragmentWord(
        word_index=1,
        line_id="main.1",
        surviving_text="λόγος",
        restored_text="",
        full_editorial_word="λόγος",
        is_truncated=False,
    )
    assert lemmatiser.resolve_word(fw_trunc, surviving_only=False) == ("θάλασσα", "N-")
    assert lemmatiser.resolve_word(fw_trunc, surviving_only=True) is None
    assert lemmatiser.resolve_word(fw_intact, surviving_only=True) == ("λόγος", "N-")

    span = FragmentClauseSpan(
        span_id="test.order",
        unit_id="thomas.3",
        transcription_id="poxy-654-dclp",
        coptic_start=0,
        coptic_end=2,
        frag_word_start=0,
        frag_word_end=2,
    )
    # Reversed order prediction vs [θάλασσα, λόγος]
    preds = [
        _tok(0, "λόγος", "λόγος", "N-"),
        _tok(1, "θαλάσσης", "θάλασσα", "N-"),
    ]
    counts, _ = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=[fw_trunc, fw_intact],
        pred_tokens=preds,
        lemmatiser=lemmatiser,
        loanword_lemmas=[],
    )
    stratum = _aggregate_frag_stratum([counts], [span])
    assert stratum.content_lemma_f1 == 1.0
    assert stratum.ordered_content_lemma_f1 == 0.5
    assert stratum.surviving_only_word_count == 1
    assert stratum.loanword_total == 0
    assert stratum.loanword_exact_match_rate is None

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lacuna-aware P.Oxy. fragment scoring & Koine itacism normalization."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.eval.benchmark import (
    _aggregate_frag_stratum,
    _AttestedGreekLemmatiser,
    _lcs_len,
    _loanword_script_stem_matches,
    _monotone_surviving_char_overlap,
    _prf,
    koine_itacism_fold,
    score_fragment_span_lacuna_aware,
    strip_greek_diacritics,
)
from zetodos_thomas.ingest.fragments import load_fragment_corpus
from zetodos_thomas.ingest.paths import resolve_fragments_corpus_path
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.evidence import FragmentClauseSpan
from zetodos_thomas.schemas.fragment import FragmentWord
from zetodos_thomas.schemas.generated import GeneratedToken
from zetodos_thomas.schemas.linguistic import GreekMorphFeatures


def _tok(
    seq: int,
    surf: str,
    lemma: str,
    pos: str,
    *,
    coptic_seqs: list[int] | None = None,
) -> GeneratedToken:
    return GeneratedToken(
        seq=seq,
        surface=surf,
        lemma=lemma,
        morph=GreekMorphFeatures(pos=pos),
        coptic_seqs=list(coptic_seqs) if coptic_seqs is not None else [seq],
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
        greek_loan_morphemes=(),
    )
    # Secondary editorial-assisted span includes truncated `θαλά[σσης]` -> `θάλασσα`
    assert counts["cl_gold"] == 1
    assert counts["cl_matched"] == 1
    # Primary strictly unrestored gold excludes truncated/restored words completely
    assert counts["so_gold"] == 0
    assert summary.gold_content_lemmas == []
    # Truncated prefix `θαλά` scores 4/4 = 1.000 char F1 against predicted `θαλάσσης`
    assert summary.surviving_char_f1_itacism == 1.0


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
    counts, summary = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=[fw_trunc, fw_intact],
        pred_tokens=preds,
        lemmatiser=lemmatiser,
        greek_loan_morphemes=(),
    )
    stratum = _aggregate_frag_stratum([counts], [span])
    assert stratum.content_lemma_f1 == 1.0
    assert stratum.ordered_content_lemma_f1 == 0.5
    assert stratum.surviving_only_word_count == 1
    assert summary.gold_content_lemmas == ["λόγος"]
    assert stratum.surviving_only_ordered_content_lemma_recall == 1.0
    assert stratum.surviving_only_ordered_content_lemma_precision == 0.5
    assert stratum.loanword_total == 0
    assert stratum.loanword_exact_match_rate is None


def test_restorations_excluded_from_primary_gold_and_char_scoring() -> None:
    """Ported JAK-40 invariant: editorial [...] never enters primary gold."""
    lemmatiser = _AttestedGreekLemmatiser(
        exact_map={"μακαριοσ": ("μακάριος", "A-")},
        ita_map={},
    )
    fw_restored = FragmentWord(
        word_index=0,
        line_id="main.40",
        surviving_text="κάρι",
        restored_text="μαός",
        full_editorial_word="μακάριός",
        is_truncated=True,
    )
    assert lemmatiser.resolve_word(fw_restored, surviving_only=True) is None
    span = FragmentClauseSpan(
        span_id="test.restoration",
        unit_id="thomas.7",
        transcription_id="poxy-654-dclp",
        coptic_start=0,
        coptic_end=1,
        frag_word_start=0,
        frag_word_end=1,
    )
    # Prediction matching ONLY restored letters ("μαός") must not get surviving credit
    preds = [_tok(0, "μαός", "μακάριος", "A-")]
    counts, summary = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=[fw_restored],
        pred_tokens=preds,
        lemmatiser=lemmatiser,
        greek_loan_morphemes=(),
    )
    assert counts["so_gold"] == 0
    assert summary.gold_content_lemmas == []
    assert summary.attested_surviving == "κάρι"
    assert summary.surviving_char_f1_itacism < 0.5


def test_unclear_surviving_tokens_are_a_separate_stratum() -> None:
    """Ported integrity invariant: unclear/dotted letters are stratified separately."""
    lemmatiser = _AttestedGreekLemmatiser(
        exact_map={"λογοσ": ("λόγος", "N-")},
        ita_map={},
    )
    fw_clear = FragmentWord(
        word_index=0,
        line_id="main.33",
        surviving_text="λόγος",
        full_editorial_word="λόγος",
        has_unclear=False,
    )
    fw_unclear = FragmentWord(
        word_index=1,
        line_id="main.33",
        surviving_text="ουσιν",
        full_editorial_word="ουσιν",
        has_unclear=True,
    )
    span = FragmentClauseSpan(
        span_id="test.unclear",
        unit_id="thomas.5",
        transcription_id="poxy-654-dclp",
        coptic_start=0,
        coptic_end=2,
        frag_word_start=0,
        frag_word_end=2,
    )
    # Predicting only the unclear word hits 1.0 on unclear stratum, low on clear stratum
    counts, _ = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=[fw_clear, fw_unclear],
        pred_tokens=[_tok(0, "ουσιν", "εἰμί", "V-")],
        lemmatiser=lemmatiser,
        greek_loan_morphemes=(),
    )
    stratum = _aggregate_frag_stratum([counts], [span])
    assert stratum.clear_surviving_word_count == 1
    assert stratum.unclear_surviving_word_count == 1
    assert stratum.unclear_surviving_char_f1_itacism == 1.0
    assert stratum.clear_surviving_char_f1_itacism < 0.5

    # Also verify real P.Oxy. 654 corpus preserves unclear annotations on lines 33 & 42
    repo_root = Path(__file__).resolve().parents[1]
    real = load_fragment_corpus(resolve_fragments_corpus_path(repo_root))
    found_tokens: dict[str, str] = {}
    unclear_words = 0
    for tx in real.transcriptions:
        if tx.id != "poxy-654-dclp":
            continue
        unclear_words = sum(1 for w in tx.words if w.has_unclear)
        for line in tx.lines:
            for tok in line.tokens:
                if tok.certainty == "unclear" and line.line_id in {
                    "main.33",
                    "main.42",
                }:
                    found_tokens[line.line_id] = tok.text
    assert found_tokens["main.33"] == "ουσιν"
    assert found_tokens["main.42"] == "ὃν"
    assert unclear_words >= 10


def test_duplicate_token_precision_cannot_game_ordered_lcs() -> None:
    """Ported JAK-40 invariant: duplicate predictions cannot score precision=1.0."""
    pred = ["οτι", "οτι", "η", "βασιλεια"]
    gold = ["οτι", "η", "βασιλεια"]
    lcs = _lcs_len(pred, gold)
    prec, rec, f1 = _prf(lcs, len(pred), len(gold))
    assert lcs == 3
    assert rec == 1.0
    assert prec == 0.75
    assert prec < 1.0
    assert f1 < 1.0


def test_token_aligned_loanword_rejects_wrong_seq_and_cognate_shift() -> None:
    """Zero-CCL loanword scoring requires exact m.seq alignment and keeps misses."""
    # Phonological double-consonant expansion handles ⲥⲁⲣⲝ -> σαρκί without
    # false-matching unrelated 3-char prefixes like πόλις vs πολλοί.
    assert _loanword_script_stem_matches("σαρξ", "σαρκι") is True
    assert _loanword_script_stem_matches("πολισ", "πολλυ") is False

    lemmatiser = _AttestedGreekLemmatiser(
        exact_map={
            "κοσμον": ("κόσμος", "N-"),
            "θεραπειασ": ("θεραπεία", "N-"),
        },
        ita_map={},
    )
    words = [
        FragmentWord(
            word_index=0,
            line_id="main.1",
            surviving_text="κόσμον",
            full_editorial_word="κόσμον",
        ),
        FragmentWord(
            word_index=1,
            line_id="main.1",
            surviving_text="θεραπείας",
            full_editorial_word="θεραπείας",
        ),
    ]
    span = FragmentClauseSpan(
        span_id="test.loanword",
        unit_id="thomas.31",
        transcription_id="poxy-1-dclp",
        coptic_start=0,
        coptic_end=15,
        frag_word_start=0,
        frag_word_end=2,
    )
    loan_ms = [(5, "ⲕⲟⲥⲙⲟⲥ"), (11, "ⲣⲑⲉⲣⲁⲡⲉⲩⲉ")]

    # Case 1: right Greek lemma predicted on WRONG Coptic seq (99 instead of 5) -> 0
    counts_wrong_seq, _ = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=words,
        pred_tokens=[_tok(0, "κόσμον", "κόσμος", "N-", coptic_seqs=[99])],
        lemmatiser=lemmatiser,
        greek_loan_morphemes=loan_ms,
    )
    assert counts_wrong_seq["lw_total"] == 2
    assert counts_wrong_seq["lw_hits"] == 0

    # Case 2: aligned `κόσμος` hits, while verb `θεραπεύω` vs attested noun `θεραπεία`
    # stays in the denominator as an honest miss (1/2 = 0.5, NOT dropped to 1/1 = 1.0!)
    counts_shift, _ = score_fragment_span_lacuna_aware(
        span=span,
        frag_words=words,
        pred_tokens=[
            _tok(0, "κόσμον", "κόσμος", "N-", coptic_seqs=[5]),
            _tok(1, "θεραπεύει", "θεραπεύω", "V-", coptic_seqs=[11]),
        ],
        lemmatiser=lemmatiser,
        greek_loan_morphemes=loan_ms,
    )
    stratum = _aggregate_frag_stratum([counts_shift], [span])
    assert stratum.loanword_total == 2
    assert stratum.loanword_exact_match_rate == 0.5

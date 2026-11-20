# SPDX-License-Identifier: Apache-2.0
"""Sahidic OT x Swete pairing stays structural and license-clean."""

from zetodos_thomas.dictionary.ot_bitext import (
    OT_LICENSE,
    choose_swete_text,
    normalize_swete_word,
    parse_swete_file,
    parse_tt_chapter,
)


def test_swete_latin_lookalike_folds_to_greek() -> None:
    assert normalize_swete_word("ΜΑΚAPΙΟΣ") == "μακαριοσ"
    assert OT_LICENSE == "CC-BY-SA-4.0"


def test_truncated_prefix_stays_an_exact_miss() -> None:
    from zetodos_thomas.dictionary.apply import TokenOut
    from zetodos_thomas.dictionary.score import score_logia

    token = TokenOut(
        index=0,
        coptic="ⲑⲁⲗⲁⲥⲥⲁ",
        greek="θαλασσησ",
        channel="direct",
        posterior=0.8,
        observation_count=4,
        span_head=0,
    )
    scored = score_logia({3: [token]}, ["θαλα", "θαλασσησ"], {3})
    assert scored["loanword_exact_matches"] == 1
    short = score_logia({3: [token]}, ["θαλα"], {3})
    assert short["loanword_exact_matches"] == 0
    assert short["loanword_stem_matches"] == 1
    assert short["loanword_truncated_prefix_misses"] == 1
    assert short["loanword_tokens"] == 1


def test_tt_and_swete_pair_by_verse_number() -> None:
    tt = """
    <verse_n verse_n="1" vname="Genesis 1:1">
    <norm norm="ϩⲛ"></norm><norm norm="ⲧ"></norm><norm norm="ⲁⲣⲭⲏ"></norm>
    </verse_n>
    <verse_n verse_n="2">
    <norm norm="ⲡ"></norm><norm norm="ⲕⲁϩ"></norm>
    </verse_n>
    """
    coptic = parse_tt_chapter(tt)
    assert coptic[1][2] == "ⲁⲣⲭⲏ"
    greek = parse_swete_file("1.1.1 ἐν\n1.1.1 ἀρχῇ\n1.2.1 γῆ\n")
    assert (1, 1) in greek
    chosen = choose_swete_text(
        {(1, 1): coptic[1]},
        {
            "short.txt": {(1, 1): ["α"]},
            "long.txt": {(9, 9): ["β"]},
        },
    )
    assert chosen[(1, 1)] == ["α"]

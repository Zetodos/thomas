# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Sahidic OT x Swete LXX versification recovery."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.ingest.ot_swete import (
    MAX_TOKEN_RATIO,
    _normalize_book_keys,
    _recover_chapter_pairs_dp,
    load_ot_swete_pairs,
)
from zetodos_thomas.schemas.linguistic import CopticMorpheme, CopticUnit

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = REPO_ROOT / "data" / "downloads"


def _mk_unit(
    uid: str, norms: list[str], *, greek_idx: frozenset[int] = frozenset()
) -> CopticUnit:
    ms = [
        CopticMorpheme(
            seq=i,
            group_seq=i,
            norm=w,
            lemma=w,
            pos="N" if i in greek_idx else "V",
            lang="Greek" if i in greek_idx else None,
        )
        for i, w in enumerate(norms)
    ]
    return CopticUnit(
        unit_id=uid,
        source_id="scriptorium-sahidic-ot",
        book="ot.test",
        chapter=1,
        verse=1,
        original_text=" ".join(norms),
        normalized_text=" ".join(norms),
        morphemes=ms,
    )


def test_normalize_epistle_of_jeremiah_and_joel_chapters() -> None:
    u_ep = _mk_unit("ot.Epistle_of_Jeremiah.1.2", ["ⲧⲉⲛⲟⲩ", "ⲃⲁⲃⲩⲗⲱⲛ"])
    norm_ep = _normalize_book_keys("Epistle_of_Jeremiah", {(1, 2): u_ep})
    assert (0, 2) in norm_ep

    u_joel3 = _mk_unit("ot.Joel.3.1", ["ⲁⲩⲱ", "ⲥ", "ⲛⲁ", "ϣⲱⲡⲉ"])
    u_joel4 = _mk_unit("ot.Joel.4.2", ["ⲉⲓⲥ", "ϩⲏⲏⲧⲉ"])
    norm_j = _normalize_book_keys("Joel", {(3, 1): u_joel3, (4, 2): u_joel4})
    assert (2, 28) in norm_j
    assert (3, 2) in norm_j


def test_chapter_dp_recovers_anchored_shift_and_rejects_unanchored_far_drift() -> None:
    c_anch = _mk_unit(
        "ot.Deut.14.29",
        ["ⲁⲩⲱ", "ϥ", "ⲛⲁ", "ⲉⲓ", "ⲛϭⲓ", "ⲡ", "ⲗⲉⲩⲓⲧⲏⲥ", "ϫⲉ", "ⲙⲛⲧϥ", "ⲙⲉⲣⲓⲥ"],
        greek_idx=frozenset({6, 9}),
    )
    g_anch = ["καὶ", "ὁ", "λευείτης", "ὅτι", "οὐκ", "ἔστιν", "αὐτῷ", "μερίς"]
    rec = _recover_chapter_pairs_dp([(29, c_anch)], [(26, g_anch)])
    assert len(rec) == 1
    assert rec[0][0] == 29 and rec[0][2] == 26

    # Far drift (3 verses) WITHOUT any shared Greek/proper-noun stem must be rejected
    c_unanch = _mk_unit("ot.Deut.14.19", ["ⲛⲁ", "ⲓ", "ⲧⲏⲣ", "ⲟⲩ", "ϩⲛ", "ⲛ", "ϩⲁⲗⲁⲧⲉ"])
    g_unanch = ["δεκάτην", "ἀποδεκατώσεις", "παντὸς", "γενήματος"]
    assert _recover_chapter_pairs_dp([(19, c_unanch)], [(22, g_unanch)]) == []


def test_real_ot_swete_versification_recovery_increases_clean_pairs() -> None:
    ot_zip = DOWNLOADS / "sahidic-ot" / "sahidic.ot_TT.zip"
    swete_dir = DOWNLOADS / "lxx-swete"
    if not (ot_zip.is_file() and swete_dir.is_dir()):
        return
    base = load_ot_swete_pairs(
        tt_zip=ot_zip, swete_dir=swete_dir, recover_versification=False
    )
    rec = load_ot_swete_pairs(
        tt_zip=ot_zip, swete_dir=swete_dir, recover_versification=True
    )
    assert len(rec) >= len(base) + 80
    for c_u, g_u in rec:
        ratio = max(len(c_u.morphemes), len(g_u.tokens)) / min(
            len(c_u.morphemes), len(g_u.tokens)
        )
        assert ratio <= MAX_TOKEN_RATIO

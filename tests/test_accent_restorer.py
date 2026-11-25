# SPDX-License-Identifier: Apache-2.0
"""Unit tests for deterministic offline KoineAccentRestorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from zetodos_thomas.ingest.holdouts import (
    all_excluded_nt_ids,
    greek_token_ngrams,
    load_nt_split,
)
from zetodos_thomas.ingest.morphgnt import load_morphgnt_nt
from zetodos_thomas.neural.accent_restorer import (
    KoineAccentRestorer,
    _normalized_sentence,
    _recessive_accent_fallback,
    shares_holdout_greek_ngram,
    strip_greek_to_bare,
)
from zetodos_thomas.neural.corpus import normalize_greek_unit_target
from zetodos_thomas.schemas.holdout import PassageRole

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = REPO_ROOT / "data" / "downloads"
SPLIT_PATH = REPO_ROOT / "data" / "training" / "splits" / "nt-split-v1.json"


def test_shares_holdout_greek_ngram_matches_movable_nu_and_capital() -> None:
    beatitude = (
        "μακάριοι οἱ πτωχοί τῷ πνεύματι ὅτι αὐτῶν ἐστί(ν) ἡ βασιλεία τῶν οὐρανῶν"
    )
    blocked = greek_token_ngrams(_normalized_sentence(beatitude.split()).lower())
    polycarp = (
        "καὶ ὅτι μακάριοι οἱ πτωχοὶ καὶ οἱ διωκόμενοι ἕνεκεν δικαιοσύνης,"
        " ὅτι αὐτῶν ἐστὶν ἡ βασιλεία τοῦ θεοῦ."
    )
    assert shares_holdout_greek_ngram(polycarp, blocked)
    soil = "καί ἕτερον ἔπεσεν εἰς τήν γῆν τήν ἀγαθήν καί φυέν"
    blocked_soil = greek_token_ngrams(_normalized_sentence(soil.split()).lower())
    assert shares_holdout_greek_ngram(
        "Εἰς τὴν γῆν τὴν ἀγαθήν, γῆν ῥέουσαν γάλα καὶ μέλι;",
        blocked_soil,
    )
    assert not shares_holdout_greek_ngram(
        "ὁ λόγος ἦν πρὸς τὸν θεόν, ἐν ἀρχῇ τῆς ζωῆς.",
        blocked,
    )


def test_strip_greek_to_bare_and_recessive_fallback() -> None:
    assert strip_greek_to_bare("Ἰησοῦς ὁ ζῶν") == "ιησους ο ζων"
    assert strip_greek_to_bare("λόγος") == "λογος"
    # Universal Greek morphological suffixes & breathings
    assert _recessive_accent_fallback("γεννηθηναι") == "γεννηθῆναι"
    assert _recessive_accent_fallback("εμπεπλησμενοι") == "ἐμπεπλησμένοι"
    assert _recessive_accent_fallback("απολαβωσιν").startswith("ἀπ")
    assert _recessive_accent_fallback("απαντες") == "ἅπαντες"


def test_accent_restorer_context_and_shard_roundtrip(tmp_path: Path) -> None:
    ar = KoineAccentRestorer(
        unigrams={"λογος": "λόγος", "εστιν": "ἐστιν", "ει": "εἰ", "ουκ": "οὐκ"},
        prev_bigrams={"ουκ εστιν": "ἔστιν"},
        next_bigrams={"ει συ": "εἶ"},
        enclitic_rules={"A1 εστιν": "ἐστίν"},
    )
    assert ar.restore_sentence("λογος εστιν") == "λόγος ἐστίν"
    assert ar.restore_sentence("ουκ εστιν") == "οὐκ ἔστιν"
    assert ar.restore_sentence("ει συ") == "εἶ σύ"

    shards = ar.save_shards(tmp_path)
    assert shards
    for p in shards:
        assert p.stat().st_size < 650_000
    reloaded = KoineAccentRestorer.load_shards(tmp_path)
    assert reloaded.restore_sentence("λογος εστιν") == "λόγος ἐστίν"
    assert reloaded.restore_sentence("ουκ εστιν") == "οὐκ ἔστιν"


def test_accent_restorer_high_accuracy_on_unseen_nt_dev() -> None:
    mg_dir = DOWNLOADS / "morphgnt-sblgnt"
    swete_dir = DOWNLOADS / "lxx-swete"
    if not mg_dir.is_dir():
        pytest.skip("MorphGNT not downloaded in this test environment")
    split = load_nt_split(SPLIT_PATH)
    excl = all_excluded_nt_ids(split, include_dev=True)
    ar = KoineAccentRestorer.build(
        morphgnt_dir=mg_dir,
        exclude_unit_ids=excl,
        swete_dir=swete_dir if swete_dir.is_dir() else None,
        include_patristics=False,
    )
    mg = load_morphgnt_nt(mg_dir)
    dev_ids = [
        uid for p in split.passages if p.role is PassageRole.DEV for uid in p.unit_ids
    ]
    tot = exact = 0
    for uid in dev_ids:
        gold = normalize_greek_unit_target(mg[uid])
        rest = ar.restore_sentence(strip_greek_to_bare(gold))
        for gw, rw in zip(gold.split(), rest.split(), strict=True):
            tot += 1
            if gw == rw:
                exact += 1
    assert tot >= 900
    assert (exact / tot) >= 0.96

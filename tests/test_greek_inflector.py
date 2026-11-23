# SPDX-License-Identifier: Apache-2.0
"""Tests for empirical + rule-based Koine Greek inflector, analyzer, and sandhi."""

from __future__ import annotations

from pathlib import Path

import pytest

from zetodos_thomas.morphology.greek_inflector import (
    GreekInflector,
    realize_greek_tokens,
)
from zetodos_thomas.schemas.linguistic import GreekMorphFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
MORPHGNT_DIR = REPO_ROOT / "data" / "downloads" / "morphgnt-sblgnt"


def test_realize_greek_tokens_sandhi_and_punct() -> None:
    assert realize_greek_tokens(["οὐ", "εὑρίσκει"]) == "οὐχ εὑρίσκει"
    assert realize_greek_tokens(["οὐ", "ἔχει"]) == "οὐκ ἔχει"
    assert realize_greek_tokens(["οὐ", "γάρ", "ἐστι", "."]) == "οὐ γάρ ἐστιν."
    assert realize_greek_tokens(["ἐκ", "αὐτοῦ"]) == "ἐξ αὐτοῦ"
    assert realize_greek_tokens(["ἐξ", "λόγου"]) == "ἐκ λόγου"
    assert realize_greek_tokens(["εἶπε", ",", "ἄγωμεν"]) == "εἶπεν, ἄγωμεν"


def test_oov_regular_koine_inflection_rules() -> None:
    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={},
        surface_analyses={},
    )
    # Regular 2nd decl masc/neut & 1st decl fem nouns not in empirical map
    assert (
        inf.inflect(
            "τόπος",
            GreekMorphFeatures(pos="N-", case="D", number="S", gender="M"),
        )
        == "τόπῳ"
    )
    assert (
        inf.inflect(
            "βασιλεία",
            GreekMorphFeatures(pos="N-", case="G", number="S", gender="F"),
        )
        == "βασιλείας"
    )
    assert (
        inf.inflect(
            "δῶμα",
            GreekMorphFeatures(pos="N-", case="G", number="S", gender="N"),
        )
        == "δώματος"
    )
    # OOV regular verb: present, aorist, subjunctive, imperative, participle
    aor3s = inf.inflect(
        "ἑρμηνεύω",
        GreekMorphFeatures(
            pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
        ),
    )
    assert aor3s == "ἡρμήνευσε"
    subj3s = inf.inflect(
        "γεύω",
        GreekMorphFeatures(
            pos="V-", person="3", tense="A", voice="A", mood="S", number="S"
        ),
    )
    assert subj3s.endswith("σῃ")
    assert (
        inf.inflect(
            "οὖς",
            GreekMorphFeatures(pos="N-", case="D", number="S", gender="N"),
        )
        == "ὠτί"
    )
    assert (
        inf.inflect(
            "ἕλκω",
            GreekMorphFeatures(
                pos="V-",
                tense="P",
                voice="A",
                mood="P",
                case="N",
                number="P",
                gender="M",
            ),
        )
        == "ἕλκοντες"
    )
    assert (
        inf.inflect(
            "γινώσκω",
            GreekMorphFeatures(
                pos="V-",
                tense="P",
                voice="A",
                mood="P",
                case="A",
                number="P",
                gender="M",
            ),
        )
        == "γινώσκοντας"
    )


def test_morphgnt_empirical_irregular_verbs_and_gender() -> None:
    if not MORPHGNT_DIR.is_dir():
        pytest.skip("MorphGNT not downloaded in this test environment")
    inf = GreekInflector.build(morphgnt_dir=MORPHGNT_DIR)

    assert inf.inflect(
        "λέγω",
        GreekMorphFeatures(
            pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
        ),
    ) in {"εἶπεν", "εἶπε"}
    assert inf.inflect(
        "ὁράω",
        GreekMorphFeatures(
            pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
        ),
    ) in {"εἶδεν", "εἶδε"}
    assert inf.inflect(
        "ἔρχομαι",
        GreekMorphFeatures(
            pos="V-", person="3", tense="A", voice="A", mood="I", number="S"
        ),
    ) in {"ἦλθεν", "ἦλθε"}
    assert (
        inf.inflect(
            "εὑρίσκω",
            GreekMorphFeatures(
                pos="V-", person="3", tense="F", voice="A", mood="I", number="S"
            ),
        )
        == "εὑρήσει"
    )
    # Suppletive/deponent future & aorist forms
    assert (
        inf.inflect(
            "ὁράω",
            GreekMorphFeatures(
                pos="V-", person="2", tense="F", voice="A", mood="I", number="P"
            ),
        )
        == "ὄψεσθε"
    )
    assert (
        inf.inflect(
            "ἐσθίω",
            GreekMorphFeatures(
                pos="V-", person="3", tense="F", voice="A", mood="I", number="S"
            ),
        )
        == "φάγεται"
    )
    assert (
        inf.inflect(
            "εἰσέρχομαι",
            GreekMorphFeatures(
                pos="V-", person="3", tense="A", voice="M", mood="I", number="P"
            ),
        )
        == "εἰσῆλθον"
    )
    assert inf.inflect(
        "ἐκβάλλω",
        GreekMorphFeatures(pos="V-", tense="P", voice="A", mood="N"),
    ) in {"ἐκβάλλειν", "ἐκβαλεῖν"}
    assert (
        inf.inflect(
            "γινώσκω",
            GreekMorphFeatures(
                pos="V-", person="2", tense="A", voice="A", mood="D", number="S"
            ),
        )
        == "γνῶθι"
    )
    liq = inf.inflect(
        "βάλλω",
        GreekMorphFeatures(
            pos="V-", person="2", tense="F", voice="A", mood="I", number="S"
        ),
    )
    assert "λλσ" not in liq
    assert inf.lemma_gender("ἀνατολή") == "F"
    assert inf.lemma_gender("πνεῦμα") == "N"
    assert inf.lemma_gender("βασιλεύς") == "M"

    # Surface analyzer recovers lemmas cleanly for Swete LXX forms
    parses = inf.analyze_surface("ἐποίησεν")
    assert any(lem == "ποιέω" for lem, _m, _lp in parses)

    for raw, exp_lem, exp_pos in (
        ("πρωὶ", "πρωΐ", {"D-"}),
        ("ἐπʼ", "ἐπί", {"P-"}),
        ("ΕΝ", "ἐν", {"P-"}),
        ("ἐπεφέρετο", "ἐπιφέρω", {"V-"}),
        ("ἀλλοφύλων", "ἀλλόφυλος", {"N-", "A-"}),
    ):
        sw_parses = inf.analyze_surface(raw)
        assert sw_parses[0][0] == exp_lem
        assert sw_parses[0][1].pos in exp_pos


def test_stage7_passive_inflection_paradigms() -> None:
    inf = GreekInflector(
        paradigms={},
        coarse_paradigms={},
        lemma_genders={},
        surface_analyses={},
    )
    assert (
        inf.inflect(
            "ἀποκαλύπτω",
            GreekMorphFeatures(
                pos="V-", person="3", tense="F", voice="P", mood="I", number="S"
            ),
        )
        == "ἀποκαλυφθήσεται"
    )
    assert (
        inf.inflect(
            "κρύπτω",
            GreekMorphFeatures(pos="V-", tense="A", voice="P", mood="N"),
        )
        == "κρυβῆναι"
    )
    assert (
        inf.inflect(
            "ταράσσω",
            GreekMorphFeatures(
                pos="V-", person="3", tense="A", voice="P", mood="S", number="S"
            ),
        )
        == "ταραχθῇ"
    )
    assert (
        inf.inflect(
            "ὁράω",
            GreekMorphFeatures(
                pos="V-", person="1", tense="A", voice="P", mood="I", number="S"
            ),
        )
        == "ὤφθην"
    )

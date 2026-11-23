# SPDX-License-Identifier: Apache-2.0
"""Tests for seeded eflomal lemma+POS aligner and Coptic multiword idioms."""

from __future__ import annotations

from pathlib import Path

import pytest

from zetodos_thomas.align.lemma_aligner import (
    LexicalCandidate,
    LexicalTranslationModel,
    extract_coptic_right_context,
    filter_coptic_for_alignment,
)
from zetodos_thomas.integrations.aligners.discovery import probe_eflomal
from zetodos_thomas.lexicon.idioms import IdiomAndValencyTable
from zetodos_thomas.morphology.greek_inflector import GreekInflector
from zetodos_thomas.schemas.linguistic import (
    CopticMorpheme,
    CopticUnit,
    GreekMorphFeatures,
    GreekToken,
    GreekUnit,
)

_HAS_EFLOMAL = probe_eflomal().available


def _c_unit(uid: str, pairs: list[tuple[str, str, str]]) -> CopticUnit:
    return CopticUnit(
        unit_id=uid,
        source_id="copsahhorner",
        book="john",
        chapter=1,
        verse=1,
        original_text=" ".join(p[0] for p in pairs),
        normalized_text=" ".join(p[0] for p in pairs),
        morphemes=[
            CopticMorpheme(group_seq=i, norm=n, lemma=lem, pos=pos)
            for i, (n, lem, pos) in enumerate(pairs)
        ],
    )


def _g_unit(uid: str, pairs: list[tuple[str, str, str]]) -> GreekUnit:
    return GreekUnit(
        unit_id=uid,
        source_id="morphgnt-sblgnt",
        book="john",
        chapter=1,
        verse=1,
        original_text=" ".join(p[0] for p in pairs),
        normalized_text=" ".join(p[0] for p in pairs),
        tokens=[
            GreekToken(
                id=f"{uid}.{i}",
                unit_id=uid,
                seq=i,
                surface=s,
                normalized=s,
                lemma=lem,
                morph=GreekMorphFeatures.from_morphgnt(pos, "--------"),
            )
            for i, (s, lem, pos) in enumerate(pairs)
        ],
    )


def test_filter_coptic_removes_analytic_clitics() -> None:
    morphs = [
        CopticMorpheme(group_seq=0, norm="ⲁ", lemma="ⲁ", pos="APST"),
        CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
        CopticMorpheme(group_seq=0, norm="ϣⲓⲛⲉ", lemma="ϣⲓⲛⲉ", pos="V"),
        CopticMorpheme(group_seq=1, norm="ⲛϭⲓ", lemma="ⲛϭⲓ", pos="PTC"),
        CopticMorpheme(group_seq=1, norm="ⲡ", lemma="ⲡ", pos="ART"),
        CopticMorpheme(group_seq=1, norm="ⲣⲱⲙⲉ", lemma="ⲣⲱⲙⲉ", pos="N"),
    ]
    kept = filter_coptic_for_alignment(morphs)
    lemmas = [k[2] for k in kept]
    assert "ⲁ" not in lemmas
    assert "ⲛϭⲓ" not in lemmas
    assert lemmas == ["ⲛⲧⲟϥ", "ϣⲓⲛⲉ", "ⲡ", "ⲣⲱⲙⲉ"]


def test_idiom_and_valency_matching(tmp_path: Path) -> None:
    # Default table has ZERO hardcoded idioms or verb/prep case overrides
    empty_table = IdiomAndValencyTable()
    assert empty_table._idioms == ()
    assert empty_table.verb_cases == {}
    assert empty_table.prep_cases == {}

    # Build empirically from clean synthetic bitext (no curated rules!)
    nt_pairs: list[tuple[CopticUnit, GreekUnit]] = []
    for i in range(6):
        c_u = _c_unit(
            f"acts.1.{i + 1}",
            [
                ("ϥ", "ⲛⲧⲟϥ", "PPERS"),
                ("ϩⲉ", "ϩⲉ", "V"),
                ("ⲉ", "ⲉ", "PREP"),
                ("ⲡ", "ⲡ", "ART"),
                ("ⲛⲟⲩⲧⲉ", "ⲛⲟⲩⲧⲉ", "N"),
            ],
        )
        g_u = GreekUnit(
            unit_id=f"acts.1.{i + 1}",
            source_id="morphgnt-sblgnt",
            book="acts",
            chapter=1,
            verse=i + 1,
            original_text="εὗρεν θεόν ἐν οἴκῳ",
            normalized_text="εὗρεν θεόν ἐν οἴκῳ",
            tokens=[
                GreekToken(
                    id=f"acts.1.{i + 1}.0",
                    unit_id=f"acts.1.{i + 1}",
                    seq=0,
                    surface="εὗρεν",
                    normalized="εὗρεν",
                    lemma="εὑρίσκω",
                    morph=GreekMorphFeatures.from_morphgnt("V-", "3AAI-S--"),
                ),
                GreekToken(
                    id=f"acts.1.{i + 1}.1",
                    unit_id=f"acts.1.{i + 1}",
                    seq=1,
                    surface="θεόν",
                    normalized="θεόν",
                    lemma="θεός",
                    morph=GreekMorphFeatures.from_morphgnt("N-", "----ASM-"),
                ),
                GreekToken(
                    id=f"acts.1.{i + 1}.2",
                    unit_id=f"acts.1.{i + 1}",
                    seq=2,
                    surface="ἐν",
                    normalized="ἐν",
                    lemma="ἐν",
                    morph=GreekMorphFeatures.from_morphgnt("P-", "--------"),
                ),
                GreekToken(
                    id=f"acts.1.{i + 1}.3",
                    unit_id=f"acts.1.{i + 1}",
                    seq=3,
                    surface="οἴκῳ",
                    normalized="οἴκῳ",
                    lemma="οἶκος",
                    morph=GreekMorphFeatures.from_morphgnt("N-", "----DSM-"),
                ),
            ],
        )
        nt_pairs.append((c_u, g_u))

    gen_units = [
        GreekUnit(
            unit_id=f"hebrews.6.{k + 4}",
            source_id="morphgnt-sblgnt",
            book="hebrews",
            chapter=6,
            verse=k + 4,
            original_text="γεύσηται δωρεᾶς",
            normalized_text="γεύσηται δωρεᾶς",
            tokens=[
                GreekToken(
                    id=f"hebrews.6.{k + 4}.0",
                    unit_id=f"hebrews.6.{k + 4}",
                    seq=0,
                    surface="γεύσηται",
                    normalized="γεύσηται",
                    lemma="γεύομαι",
                    morph=GreekMorphFeatures.from_morphgnt("V-", "3AMS-S--"),
                ),
                GreekToken(
                    id=f"hebrews.6.{k + 4}.1",
                    unit_id=f"hebrews.6.{k + 4}",
                    seq=1,
                    surface="δωρεᾶς",
                    normalized="δωρεᾶς",
                    lemma="δωρεά",
                    morph=GreekMorphFeatures.from_morphgnt("N-", "----GSF-"),
                ),
            ],
        )
        for k in range(3)
    ]

    table = IdiomAndValencyTable.build_from_bitext(
        nt_pairs=nt_pairs,
        morphgnt_units=[g for _c, g in nt_pairs] + gen_units,
    )
    assert table.verb_object_case("γεύομαι") == "G"
    assert table.prep_case("ἐν") == "D"
    morphs = [
        CopticMorpheme(group_seq=0, norm="ϥ", lemma="ⲛⲧⲟϥ", pos="PPERS"),
        CopticMorpheme(group_seq=0, norm="ⲛⲁ", lemma="ⲛⲁ", pos="FUT"),
        CopticMorpheme(group_seq=0, norm="ϩⲉ", lemma="ϩⲉ", pos="V"),
        CopticMorpheme(group_seq=1, norm="ⲉ", lemma="ⲉ", pos="PREP"),
        CopticMorpheme(group_seq=1, norm="ⲟⲩ", lemma="ⲟⲩ", pos="ART"),
        CopticMorpheme(group_seq=1, norm="ⲛⲟⲩⲧⲉ", lemma="ⲛⲟⲩⲧⲉ", pos="N"),
    ]
    hit = table.match_at(morphs, 2)
    assert hit is not None
    rule, span_len = hit
    assert rule.greek_lemma == "εὑρίσκω"
    assert rule.required_case == "A"
    assert span_len == 2

    out_p = tmp_path / "idiom-valency.json"
    table.save_json(out_p)
    reloaded = IdiomAndValencyTable.load_json(out_p)
    assert reloaded.match_at(morphs, 2) == hit
    assert reloaded.verb_object_case("γεύομαι") == "G"
    assert reloaded.prep_case("ἐν") == "D"


def test_right_context_and_shard_roundtrip_without_eflomal(tmp_path: Path) -> None:
    u_prep = _c_unit(
        "ctx.unit.1", [("ϩⲉ", "ϩⲉ", "V"), ("ⲉ", "ⲉ", "PREP"), ("ⲙⲟⲩ", "ⲙⲟⲩ", "N")]
    )
    u_end = _c_unit("ctx.unit.2", [("ϩⲉ", "ϩⲉ", "V")])
    assert extract_coptic_right_context(u_prep.morphemes, 0) == "prep:ⲉ"
    assert extract_coptic_right_context(u_end.morphemes, 0) == "END"

    c_fall = LexicalCandidate(
        greek_lemma="πίπτω",
        greek_pos="V-",
        score=-0.3,
        prob_g_given_c=0.74,
        prob_c_given_g=0.70,
        provenance="N",
        count_nt=10.0,
        count_ot=0.0,
    )
    c_find = LexicalCandidate(
        greek_lemma="εὑρίσκω",
        greek_pos="V-",
        score=-0.5,
        prob_g_given_c=0.60,
        prob_c_given_g=0.55,
        provenance="N",
        count_nt=8.0,
        count_ot=0.0,
    )
    c_find_ctx = LexicalCandidate(
        greek_lemma="εὑρίσκω",
        greek_pos="V-",
        score=-0.1,
        prob_g_given_c=0.90,
        prob_c_given_g=0.80,
        provenance="N",
        count_nt=8.0,
        count_ot=0.0,
    )
    model = LexicalTranslationModel(
        by_lemma_pos={("ϩⲉ", "V"): [c_fall, c_find]},
        by_lemma={"ϩⲉ": [c_fall, c_find]},
        by_lemma_ctx={("ϩⲉ", "V", "prep:ⲉ"): [c_find_ctx]},
    )
    shards = model.save_shards(tmp_path)
    assert any(p.name.startswith("lemma-ctx-") for p in shards)
    reloaded = LexicalTranslationModel.load_shards(tmp_path)
    assert (
        reloaded.lookup(coptic_lemma="ϩⲉ", coptic_pos="V", right_ctx="prep:ⲉ")[
            0
        ].greek_lemma
        == "εὑρίσκω"
    )
    assert (
        reloaded.lookup(coptic_lemma="ϩⲉ", coptic_pos="V", right_ctx="END")[
            0
        ].greek_lemma
        == "πίπτω"
    )


@pytest.mark.skipif(not _HAS_EFLOMAL, reason="eflomal not installed (run thomas setup)")
def test_eflomal_synthetic_bitext_and_domain_weighting() -> None:
    inf = GreekInflector(
        paradigms={}, coarse_paradigms={}, lemma_genders={}, surface_analyses={}
    )
    nt_pairs = [
        (
            _c_unit(
                f"nt.a.{i}",
                [
                    ("ⲡ", "ⲡ", "ART"),
                    ("ϣⲁϫⲉ", "ϣⲁϫⲉ", "N"),
                    ("ⲛ", "ⲛ", "PREP"),
                    ("ⲛⲟⲩⲧⲉ", "ⲛⲟⲩⲧⲉ", "N"),
                ],
            ),
            _g_unit(
                f"nt.a.{i}",
                [
                    ("ὁ", "ὁ", "RA"),
                    ("λόγος", "λόγος", "N-"),
                    ("τοῦ", "ὁ", "RA"),
                    ("θεοῦ", "θεός", "N-"),
                ],
            ),
        )
        for i in range(10)
    ] + [
        (
            _c_unit(f"nt.b.{i}", [("ϣⲁϫⲉ", "ϣⲁϫⲉ", "N"), ("ⲛⲁⲛⲟⲩ", "ⲛⲁⲛⲟⲩ", "VSTAT")]),
            _g_unit(f"nt.b.{i}", [("λόγος", "λόγος", "N-"), ("καλός", "καλός", "A-")]),
        )
        for i in range(6)
    ]
    ot_pairs = [
        (
            _c_unit(f"ot.{i}", [("ϣⲁϫⲉ", "ϣⲁϫⲉ", "N"), ("ⲛⲟⲩⲧⲉ", "ⲛⲟⲩⲧⲉ", "N")]),
            _g_unit(f"ot.{i}", [("ῥῆμα", "ῥῆμα", "N-"), ("θεός", "θεός", "N-")]),
        )
        for i in range(6)
    ]
    model = LexicalTranslationModel.build_from_bitext(
        nt_pairs=nt_pairs,
        ot_pairs=ot_pairs,
        inflector=inf,
        nt_weight=3.0,
    )
    cands = model.lookup(coptic_lemma="ϣⲁϫⲉ", coptic_pos="N")
    assert cands and cands[0].greek_lemma == "λόγος"
    assert cands[0].provenance == "N"


@pytest.mark.skipif(not _HAS_EFLOMAL, reason="eflomal not installed (run thomas setup)")
def test_contextual_disambiguation_and_shard_roundtrip(tmp_path: Path) -> None:
    inf = GreekInflector(
        paradigms={}, coarse_paradigms={}, lemma_genders={}, surface_analyses={}
    )
    nt_pairs = (
        [
            (
                _c_unit(
                    f"ctx.a.{i}",
                    [("ϩⲉ", "ϩⲉ", "V"), ("ⲉ", "ⲉ", "PREP"), ("ⲙⲟⲩ", "ⲙⲟⲩ", "N")],
                ),
                _g_unit(
                    f"ctx.a.{i}",
                    [
                        ("εὗρεν", "εὑρίσκω", "V-"),
                        ("εἰς", "εἰς", "P-"),
                        ("θάνατον", "θάνατος", "N-"),
                    ],
                ),
            )
            for i in range(6)
        ]
        + [
            (
                _c_unit(f"ctx.b.{i}", [("ϩⲉ", "ϩⲉ", "V")]),
                _g_unit(f"ctx.b.{i}", [("ἔπεσεν", "πίπτω", "V-")]),
            )
            for i in range(6)
        ]
        + [
            (
                _c_unit(
                    f"ctx.c.{i}",
                    [
                        ("ϣⲓⲛⲉ", "ϣⲓⲛⲉ", "V"),
                        ("ⲉ", "ⲉ", "PREP"),
                        ("ⲛⲟⲩⲧⲉ", "ⲛⲟⲩⲧⲉ", "N"),
                    ],
                ),
                _g_unit(
                    f"ctx.c.{i}",
                    [
                        ("ἐζήτησεν", "ζητέω", "V-"),
                        ("εἰς", "εἰς", "P-"),
                        ("θεόν", "θεός", "N-"),
                    ],
                ),
            )
            for i in range(4)
        ]
    )
    model = LexicalTranslationModel.build_from_bitext(
        nt_pairs=nt_pairs,
        ot_pairs=[],
        inflector=inf,
        nt_weight=3.0,
    )
    assert extract_coptic_right_context(nt_pairs[0][0].morphemes, 0) == "prep:ⲉ"
    assert extract_coptic_right_context(nt_pairs[6][0].morphemes, 0) == "END"

    shards = model.save_shards(tmp_path)
    assert any(p.name.startswith("lemma-ctx-") for p in shards)
    reloaded = LexicalTranslationModel.load_shards(tmp_path)
    assert (
        reloaded.lookup(coptic_lemma="ϩⲉ", coptic_pos="V", right_ctx="prep:ⲉ")[
            0
        ].greek_lemma
        == "εὑρίσκω"
    )
    assert (
        reloaded.lookup(coptic_lemma="ϩⲉ", coptic_pos="V", right_ctx="END")[
            0
        ].greek_lemma
        == "πίπτω"
    )

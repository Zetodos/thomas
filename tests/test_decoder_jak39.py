# SPDX-License-Identifier: Apache-2.0
"""JAK-39: minimal deterministic decoder — ranking, abstention, leakage.

Regression coverage for Jake review 5898099717 on PR 18.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.reconstruct.decoder import decode_request, request_from_surfaces
from zetodos_thomas.reconstruct.model import (
    DecodeModel,
    assert_compatible_provenance,
    index_decode_model,
    load_decode_model,
    validate_decode_evidence,
)
from zetodos_thomas.reconstruct.ranking import RANKING_RULES_ID
from zetodos_thomas.reconstruct.validate import validate_decode_result
from zetodos_thomas.schemas.decode import (
    AbstentionReason,
    DecodeRequest,
    DecoderEvidenceKind,
    DecodeSpanInput,
    GreekFormKind,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.lexicon import (
    ConfidenceComponents,
    ConfidenceEstimate,
    LexicalCandidateTable,
    LexicalLayer,
    ObservedWordformCandidate,
    SplitMembership,
)
from zetodos_thomas.schemas.mappings import (
    AblationCondition,
    AblationPackage,
    ConstructionFamily,
    SpotCheckRecord,
    StatisticalConstruction,
    StatisticalConstructionTable,
    VerifiedLoanword,
    VerifiedLoanwordTable,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spot() -> SpotCheckRecord:
    return SpotCheckRecord(
        checked_by="test",
        checked_at_utc="2026-11-13T00:00:00Z",
        method="fixture",
        result="pass",
        tei_entry_xml_id="C0",
        tei_form_xml_ids=["CF0"],
        tei_orth_attested="ⲙⲁⲕⲁⲣⲓⲟⲥ",
        tei_greek_lemma="μακάριος",
        tei_greek_lemma_id="0",
        parent_tei_sha256="0" * 64,
    )


def _tiny_model(*, include_loanwords: bool) -> DecodeModel:
    condition = (
        AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
        if include_loanwords
        else AblationCondition.NT_ALIGNER_ONLY
    )
    package = AblationPackage(
        id="ablation-fixture",
        title="fixture",
        condition=condition,
        includes={
            "lexical_candidates": "fixture",
            "statistical_phrase_constructions": "fixture",
            "verified_loanwords": "fixture" if include_loanwords else None,
        },
        excludes={"verified_loanword_constraints": not include_loanwords},
    )
    lexical = LexicalCandidateTable(
        id="lex-fixture",
        title="fixture",
        layers={
            "observed_wordform": {"entry_count": 2},
            "inferred_lemma": {"entry_count": 0},
            "confidence_estimate": {"entry_count": 2},
        },
        observed_wordforms=[
            ObservedWordformCandidate(
                coptic_wordform="ⲁⲛ",
                greek_wordform="ουχ",
                observation_count=4,
                per_aligner_observation_counts={"eflomal": 4, "fast_align": 4},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.1"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲁⲛ",
                greek_wordform="ουκ",
                observation_count=2,
                per_aligner_observation_counts={"eflomal": 2, "fast_align": 1},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.2"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ϣⲁϫⲉ",
                greek_wordform="λογος",
                observation_count=3,
                per_aligner_observation_counts={"eflomal": 3, "fast_align": 3},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.1"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲁⲛⲟⲛ",
                greek_wordform="πληρωματος",
                observation_count=1,
                per_aligner_observation_counts={"eflomal": 1},
                contributing_aligners=["eflomal"],
                source_unit_ids=["john.1.16"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲇⲉ",
                greek_wordform="δε",
                observation_count=69,
                per_aligner_observation_counts={"eflomal": 69, "fast_align": 69},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.1"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲇⲉ",
                greek_wordform="ουδε",
                observation_count=10,
                per_aligner_observation_counts={"eflomal": 10, "fast_align": 10},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.3"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲡⲁⲣⲁⲃⲟⲗⲏ",
                greek_wordform="παραβολην",
                observation_count=2,
                per_aligner_observation_counts={"eflomal": 2, "fast_align": 2},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["matthew.13.18"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲁⲓⲧⲉⲓ",
                greek_wordform="αιτειτε",
                observation_count=1,
                per_aligner_observation_counts={"eflomal": 1, "fast_align": 1},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["matthew.7.7"],
                split_membership=SplitMembership.TRAIN,
            ),
            ObservedWordformCandidate(
                coptic_wordform="ⲁⲓⲧⲉⲓ",
                greek_wordform="αιτουντι",
                observation_count=1,
                per_aligner_observation_counts={"eflomal": 1, "fast_align": 1},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["matthew.7.8"],
                split_membership=SplitMembership.TRAIN,
            ),
        ],
        inferred_lemmas=[],
        confidence_estimates=[
            ConfidenceEstimate(
                applies_to_layer=LexicalLayer.OBSERVED_WORDFORM,
                key={"coptic_wordform": "ⲁⲛ", "greek_wordform": "ουχ"},
                method="heuristic_ranking_uncalibrated_v1",
                heuristic_ranking_score=0.9,
                components=ConfidenceComponents(),
            ),
            ConfidenceEstimate(
                applies_to_layer=LexicalLayer.OBSERVED_WORDFORM,
                key={"coptic_wordform": "ⲁⲛ", "greek_wordform": "ουκ"},
                method="heuristic_ranking_uncalibrated_v1",
                heuristic_ranking_score=0.4,
                components=ConfidenceComponents(),
            ),
        ],
    )
    constructions = StatisticalConstructionTable(
        id="stat-fixture",
        title="fixture",
        constructions=[
            StatisticalConstruction(
                construction_id="stat-negation-ⲁⲛ--ουχ",
                family=ConstructionFamily.NEGATION,
                coptic_pattern=["ⲁⲛ"],
                greek_pattern=["ουχ"],
                observation_count=4,
                per_aligner_observation_counts={"eflomal": 1, "fast_align": 4},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.1"],
            ),
            StatisticalConstruction(
                construction_id="stat-negation-ⲁⲛ--ουκ-ειμι",
                family=ConstructionFamily.NEGATION,
                coptic_pattern=["ⲁⲛ"],
                greek_pattern=["ουκ", "ειμι"],
                observation_count=3,
                per_aligner_observation_counts={"eflomal": 3, "fast_align": 3},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.20"],
            ),
            StatisticalConstruction(
                construction_id="stat-article-ⲡ-ϣⲁϫⲉ--ο-λογος",
                family=ConstructionFamily.ARTICLE,
                coptic_pattern=["ⲡ", "ϣⲁϫⲉ"],
                greek_pattern=["ο", "λογος"],
                observation_count=2,
                per_aligner_observation_counts={"eflomal": 2, "fast_align": 2},
                contributing_aligners=["eflomal", "fast_align"],
                source_unit_ids=["john.1.1", "john.1.14"],
            ),
        ],
        summary={"phase2_active_aligners": ["eflomal", "fast_align"]},
    )
    loanwords = None
    if include_loanwords:
        loanwords = VerifiedLoanwordTable(
            id="loan-fixture",
            title="fixture",
            entries=[
                VerifiedLoanword(
                    loanword_id="verified-makarios",
                    coptic_lemma="ⲙⲁⲕⲁⲣⲓⲟⲥ",
                    coptic_forms=["ⲙⲁⲕⲁⲣⲓⲟⲥ", "ⲙⲁⲕⲁⲣⲓⲟⲛ"],
                    greek_lemma="μακάριος",
                    greek_lemma_normalized="μακαριος",
                    tla_entry_id="C0",
                    ddglc_greek_lemma_id="0",
                    attribution="fixture",
                    spot_check=_spot(),
                ),
                VerifiedLoanword(
                    loanword_id="verified-de",
                    coptic_lemma="ⲇⲉ",
                    coptic_forms=["ⲇⲉ"],
                    greek_lemma="δέ",
                    greek_lemma_normalized="δε",
                    tla_entry_id="C1",
                    ddglc_greek_lemma_id="1",
                    attribution="fixture",
                    spot_check=_spot(),
                ),
                VerifiedLoanword(
                    loanword_id="verified-parabole",
                    coptic_lemma="ⲡⲁⲣⲁⲃⲟⲗⲏ",
                    coptic_forms=["ⲡⲁⲣⲁⲃⲟⲗⲏ"],
                    greek_lemma="παραβολή",
                    greek_lemma_normalized="παραβολη",
                    tla_entry_id="C2",
                    ddglc_greek_lemma_id="2",
                    attribution="fixture",
                    spot_check=_spot(),
                ),
                VerifiedLoanword(
                    loanword_id="verified-aitei",
                    coptic_lemma="ⲁⲓⲧⲉⲓ",
                    coptic_forms=["ⲁⲓⲧⲉⲓ", "ⲁⲓⲧⲉ"],
                    greek_lemma="αἰτέω",
                    greek_lemma_normalized="αιτεω",
                    tla_entry_id="C3",
                    ddglc_greek_lemma_id="3",
                    attribution="fixture",
                    spot_check=_spot(),
                ),
            ],
        )
    model = DecodeModel(
        condition=condition,
        package=package,
        lexical=lexical,
        constructions=constructions,
        loanwords=loanwords,
        provenance={"owned_by": "JAK-39", "fixture": True},
    )
    return index_decode_model(model)


def test_construction_outranks_lexical_and_is_stable() -> None:
    model = _tiny_model(include_loanwords=False)
    req = request_from_surfaces(
        request_id="t1",
        unit_id="demo.1",
        surfaces=["ⲁⲛ"],
        ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
        max_candidates_per_span=5,
    )
    a = decode_request(model, req)
    b = decode_request(model, req)
    assert a.model_dump(mode="json") == b.model_dump(mode="json")
    span = a.spans[0]
    assert span.status is SpanDecodeStatus.CANDIDATES
    assert span.candidates[0].evidence[0].kind is (
        DecoderEvidenceKind.STATISTICAL_CONSTRUCTION
    )
    assert span.candidates[0].greek_surfaces == ["ουχ"]
    kinds = {c.evidence[0].kind for c in span.candidates}
    assert DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM in kinds
    assert a.ranking_rules_id == RANKING_RULES_ID
    report = validate_decode_result(a)
    assert report.ok, report.issues


def test_dedupe_by_greek_hypothesis_de_k2() -> None:
    """Jake r4568699478: ⲇⲉ with k=2 → distinct δε + ουδε."""
    model = _tiny_model(include_loanwords=True)
    out = decode_request(
        model,
        request_from_surfaces(
            request_id="de-k2",
            unit_id="demo.de",
            surfaces=["ⲇⲉ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            max_candidates_per_span=2,
        ),
    )
    span = out.spans[0]
    assert span.status is SpanDecodeStatus.CANDIDATES
    greek = [tuple(c.greek_surfaces) for c in span.candidates]
    assert greek == [("δε",), ("ουδε",)]
    top = span.candidates[0]
    assert top.greek_surfaces == ["δε"]
    kinds = {e.kind for e in top.evidence}
    assert DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM in kinds
    assert DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD in kinds
    assert validate_decode_result(out).ok


def test_construction_id_hint_keeps_alternatives() -> None:
    """Jake r4568699485: valid hint boosts but does not drop ουκ / ουκ ειμι."""
    model = _tiny_model(include_loanwords=False)
    out = decode_request(
        model,
        request_from_surfaces(
            request_id="hint-an",
            unit_id="demo.1",
            surfaces=["ⲁⲛ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
            construction_id="stat-negation-ⲁⲛ--ουχ",
            max_candidates_per_span=10,
        ),
    )
    span = out.spans[0]
    assert span.status is SpanDecodeStatus.CANDIDATES
    assert span.candidates[0].greek_surfaces == ["ουχ"]
    surfaces = {tuple(c.greek_surfaces) for c in span.candidates}
    assert ("ουχ",) in surfaces
    assert ("ουκ",) in surfaces
    assert ("ουκ", "ειμι") in surfaces

    unknown = decode_request(
        model,
        request_from_surfaces(
            request_id="hint-bad",
            unit_id="demo.1",
            surfaces=["ⲁⲛ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
            construction_id="stat-does-not-exist",
        ),
    )
    assert (
        unknown.spans[0].abstention_reason is AbstentionReason.CONSTRUCTION_ID_UNKNOWN
    )


def test_insufficient_evidence_anon_singleton() -> None:
    """Jake r4568699492: ⲁⲛⲟⲛ → πληρωματος singleton eflomal-only abstains."""
    model = _tiny_model(include_loanwords=False)
    out = decode_request(
        model,
        request_from_surfaces(
            request_id="anon",
            unit_id="demo.we",
            surfaces=["ⲁⲛⲟⲛ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
        ),
    )
    assert out.spans[0].status is SpanDecodeStatus.ABSTAIN
    assert out.spans[0].abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert out.spans[0].candidates == []


def test_loanword_lemma_below_attested_parabole_and_aitei() -> None:
    """Jake r4568699501: attested surfaces outrank CCL lemmas."""
    model = _tiny_model(include_loanwords=True)
    para = decode_request(
        model,
        request_from_surfaces(
            request_id="para",
            unit_id="demo.para",
            surfaces=["ⲡⲁⲣⲁⲃⲟⲗⲏ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            max_candidates_per_span=5,
        ),
    )
    assert para.spans[0].status is SpanDecodeStatus.CANDIDATES
    top = para.spans[0].candidates[0]
    assert top.greek_surfaces == ["παραβολην"]
    assert top.form_kind is GreekFormKind.ATTESTED_SURFACE
    greek = [tuple(c.greek_surfaces) for c in para.spans[0].candidates]
    assert ("παραβολην",) in greek
    assert ("παραβολη",) in greek
    assert greek.index(("παραβολην",)) < greek.index(("παραβολη",))
    lemma = next(
        c for c in para.spans[0].candidates if c.greek_surfaces == ["παραβολη"]
    )
    assert lemma.form_kind is GreekFormKind.LEMMA_CLUE

    aitei = decode_request(
        model,
        request_from_surfaces(
            request_id="aitei",
            unit_id="demo.aitei",
            surfaces=["ⲁⲓⲧⲉⲓ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            max_candidates_per_span=5,
        ),
    )
    surfaces = [tuple(c.greek_surfaces) for c in aitei.spans[0].candidates]
    assert ("αιτεω",) in surfaces
    assert ("αιτειτε",) in surfaces or ("αιτουντι",) in surfaces
    assert surfaces[0] != ("αιτεω",)
    assert aitei.spans[0].candidates[0].form_kind is GreekFormKind.ATTESTED_SURFACE


def test_loanword_arm_b_vs_arm_a_separation() -> None:
    arm_b = _tiny_model(include_loanwords=True)
    arm_a = _tiny_model(include_loanwords=True)
    arm_a.condition = AblationCondition.NT_ALIGNER_ONLY
    arm_a.package = AblationPackage(
        id="ablation-fixture-a",
        title="fixture",
        condition=AblationCondition.NT_ALIGNER_ONLY,
        includes={
            "lexical_candidates": "fixture",
            "statistical_phrase_constructions": "fixture",
            "verified_loanwords": None,
        },
        excludes={"verified_loanword_constraints": True},
    )
    out_b = decode_request(
        arm_b,
        request_from_surfaces(
            request_id="t5b",
            unit_id="thomas.7",
            surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        ),
    )
    assert out_b.spans[0].status is SpanDecodeStatus.CANDIDATES
    assert out_b.spans[0].candidates[0].form_kind is GreekFormKind.LEMMA_CLUE
    assert out_b.spans[0].candidates[0].greek_surfaces == ["μακαριος"]

    out_a = decode_request(
        arm_a,
        request_from_surfaces(
            request_id="t5a",
            unit_id="thomas.7",
            surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
        ),
    )
    assert out_a.spans[0].abstention_reason is (
        AbstentionReason.ABLATION_EXCLUDES_LOANWORDS
    )


def test_poxy_override_bypass_and_cross_unit_ids() -> None:
    """Jake r4568699509: fragment bypass + cross-unit token/source rejected."""
    model = _tiny_model(include_loanwords=True)
    bypass = DecodeRequest(
        id="bypass",
        ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        unit_id="poxy-654-dclp",
        spans=[
            DecodeSpanInput(
                span_id="v1:decode-input/thomas.7/s/0",
                source_id="decode-input",
                unit_id="thomas.7",
                surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            )
        ],
    )
    out = decode_request(model, bypass)
    assert out.spans[0].status is SpanDecodeStatus.ABSTAIN
    assert out.spans[0].abstention_reason in {
        AbstentionReason.FORBIDDEN_QUERY_UNIT,
        AbstentionReason.INPUT_ID_INCONSISTENT,
    }

    cross = DecodeRequest(
        id="cross",
        ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        unit_id="thomas.7",
        spans=[
            DecodeSpanInput(
                span_id="v1:copsahhorner/john.1.1/s/x",
                source_id="copsahhorner",
                unit_id="thomas.7",
                token_ids=["v1:copsahhorner/john.1.1/t/0"],
                surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            )
        ],
    )
    out2 = decode_request(model, cross)
    assert out2.spans[0].abstention_reason is AbstentionReason.INPUT_ID_INCONSISTENT

    mismatch_src = DecodeRequest(
        id="mismatch-src",
        ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        unit_id="thomas.7",
        spans=[
            DecodeSpanInput(
                span_id="v1:decode-input/thomas.7/s/0",
                source_id="other-source",
                unit_id="thomas.7",
                surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            )
        ],
    )
    out3 = decode_request(model, mismatch_src)
    assert out3.spans[0].abstention_reason is AbstentionReason.INPUT_ID_INCONSISTENT


def test_include_inferred_lemmas_removed_from_v1_schema() -> None:
    """Jake r4568699520: no silent no-op flag on DecodeRequest."""
    assert "include_inferred_lemmas" not in DecodeRequest.model_fields


def test_provenance_mismatch_and_holdout_fail_closed() -> None:
    """Jake r4568699505: mismatched corpus/split + holdout observations fail."""
    model = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    lex = model.lexical.model_copy(deep=True)
    con = model.constructions.model_copy(deep=True)
    con.provenance = {
        **con.provenance,
        "split_id": "horner-vaticanus-nt-split-v2",
        "pilot_corpus_id": "horner-vaticanus-nt-pilot-v2",
    }
    with pytest.raises(ValueError, match="provenance mismatch"):
        assert_compatible_provenance(lex, con)

    bad = model.lexical.model_copy(deep=True)
    bad.observed_wordforms[0].source_unit_ids = ["matthew.13.31"]
    with pytest.raises(ValueError, match="lexical candidate validation failed"):
        validate_decode_evidence(
            repo_root=REPO_ROOT,
            lexical=bad,
            constructions=model.constructions,
            loanwords=model.loanwords,
            require_loanword_tei=True,
        )


def test_committed_ablation_models_load_and_decode() -> None:
    model_b = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    assert model_b.package.adopted_aligner is None
    assert model_b.provenance.get("artifact_digests", {}).get("lexical_sha256")
    assert model_b.provenance.get("evidence_provenance", {}).get("split_id")

    de = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-de",
            unit_id="demo.de",
            surfaces=["ⲇⲉ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            max_candidates_per_span=2,
        ),
    )
    assert [tuple(c.greek_surfaces) for c in de.spans[0].candidates] == [
        ("δε",),
        ("ουδε",),
    ]

    neg = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-neg",
            unit_id="demo.neg",
            surfaces=["ⲁⲛ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            construction_id="stat-negation-ⲁⲛ--ουχ",
            max_candidates_per_span=10,
        ),
    )
    surfaces = {tuple(c.greek_surfaces) for c in neg.spans[0].candidates}
    assert ("ουχ",) in surfaces
    assert ("ουκ",) in surfaces or ("ουκ", "ειμι") in surfaces

    anon = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-anon",
            unit_id="demo.we",
            surfaces=["ⲁⲛⲟⲛ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        ),
    )
    assert anon.spans[0].abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE

    para = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-para",
            unit_id="demo.para",
            surfaces=["ⲡⲁⲣⲁⲃⲟⲗⲏ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            max_candidates_per_span=5,
        ),
    )
    g = [tuple(c.greek_surfaces) for c in para.spans[0].candidates]
    assert g[0] != ("παραβολη",)
    assert ("παραβολην",) in g
    assert ("παραβολη",) in g
    assert g.index(("παραβολην",)) < g.index(("παραβολη",))

    aitei = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-aitei",
            unit_id="demo.aitei",
            surfaces=["ⲁⲓⲧⲉⲓ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            max_candidates_per_span=5,
        ),
    )
    ag = [tuple(c.greek_surfaces) for c in aitei.spans[0].candidates]
    assert ag[0] != ("αιτεω",)
    assert ("αιτειτε",) in ag or ("αιτουντι",) in ag

    assert validate_decode_result(de).ok
    assert validate_decode_result(para).ok


def test_cli_run_validate_and_ranking_rules(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "decode.json"
    result = runner.invoke(
        app,
        [
            "decode",
            "run",
            "--repo-root",
            str(REPO_ROOT),
            "--ablation",
            "nt_aligner_plus_verified_loanwords",
            "--surface",
            "ⲙⲁⲕⲁⲣⲓⲟⲥ",
            "--unit-id",
            "thomas.7",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["owned_by"] == "JAK-39"
    assert payload["provenance"]["artifact_digests"]["lexical_sha256"]

    v = runner.invoke(app, ["decode", "validate", str(out)])
    assert v.exit_code == 0, v.output

    rules = runner.invoke(app, ["decode", "ranking-rules"])
    assert rules.exit_code == 0
    assert RANKING_RULES_ID in rules.output

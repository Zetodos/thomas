# SPDX-License-Identifier: Apache-2.0
"""JAK-39: minimal deterministic decoder — ranking, abstention, leakage."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.reconstruct.decoder import decode_request, request_from_surfaces
from zetodos_thomas.reconstruct.model import (
    DecodeModel,
    index_decode_model,
    load_decode_model,
)
from zetodos_thomas.reconstruct.ranking import RANKING_RULES_ID
from zetodos_thomas.reconstruct.validate import validate_decode_result
from zetodos_thomas.schemas.decode import (
    AbstentionReason,
    DecodeRequest,
    DecoderEvidenceKind,
    DecodeSpanInput,
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
    EvidenceClass,
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
                )
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
    kinds = [c.evidence[0].kind for c in span.candidates]
    assert DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM in kinds
    assert a.ranking_rules_id == RANKING_RULES_ID
    report = validate_decode_result(a)
    assert report.ok, report.issues


def test_multi_token_construction_and_unknown_abstains() -> None:
    model = _tiny_model(include_loanwords=False)
    ok = decode_request(
        model,
        request_from_surfaces(
            request_id="t2",
            unit_id="demo.1",
            surfaces=["ⲡ", "ϣⲁϫⲉ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
            construction_id="stat-article-ⲡ-ϣⲁϫⲉ--ο-λογος",
        ),
    )
    assert ok.spans[0].status is SpanDecodeStatus.CANDIDATES
    assert ok.spans[0].candidates[0].greek_surfaces == ["ο", "λογος"]
    assert "det_noun_construction" in ok.spans[0].candidates[0].grammatical_notes

    unknown = decode_request(
        model,
        request_from_surfaces(
            request_id="t3",
            unit_id="demo.1",
            surfaces=["ⲡ", "ϣⲁϫⲉ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
            construction_id="stat-does-not-exist",
        ),
    )
    assert unknown.spans[0].status is SpanDecodeStatus.ABSTAIN
    assert (
        unknown.spans[0].abstention_reason is AbstentionReason.CONSTRUCTION_ID_UNKNOWN
    )
    assert unknown.spans[0].candidates == []


def test_abstain_when_unsupported_never_invents_greek() -> None:
    model = _tiny_model(include_loanwords=False)
    result = decode_request(
        model,
        request_from_surfaces(
            request_id="t4",
            unit_id="demo.1",
            surfaces=["ϫϫϫϫ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
        ),
    )
    span = result.spans[0]
    assert span.status is SpanDecodeStatus.ABSTAIN
    assert span.abstention_reason is AbstentionReason.NO_ATTESTED_EVIDENCE
    assert span.candidates == []


def test_loanword_arm_b_vs_arm_a_separation() -> None:
    arm_b = _tiny_model(include_loanwords=True)
    # Arm A indexes the same loanword table for abstention messaging but does
    # not emit independently_verified candidates.
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
    req_b = request_from_surfaces(
        request_id="t5b",
        unit_id="thomas.7",
        surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
        ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    out_b = decode_request(arm_b, req_b)
    assert out_b.spans[0].status is SpanDecodeStatus.CANDIDATES
    top = out_b.spans[0].candidates[0]
    assert top.evidence[0].kind is (DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD)
    assert top.evidence[0].evidence_class is EvidenceClass.INDEPENDENTLY_VERIFIED
    assert top.greek_surfaces == ["μακαριος"]
    assert (
        "inflection" in top.restrictions.lower() or "case" in top.restrictions.lower()
    )

    req_a = request_from_surfaces(
        request_id="t5a",
        unit_id="thomas.7",
        surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
        ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
    )
    out_a = decode_request(arm_a, req_a)
    assert out_a.spans[0].status is SpanDecodeStatus.ABSTAIN
    assert out_a.spans[0].abstention_reason is (
        AbstentionReason.ABLATION_EXCLUDES_LOANWORDS
    )


def test_poxy_query_unit_refused() -> None:
    model = _tiny_model(include_loanwords=True)
    req = DecodeRequest(
        id="t6",
        ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        unit_id="poxy-654-dclp",
        spans=[
            DecodeSpanInput(
                span_id="v1:decode-input/poxy-654-dclp/s/0",
                unit_id="poxy-654-dclp",
                surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            )
        ],
    )
    out = decode_request(model, req)
    assert out.spans[0].abstention_reason is AbstentionReason.FORBIDDEN_QUERY_UNIT


def test_committed_ablation_models_load_and_decode() -> None:
    """Integration: real JAK-37/38 artifacts, keep-both aligners, no LLM."""
    model_b = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    assert model_b.package.adopted_aligner is None
    assert model_b.package.phase2_active_aligners == ["eflomal", "fast_align"]
    assert model_b.includes_verified_loanwords

    # Construction from committed statistical table.
    neg = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-neg",
            unit_id="demo.neg",
            surfaces=["ⲁⲛ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            construction_id="stat-negation-ⲁⲛ--ουχ",
        ),
    )
    assert neg.spans[0].status is SpanDecodeStatus.CANDIDATES
    assert neg.spans[0].candidates[0].greek_surfaces == ["ουχ"]

    # Arm B loanword for Thomas 7 Coptic surface (no P.Oxy. Greek consulted).
    mak = decode_request(
        model_b,
        request_from_surfaces(
            request_id="int-mak",
            unit_id="thomas.7",
            surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        ),
    )
    assert mak.spans[0].status is SpanDecodeStatus.CANDIDATES
    assert mak.spans[0].candidates[0].evidence[0].ref_id == "verified-makarios"
    assert validate_decode_result(mak).ok

    model_a = load_decode_model(REPO_ROOT, condition=AblationCondition.NT_ALIGNER_ONLY)
    assert not model_a.includes_verified_loanwords
    mak_a = decode_request(
        model_a,
        request_from_surfaces(
            request_id="int-mak-a",
            unit_id="thomas.7",
            surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            ablation_condition=AblationCondition.NT_ALIGNER_ONLY,
        ),
    )
    assert mak_a.spans[0].status is SpanDecodeStatus.ABSTAIN
    assert mak_a.spans[0].abstention_reason is (
        AbstentionReason.ABLATION_EXCLUDES_LOANWORDS
    )


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
    assert payload["adopted_aligner"] is None
    assert payload["provenance"]["llm"] is False

    v = runner.invoke(app, ["decode", "validate", str(out)])
    assert v.exit_code == 0, v.output

    rules = runner.invoke(app, ["decode", "ranking-rules"])
    assert rules.exit_code == 0
    assert RANKING_RULES_ID in rules.output

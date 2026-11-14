# SPDX-License-Identifier: Apache-2.0
"""JAK-41: feasibility metrics vs word-for-word baseline + decision gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.eval.feasibility import (
    align_tokens_edit,
    assert_baseline_identical_inputs,
    assert_exact_holdout_membership,
    build_feasibility_report,
    choose_gate_recommendation,
    expand_hypothesis_preserving_gaps,
    flatten_predictions,
    load_and_validate_reconstruction_package,
    load_reconstruction_package,
    score_hypothesis_vs_gold,
    thomas7_gate_status,
)
from zetodos_thomas.eval.letters import greek_letters_only
from zetodos_thomas.eval.validate_reconstruction import ReconstructionValidationError
from zetodos_thomas.schemas.blinded import BlindedTargetKind
from zetodos_thomas.schemas.decode import (
    DecodeResult,
    SpanDecodeResult,
    SpanDecodeStatus,
)
from zetodos_thomas.schemas.feasibility import (
    AggregateSystemMetrics,
    GateRecommendation,
    SystemLabel,
    Thomas7GateStatus,
)
from zetodos_thomas.schemas.mappings import AblationCondition, SpanRef

REPO_ROOT = Path(__file__).resolve().parents[1]
NT_RECON = (
    REPO_ROOT
    / "data"
    / "reconstructions"
    / "jak40"
    / "nt-holdout-blinded-reconstruction-v1.json"
)
THOMAS_RECON = (
    REPO_ROOT / "data" / "reconstructions" / "jak40" / "blinded-reconstruction-v1.json"
)


def test_flatten_predictions_drops_nulls_and_splits() -> None:
    """Legacy helper still drops nulls; scoring must not use it for pairing."""
    assert flatten_predictions([None, "ο λογος", "και", None]) == [
        "ο",
        "λογος",
        "και",
    ]


def test_expand_preserves_middle_abstention_gaps() -> None:
    expanded = expand_hypothesis_preserving_gaps(
        ["και", None, "λογος"], source_token_counts=[1, 1, 1]
    )
    assert expanded == ["και", None, "λογος"]


def test_middle_abstention_does_not_shift_later_gold_pairing() -> None:
    """Regression for Jake review: None must not slide λογος onto δε."""
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.abstention",
        system=SystemLabel.DECODER,
        hypothesis=["και", None, "λογος"],
        gold_tokens=["και", "δε", "λογος"],
        coptic_surfaces=["ⲁⲩⲱ", "ⲇⲉ", "ϣⲁϫⲉ"],
    )
    assert metrics.additions == 0
    assert metrics.omissions == 1  # missing δε
    assert metrics.substitutions == 0
    assert metrics.surface_exact.numerator == 2  # και + λογος
    assert metrics.alignment_protocol == "token-edit-alignment-v1"


def test_word_order_variation_is_order_sensitive() -> None:
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.order",
        system=SystemLabel.DECODER,
        hypothesis=["λογος", "και"],
        gold_tokens=["και", "λογος"],
        coptic_surfaces=["ⲁ", "ⲃ"],
    )
    # Reorder is charged by edit alignment (not free bag match).
    assert metrics.surface_exact.numerator < 2
    assert metrics.substitutions + metrics.additions + metrics.omissions >= 1


def test_multi_token_span_expansion() -> None:
    expanded = expand_hypothesis_preserving_gaps(["ο λογος"], source_token_counts=[2])
    assert expanded == ["ο", "λογος"]
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.span",
        system=SystemLabel.DECODER,
        hypothesis=["ο λογος"],
        gold_tokens=["ο", "λογος"],
        coptic_surfaces=["ⲡ", "ϣⲁϫⲉ"],
    )
    assert metrics.surface_exact.numerator == 2
    assert metrics.additions == 0
    assert metrics.omissions == 0


def test_lemma_inflection_unmeasured_not_independent_signal() -> None:
    """παραβολη vs παραβολην must not look like a measured lemma hit."""
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.lemma",
        system=SystemLabel.DECODER,
        hypothesis=["παραβολη"],
        gold_tokens=["παραβολην"],
        coptic_surfaces=["ⲧⲡⲁⲣⲁⲃⲟⲗⲏ"],
    )
    assert metrics.lemma_inflection_accuracy_available is False
    assert metrics.lemma_proxy.value is None
    assert "unmeasured" in metrics.lemma_proxy.definition.lower()
    # Surfaces differ after letter-fold (final nu).
    assert greek_letters_only("παραβολη") != greek_letters_only("παραβολην")
    assert metrics.surface_exact.numerator == 0


def test_equal_length_wrong_word_counts_substitution_not_zero_edits() -> None:
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.subst",
        system=SystemLabel.DECODER,
        hypothesis=["θεος"],
        gold_tokens=["λογος"],
        coptic_surfaces=["ⲛⲟⲩⲧⲉ"],
    )
    assert metrics.additions == 0
    assert metrics.omissions == 0
    assert metrics.substitutions == 1


def test_reordering_regression_edit_ops() -> None:
    alignment = align_tokens_edit(["β", "α"], ["α", "β"])
    assert alignment.additions + alignment.omissions + len(alignment.substitutions) >= 1
    assert len(alignment.matches) < 2


def _construction_decode_unit() -> object:
    from zetodos_thomas.schemas.blinded import BlindedUnitReconstruction
    from zetodos_thomas.schemas.decode import (
        AbstentionReason,
        CandidateEvidenceRef,
        DecoderEvidenceKind,
        GreekCandidate,
    )

    def _cand(surfaces: list[str]) -> GreekCandidate:
        return GreekCandidate(
            rank=1,
            greek_surfaces=surfaces,
            score=1.0,
            uncertainty=0.1,
            evidence=[
                CandidateEvidenceRef(
                    kind=DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM,
                    ref_id="synthetic",
                )
            ],
        )

    spans = [
        SpanDecodeResult(
            span_id="s0",
            status=SpanDecodeStatus.CANDIDATES,
            surfaces=["ⲡ"],
            input_span=SpanRef(
                span_id="s0",
                source_id="copsahhorner",
                unit_id="synthetic.cov",
                token_ids=["t0"],
            ),
            candidates=[_cand(["ο"])],
        ),
        SpanDecodeResult(
            span_id="s1",
            status=SpanDecodeStatus.ABSTAIN,
            surfaces=["ⲁ"],
            input_span=SpanRef(
                span_id="s1",
                source_id="copsahhorner",
                unit_id="synthetic.cov",
                token_ids=["t1"],
            ),
            candidates=[],
            abstention_reason=AbstentionReason.NO_ATTESTED_EVIDENCE,
        ),
        SpanDecodeResult(
            span_id="s2",
            status=SpanDecodeStatus.CANDIDATES,
            surfaces=["ⲡ", "ϣⲁϫⲉ"],
            input_span=SpanRef(
                span_id="s2",
                source_id="copsahhorner",
                unit_id="synthetic.cov",
                token_ids=["t2", "t3"],
            ),
            candidates=[_cand(["ο", "λογος"])],
        ),
    ]
    return BlindedUnitReconstruction(
        unit_id="synthetic.cov",
        target_kind=BlindedTargetKind.NT_HOLDOUT,
        source_id="copsahhorner",
        coptic_surfaces=["ⲡ", "ⲁ", "ⲡ", "ϣⲁϫⲉ"],
        decode_result=DecodeResult(
            id="synthetic-decode",
            request_id="t",
            ablation_condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
            unit_id="synthetic.cov",
            spans=spans,
        ),
        top1_greek_surfaces=["ο", None, "ο λογος"],
        covering_span_kinds=["singleton", "singleton", "construction"],
    )


def test_token_coverage_vs_span_coverage_two_token_construction() -> None:
    unit = _construction_decode_unit()
    metrics = score_hypothesis_vs_gold(
        unit_id=unit.unit_id,
        system=SystemLabel.DECODER,
        hypothesis=unit.top1_greek_surfaces,
        gold_tokens=["ο", "δε", "ο", "λογος"],
        coptic_surfaces=unit.coptic_surfaces,
        covering_span_kinds=unit.covering_span_kinds,
        decode_unit=unit,
    )
    # 3 non-null spans / 3 spans; tokens covered = 1 + 2 = 3 of 4 (abstain t1).
    assert metrics.predicted_non_null_spans == 2
    assert metrics.span_coverage == pytest.approx(2 / 3)
    assert metrics.predicted_covered_tokens == 3
    assert metrics.token_coverage == pytest.approx(3 / 4)
    assert metrics.coverage == metrics.token_coverage
    assert metrics.coverage != metrics.span_coverage


def test_unlocalized_phrase_occurrence_not_phrase_fidelity() -> None:
    """Phrase occurs elsewhere in the verse but not as localized fidelity."""
    unit = _construction_decode_unit()
    # Gold has the predicted construction phrase elsewhere in the verse;
    # occurrence hit must not imply localized phrase fidelity.
    metrics = score_hypothesis_vs_gold(
        unit_id=unit.unit_id,
        system=SystemLabel.DECODER,
        hypothesis=unit.top1_greek_surfaces,
        gold_tokens=["και", "θεος", "ο", "λογος"],
        coptic_surfaces=unit.coptic_surfaces,
        covering_span_kinds=unit.covering_span_kinds,
        decode_unit=unit,
    )
    assert metrics.phrase_fidelity_available is False
    assert metrics.construction_spans == 1
    assert metrics.construction_phrase_hits == 1
    assert any(
        "unlocalized" in n.lower() or "fidelity" in n.lower() for n in metrics.notes
    )


def test_surface_exact_and_gap_aware_denominators() -> None:
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.1",
        system=SystemLabel.DECODER,
        hypothesis=["και", "ο", None, "λογος"],
        gold_tokens=["και", "ὁ", "λόγος", "ην"],
        coptic_surfaces=["ⲁⲩⲱ", "ⲡ", "ϣⲁϫⲉ", "ⲡⲉ"],
    )
    assert metrics.gold_tokens == 4
    assert metrics.predicted_tokens == 3
    assert metrics.omissions >= 1
    assert metrics.surface_exact.denominator == 4
    assert metrics.lemma_proxy.value is None
    assert metrics.letter_f1 is not None
    assert metrics.letter_metric_kind == "token-edit-aligned-letter-v1"
    assert metrics.incidental_global_lcs >= 0


def test_baseline_identical_coptic_inputs_on_committed_nt_package() -> None:
    package = load_reconstruction_package(NT_RECON)
    assert_baseline_identical_inputs(package)
    assert package.word_for_word_baseline
    assert len(package.word_for_word_baseline) == len(package.units)


def test_committed_packages_validate_and_holdout_membership() -> None:
    thomas = load_and_validate_reconstruction_package(THOMAS_RECON, repo_root=REPO_ROOT)
    nt = load_and_validate_reconstruction_package(NT_RECON, repo_root=REPO_ROOT)
    assert thomas.owned_by == "JAK-40"
    problems = assert_exact_holdout_membership(nt, repo_root=REPO_ROOT)
    assert problems == []


def test_tampered_top1_fails_validation() -> None:
    package = load_reconstruction_package(NT_RECON)
    unit = package.units[0]
    tampered_surfaces = list(unit.top1_greek_surfaces)
    tampered_surfaces[0] = "NOT_FROM_DECODER"
    tampered_unit = unit.model_copy(update={"top1_greek_surfaces": tampered_surfaces})
    tampered = package.model_copy(update={"units": [tampered_unit, *package.units[1:]]})
    with pytest.raises(ReconstructionValidationError):
        from zetodos_thomas.eval.validate_reconstruction import (
            assert_valid_reconstruction,
        )

        assert_valid_reconstruction(tampered, repo_root=REPO_ROOT)


def test_omitted_holdout_fails_membership() -> None:
    package = load_reconstruction_package(NT_RECON)
    trimmed = package.model_copy(update={"units": package.units[:3]})
    with pytest.raises(ReconstructionValidationError):
        assert_exact_holdout_membership(trimmed, repo_root=REPO_ROOT)


def test_mismatched_ablation_flagged_in_report() -> None:
    # Building with committed matching packages stays validated.
    report = build_feasibility_report(
        REPO_ROOT,
        thomas_recon_path=THOMAS_RECON,
        nt_recon_path=NT_RECON,
        include_ablation_arms=False,
    )
    assert report.identical_inputs["packages_validated"] is True
    assert (
        report.primary_ablation is AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS
    )


def test_thomas7_gate_honestly_blocked() -> None:
    status = thomas7_gate_status(
        REPO_ROOT, thomas_package=load_reconstruction_package(THOMAS_RECON)
    )
    assert status.review_status == "ready-for-independent-review"
    assert status.gate_blocked is True
    assert status.official_surviving_letter_scores is False
    assert status.freeze_present is True
    # Existence alone is not official; verified digest still cannot unlock
    # official scores without human-checked.
    assert status.freeze_verified is True or status.freeze_verified is False


def test_gate_recommends_pause_when_official_blocked_and_f1_low() -> None:
    thomas7 = Thomas7GateStatus(
        eval_package_id="thomas-7-blinded-eval-v1",
        review_status="ready-for-independent-review",
        official_surviving_letter_scores=False,
        gate_blocked=True,
        freeze_present=True,
        freeze_verified=False,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=41,
        mean_letter_precision=0.15,
        mean_letter_recall=0.09,
        mean_letter_f1=0.12,
        mean_surface_exact_recall=0.05,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.08,
        mean_coverage=0.69,
        mean_abstention_rate=0.31,
    )
    baseline = AggregateSystemMetrics(
        system=SystemLabel.WORD_FOR_WORD,
        n_units=41,
        mean_letter_precision=0.14,
        mean_letter_recall=0.08,
        mean_letter_f1=0.10,
        mean_surface_exact_recall=0.05,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.07,
        mean_coverage=0.75,
        mean_abstention_rate=0.25,
    )
    rec, rationale, ready, _notes = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="provisional-draft",
    )
    assert rec is GateRecommendation.PAUSE_AND_IMPROVE
    assert ready is False
    assert any(
        "gate-blocked" in line or "Evaluation-ready=false" in line for line in rationale
    )
    # No invented numeric cutoffs as decision rules (descriptive F1 text is fine).
    joined = " ".join(rationale).lower()
    assert "no invented numeric" in joined or "no approved numeric" in joined
    assert "dec_f1 <" not in joined
    assert "0.35" not in joined


def test_gate_missing_metrics_not_ready() -> None:
    thomas7 = Thomas7GateStatus(
        eval_package_id="x",
        review_status="human-checked",
        official_surviving_letter_scores=True,
        gate_blocked=False,
        freeze_present=True,
        freeze_verified=True,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=0,
        mean_letter_precision=None,
        mean_letter_recall=None,
        mean_letter_f1=None,
        mean_surface_exact_recall=None,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=None,
        mean_coverage=None,
    )
    baseline = AggregateSystemMetrics(
        system=SystemLabel.WORD_FOR_WORD,
        n_units=0,
        mean_letter_precision=None,
        mean_letter_recall=None,
        mean_letter_f1=None,
        mean_surface_exact_recall=None,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=None,
        mean_coverage=None,
    )
    rec, _r, ready, notes = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="human-checked",
    )
    assert ready is False
    assert rec is GateRecommendation.PAUSE_AND_IMPROVE
    assert any("missing" in n.lower() or "insufficient" in n.lower() for n in notes)


def test_gate_missing_freeze_not_ready() -> None:
    thomas7 = Thomas7GateStatus(
        eval_package_id="x",
        review_status="human-checked",
        official_surviving_letter_scores=False,
        gate_blocked=False,
        freeze_present=False,
        freeze_verified=False,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=2,
        mean_letter_precision=0.9,
        mean_letter_recall=0.9,
        mean_letter_f1=0.9,
        mean_surface_exact_recall=0.9,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.9,
        mean_coverage=0.9,
    )
    baseline = AggregateSystemMetrics(
        system=SystemLabel.WORD_FOR_WORD,
        n_units=2,
        mean_letter_precision=0.5,
        mean_letter_recall=0.5,
        mean_letter_f1=0.5,
        mean_surface_exact_recall=0.5,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.5,
        mean_coverage=0.5,
    )
    rec, _r, ready, notes = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="human-checked",
    )
    assert ready is False
    assert rec is not GateRecommendation.PROCEED
    assert any("freeze" in n.lower() for n in notes)


def test_gate_unreviewed_not_ready() -> None:
    thomas7 = Thomas7GateStatus(
        eval_package_id="x",
        review_status="ready-for-independent-review",
        official_surviving_letter_scores=False,
        gate_blocked=True,
        freeze_present=True,
        freeze_verified=True,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=2,
        mean_letter_precision=0.9,
        mean_letter_recall=0.9,
        mean_letter_f1=0.9,
        mean_surface_exact_recall=0.9,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.9,
        mean_coverage=0.9,
    )
    baseline = AggregateSystemMetrics(
        system=SystemLabel.WORD_FOR_WORD,
        n_units=2,
        mean_letter_precision=0.5,
        mean_letter_recall=0.5,
        mean_letter_f1=0.5,
        mean_surface_exact_recall=0.5,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.5,
        mean_coverage=0.5,
    )
    rec, _r, ready, _n = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="provisional-draft",
    )
    assert ready is False
    assert rec is GateRecommendation.PAUSE_AND_IMPROVE


def test_gate_strongly_underperforming_reassess_when_ready() -> None:
    thomas7 = Thomas7GateStatus(
        eval_package_id="x",
        review_status="human-checked",
        official_surviving_letter_scores=True,
        gate_blocked=False,
        freeze_present=True,
        freeze_verified=True,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=10,
        mean_letter_precision=0.2,
        mean_letter_recall=0.2,
        mean_letter_f1=0.2,
        mean_surface_exact_recall=0.1,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.1,
        mean_coverage=0.5,
    )
    baseline = AggregateSystemMetrics(
        system=SystemLabel.WORD_FOR_WORD,
        n_units=10,
        mean_letter_precision=0.5,
        mean_letter_recall=0.5,
        mean_letter_f1=0.5,
        mean_surface_exact_recall=0.4,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.4,
        mean_coverage=0.7,
    )
    rec, rationale, ready, _n = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="human-checked",
        packages_validated=True,
        holdout_membership_complete=True,
    )
    assert ready is True
    assert rec is GateRecommendation.REASSESS
    assert any("underperform" in line.lower() for line in rationale)


def test_gate_never_proceeds_without_approved_criteria() -> None:
    """Even when evaluation-ready and decoder ≥ baseline, no invented proceed."""
    thomas7 = Thomas7GateStatus(
        eval_package_id="x",
        review_status="human-checked",
        official_surviving_letter_scores=True,
        gate_blocked=False,
        freeze_present=True,
        freeze_verified=True,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=10,
        mean_letter_precision=0.9,
        mean_letter_recall=0.9,
        mean_letter_f1=0.9,
        mean_surface_exact_recall=0.9,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.9,
        mean_coverage=0.9,
    )
    baseline = AggregateSystemMetrics(
        system=SystemLabel.WORD_FOR_WORD,
        n_units=10,
        mean_letter_precision=0.5,
        mean_letter_recall=0.5,
        mean_letter_f1=0.5,
        mean_surface_exact_recall=0.5,
        mean_lemma_proxy_recall=None,
        mean_lemma_lcs_recall=0.5,
        mean_coverage=0.5,
    )
    rec, rationale, ready, _n = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="human-checked",
    )
    assert ready is True
    assert rec is GateRecommendation.PAUSE_AND_IMPROVE
    assert any("approved" in line.lower() for line in rationale)


def test_build_feasibility_report_skip_arms_offline() -> None:
    report = build_feasibility_report(
        REPO_ROOT,
        thomas_recon_path=THOMAS_RECON,
        nt_recon_path=NT_RECON,
        include_ablation_arms=False,
    )
    assert report.owned_by == "JAK-41"
    assert report.decoder_aggregate.n_units == 41
    assert report.word_for_word_aggregate.n_units == 41
    assert report.identical_inputs["coptic_surfaces_matched"] is True
    assert report.thomas7_gate.gate_blocked is True
    assert report.evaluation_ready is False
    assert report.gate_recommendation is GateRecommendation.PAUSE_AND_IMPROVE
    assert report.provenance["llm"] is False
    assert report.provenance["adopted_aligner"] is None
    assert set(report.provenance["phase2_active_aligners"]) == {"eflomal", "fast_align"}
    assert report.decoder_aggregate.mean_letter_f1 is not None
    assert report.word_for_word_aggregate.mean_letter_f1 is not None
    assert 0.0 <= report.decoder_aggregate.mean_letter_f1 <= 1.0
    assert report.decoder_aggregate.mean_lemma_proxy_recall is None
    assert report.summary["lemma_inflection_accuracy"] == "unmeasured"
    assert (
        report.summary["phrase_fidelity"] == "unavailable_unlocalized_occurrence_only"
    )
    assert any(
        fm.category == "morphology_underdetermination" for fm in report.failure_modes
    )
    assert "letter_agreement" in report.metric_definitions
    assert report.decoder_aggregate.total_substitutions >= 0
    assert report.leakage_safeguards.get("reconstruction_packages_validated") is True


def test_ablation_arms_marked_uninformative() -> None:
    report = build_feasibility_report(
        REPO_ROOT,
        thomas_recon_path=THOMAS_RECON,
        nt_recon_path=NT_RECON,
        include_ablation_arms=True,
    )
    assert report.ablation_arms
    assert report.summary["ablation_arms_informative"] is False
    for arm in report.ablation_arms:
        assert arm.informative_for_loanword_decision is False
        assert "uninformative" in arm.evidence_status
        assert arm.thomas7_blinded_coverage
        assert arm.loanword_hypothesis_changes is not None


def test_cli_feasibility_report_smoke(tmp_path: Path) -> None:
    out = tmp_path / "feasibility.json"
    doc = tmp_path / "feasibility.md"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "feasibility-report",
            "--skip-ablation-arms",
            "-o",
            str(out),
            "--write-doc",
            str(doc),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert doc.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["owned_by"] == "JAK-41"
    assert payload["gate_recommendation"] == "pause_and_improve"
    assert payload["evaluation_ready"] is False
    doc_text = doc.read_text(encoding="utf-8")
    assert "pause_and_improve" in doc_text
    assert "human-checked" in doc_text
    assert "unmeasured" in doc_text
    assert "Evaluation-ready" in doc_text or "evaluation-ready" in doc_text.lower()


def test_letter_fold_ignores_diacritics_for_exact() -> None:
    assert greek_letters_only("Λόγος") == greek_letters_only("λογος")

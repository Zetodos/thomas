# SPDX-License-Identifier: Apache-2.0
"""JAK-41: feasibility metrics vs word-for-word baseline + decision gate."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.eval.feasibility import (
    assert_baseline_identical_inputs,
    build_feasibility_report,
    choose_gate_recommendation,
    flatten_predictions,
    load_reconstruction_package,
    score_hypothesis_vs_gold,
    thomas7_gate_status,
)
from zetodos_thomas.eval.letters import greek_letters_only
from zetodos_thomas.schemas.feasibility import (
    AggregateSystemMetrics,
    GateRecommendation,
    SystemLabel,
    Thomas7GateStatus,
)

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
    assert flatten_predictions([None, "ο λογος", "και", None]) == [
        "ο",
        "λογος",
        "και",
    ]


def test_surface_exact_and_lemma_proxy_denominators() -> None:
    metrics = score_hypothesis_vs_gold(
        unit_id="synthetic.1",
        system=SystemLabel.DECODER,
        hypothesis=["και", "ο", None, "λογος"],
        gold_tokens=["και", "ὁ", "λόγος", "ην"],
        coptic_surfaces=["ⲁⲩⲱ", "ⲡ", "ϣⲁϫⲉ", "ⲡⲉ"],
    )
    assert metrics.gold_tokens == 4
    assert metrics.predicted_tokens == 3
    assert metrics.omissions == 1
    assert metrics.additions == 0
    # Positional exact after letter-fold; unpaired gold counts as omission.
    assert metrics.surface_exact.denominator == 4
    assert metrics.surface_exact.numerator >= 1
    assert metrics.lemma_proxy.denominator == 4
    assert metrics.letter_f1 is not None
    assert metrics.letter_metric_kind == "token-positional-letter-v1"
    assert metrics.incidental_global_lcs >= metrics.letter_true_positives


def test_baseline_identical_coptic_inputs_on_committed_nt_package() -> None:
    package = load_reconstruction_package(NT_RECON)
    assert_baseline_identical_inputs(package)
    assert package.word_for_word_baseline
    assert len(package.word_for_word_baseline) == len(package.units)


def test_thomas7_gate_honestly_blocked() -> None:
    status = thomas7_gate_status(REPO_ROOT)
    assert status.review_status == "ready-for-independent-review"
    assert status.gate_blocked is True
    assert status.official_surviving_letter_scores is False
    assert status.freeze_present is True


def test_gate_recommends_pause_when_official_blocked_and_f1_low() -> None:
    thomas7 = Thomas7GateStatus(
        eval_package_id="thomas-7-blinded-eval-v1",
        review_status="ready-for-independent-review",
        official_surviving_letter_scores=False,
        gate_blocked=True,
        freeze_present=True,
    )
    decoder = AggregateSystemMetrics(
        system=SystemLabel.DECODER,
        n_units=41,
        mean_letter_precision=0.15,
        mean_letter_recall=0.09,
        mean_letter_f1=0.12,
        mean_surface_exact_recall=0.05,
        mean_lemma_proxy_recall=0.05,
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
        mean_lemma_proxy_recall=0.05,
        mean_lemma_lcs_recall=0.07,
        mean_coverage=0.75,
        mean_abstention_rate=0.25,
    )
    rec, rationale = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder,
        baseline=baseline,
        nt_review_status="provisional-draft",
    )
    assert rec is GateRecommendation.PAUSE_AND_IMPROVE
    assert any("gate-blocked" in line for line in rationale)


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
    assert report.gate_recommendation is GateRecommendation.PAUSE_AND_IMPROVE
    assert report.provenance["llm"] is False
    assert report.provenance["adopted_aligner"] is None
    assert set(report.provenance["phase2_active_aligners"]) == {"eflomal", "fast_align"}
    assert report.decoder_aggregate.mean_letter_f1 is not None
    assert report.word_for_word_aggregate.mean_letter_f1 is not None
    # Decoder should be in the same ballpark as baseline (no invented threshold).
    assert 0.0 <= report.decoder_aggregate.mean_letter_f1 <= 1.0
    assert any(
        fm.category == "morphology_underdetermination" for fm in report.failure_modes
    )
    assert "letter_agreement" in report.metric_definitions


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
    assert "pause_and_improve" in doc.read_text(encoding="utf-8")
    assert "human-checked" in doc.read_text(encoding="utf-8")


def test_letter_fold_ignores_diacritics_for_exact() -> None:
    assert greek_letters_only("Λόγος") == greek_letters_only("λογος")

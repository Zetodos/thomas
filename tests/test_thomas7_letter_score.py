# SPDX-License-Identifier: Apache-2.0
"""Restored JAK-40 Thomas 7 surviving-letter freeze, kept off the v3 table."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.cli.eval_cmd import eval_run
from zetodos_thomas.cli.reconstruct_cmd import reconstruct_run
from zetodos_thomas.eval.review_gate import (
    HUMAN_CHECKED,
    READY_FOR_INDEPENDENT_REVIEW,
    EvalReviewGateError,
)
from zetodos_thomas.eval.surviving_letters import (
    DEFAULT_EVAL_PACKAGE_REL,
    DEFAULT_FREEZE_REL,
    DEFAULT_RECON_REL,
    DEFAULT_SCORE_REL,
    DEFAULT_THOMAS7_LINE_IDS,
    BlindedScoreIntegrityError,
    assert_restorations_excluded_from_gold,
    load_freeze_record,
    load_reconstruction_package,
    package_content_sha256,
    score_thomas7_surviving,
    try_score_thomas7_or_gate_report,
    verify_official_freeze,
)
from zetodos_thomas.schemas.feasibility import (
    FeasibilityReport,
    GateRecommendation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_BENCHMARK = REPO_ROOT / "docs" / "benchmark-report-v3.md"
LETTER_SCORE_DOC = REPO_ROOT / "docs" / "thomas-7-surviving-letter-score.md"
GATE_METRICS = (
    REPO_ROOT / "data" / "reconstructions" / "jak41" / "feasibility-metrics-v1.json"
)
EXPLORATORY_SPAN = "thomas.7.poxy654.a"


def _option_default(fn: Callable[..., object], name: str) -> object:
    default = inspect.signature(fn).parameters[name].default
    return getattr(default, "default", default)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_cli_defaults_stay_on_model_v1() -> None:
    assert _option_default(eval_run, "model") == "v1"
    assert _option_default(reconstruct_run, "model") == "v1"
    assert (REPO_ROOT / "data" / "mappings" / "model-v2").is_dir()


def test_freeze_matches_recovered_reconstruction() -> None:
    package = load_reconstruction_package(REPO_ROOT / DEFAULT_RECON_REL)
    freeze = load_freeze_record(REPO_ROOT / DEFAULT_FREEZE_REL)
    verify_official_freeze(package, freeze)
    assert freeze.sealed_before_fragment_load is True
    assert freeze.unit_id == "thomas.7"
    assert package_content_sha256(package) == freeze.reconstruction_sha256
    assert package.blinding["greek_consulted_during_reconstruction"] is False
    assert "decode-input" not in (REPO_ROOT / DEFAULT_RECON_REL).read_text(
        encoding="utf-8"
    )


def test_tampered_prediction_fails_freeze() -> None:
    package = load_reconstruction_package(REPO_ROOT / DEFAULT_RECON_REL)
    freeze = load_freeze_record(REPO_ROOT / DEFAULT_FREEZE_REL)
    unit = package.units[0]
    tampered_top1 = list(unit.top1_greek_surfaces)
    assert tampered_top1
    tampered_top1[0] = "ταμπερ"
    tampered = package.model_copy(
        update={
            "units": [
                unit.model_copy(update={"top1_greek_surfaces": tampered_top1}),
                *package.units[1:],
            ]
        }
    )
    assert package_content_sha256(tampered) != freeze.reconstruction_sha256
    with pytest.raises(BlindedScoreIntegrityError, match="digest"):
        verify_official_freeze(tampered, freeze)


def test_eval_package_is_not_human_checked() -> None:
    pack = _load(REPO_ROOT / DEFAULT_EVAL_PACKAGE_REL)
    assert pack["review_status"] == READY_FOR_INDEPENDENT_REVIEW
    assert pack["review_status"] != HUMAN_CHECKED
    assert pack["id"] == "thomas-7-blinded-eval-v1"


def test_committed_score_is_gate_blocked_and_separate_from_v3() -> None:
    package = load_reconstruction_package(REPO_ROOT / DEFAULT_RECON_REL)
    report = try_score_thomas7_or_gate_report(REPO_ROOT, package)
    committed = _load(REPO_ROOT / DEFAULT_SCORE_REL)
    assert report.model_dump(mode="json") == committed
    assert committed["official"] is False
    assert committed["freeze_verified"] is False
    assert committed["gold"] is None
    assert committed["scores"] == []
    assert committed["review_status"] == READY_FOR_INDEPENDENT_REVIEW
    blinding = committed["blinding"]
    assert isinstance(blinding, dict)
    assert blinding["gate_blocked"] is True
    assert blinding["greek_loaded_at_scoring"] is False
    v3 = V3_BENCHMARK.read_text(encoding="utf-8")
    assert "jak40-thomas7" not in v3
    assert EXPLORATORY_SPAN in v3
    doc = LETTER_SCORE_DOC.read_text(encoding="utf-8")
    assert "official: false" in doc
    assert EXPLORATORY_SPAN in doc
    assert "benchmark-report-v3.md" in doc


def test_gate_report_does_not_open_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    def _opened(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fragment corpus opened during gate report")

    monkeypatch.setattr(
        "zetodos_thomas.eval.surviving_letters.load_fragments_at_eval_time",
        _opened,
    )
    package = load_reconstruction_package(REPO_ROOT / DEFAULT_RECON_REL)
    blocked = try_score_thomas7_or_gate_report(REPO_ROOT, package)
    assert blocked.official is False
    assert blocked.gold is None
    assert blocked.scores == []


def test_official_scoring_refuses_unreviewed_package() -> None:
    package = load_reconstruction_package(REPO_ROOT / DEFAULT_RECON_REL)
    with pytest.raises(EvalReviewGateError, match="human-checked"):
        score_thomas7_surviving(
            REPO_ROOT,
            package,
            require_human_checked=True,
        )


def test_unofficial_smoke_does_not_replace_the_score() -> None:
    """Surviving-letter smoke may open fragments at score time only.

    It must stay unofficial and must not rewrite the frozen predictions or
    the committed gate-blocked score.
    """
    before_recon = (REPO_ROOT / DEFAULT_RECON_REL).read_bytes()
    before_score = (REPO_ROOT / DEFAULT_SCORE_REL).read_bytes()
    before_pack = (REPO_ROOT / DEFAULT_EVAL_PACKAGE_REL).read_bytes()
    package = load_reconstruction_package(REPO_ROOT / DEFAULT_RECON_REL)
    report = score_thomas7_surviving(
        REPO_ROOT,
        package,
        require_human_checked=False,
    )
    assert report.official is False
    assert report.review_status == READY_FOR_INDEPENDENT_REVIEW
    assert report.gold is not None
    assert report.gold.line_ids == list(DEFAULT_THOMAS7_LINE_IDS)
    assert_restorations_excluded_from_gold(report.gold)
    assert report.scores
    assert report.scores[0].official is False
    assert report.scores[0].metric_kind == "gap-aware-ordered-runs-v1"
    assert report.scores[0].gold_denominator > 0
    assert len(report.gold.restoration_letters_excluded) > 0
    assert report.blinding["greek_loaded_at_scoring"] is True
    assert report.blinding["restorations_in_gold_denominator"] is False
    assert (REPO_ROOT / DEFAULT_RECON_REL).read_bytes() == before_recon
    assert (REPO_ROOT / DEFAULT_SCORE_REL).read_bytes() == before_score
    assert (REPO_ROOT / DEFAULT_EVAL_PACKAGE_REL).read_bytes() == before_pack


def test_recovered_jak41_gate_stays_pause_and_blocked() -> None:
    report = FeasibilityReport.model_validate(_load(GATE_METRICS))
    assert report.gate_recommendation is GateRecommendation.PAUSE_AND_IMPROVE
    assert report.evaluation_ready is False
    assert report.thomas7_gate.official_surviving_letter_scores is False
    assert report.thomas7_gate.review_status == READY_FOR_INDEPENDENT_REVIEW
    assert report.thomas7_gate.gate_blocked is True
    assert report.thomas7_gate.freeze_present is True
    assert report.thomas7_gate.freeze_verified is True
    assert report.thomas7_gate.human_checked_required is True


def test_cli_gate_report_and_fail_closed(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "score.json"
    gated = runner.invoke(
        app,
        [
            "eval",
            "score-thomas7",
            "--repo-root",
            str(REPO_ROOT),
            "--gate-report",
            "-o",
            str(out),
        ],
    )
    assert gated.exit_code == 0, gated.output
    written = _load(out)
    committed = _load(REPO_ROOT / DEFAULT_SCORE_REL)
    assert written == committed
    closed = runner.invoke(
        app,
        [
            "eval",
            "score-thomas7",
            "--repo-root",
            str(REPO_ROOT),
            "--fail-closed",
            "-o",
            str(tmp_path / "closed.json"),
        ],
    )
    assert closed.exit_code == 1
    closed_text = closed.output + (getattr(closed, "stderr", None) or "")
    assert "human-checked" in closed_text

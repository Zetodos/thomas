# SPDX-License-Identifier: Apache-2.0
"""JAK-40: blinded reconstruction + surviving-letter eval harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.align.lexical_baseline import LexicalBaselineModel
from zetodos_thomas.cli.app import app
from zetodos_thomas.eval.blinded_reconstruct import (
    BlindedReconstructError,
    assert_no_fragment_greek_in_scope,
    coptic_surfaces_from_normalized,
    reconstruct_blinded_package,
    reconstruct_unit,
)
from zetodos_thomas.eval.letters import greek_letters_only, letter_agreement
from zetodos_thomas.eval.review_gate import EvalReviewGateError
from zetodos_thomas.eval.surviving_letters import (
    DEFAULT_THOMAS7_LINE_IDS,
    assert_restorations_excluded_from_gold,
    extract_surviving_letter_gold,
    load_fragments_at_eval_time,
    score_thomas7_surviving,
    transcription_by_id,
    try_score_thomas7_or_gate_report,
)
from zetodos_thomas.eval.word_for_word import fit_argmax_table, predict_surfaces
from zetodos_thomas.reconstruct.model import load_decode_model
from zetodos_thomas.schemas.blinded import BlindedTargetKind
from zetodos_thomas.schemas.fragment import (
    FragmentLayerKind,
    FragmentLine,
    FragmentToken,
)
from zetodos_thomas.schemas.mappings import AblationCondition

REPO_ROOT = Path(__file__).resolve().parents[1]
BLIND_PATH = REPO_ROOT / "data" / "evidence" / "eval" / "thomas-7-blinded-v1.json"
RECON_DIR = REPO_ROOT / "data" / "reconstructions" / "jak40"


def test_letter_agreement_ignores_accents_and_punct() -> None:
    metrics = letter_agreement("Μακάριος ἐστίν!", "καρι εστιν")
    assert metrics.gold_denominator == len(greek_letters_only("καρι εστιν"))
    assert metrics.true_positives >= 8
    assert metrics.precision is not None


def test_restorations_excluded_from_synthetic_gold() -> None:
    from zetodos_thomas.schemas.fragment import FragmentTranscription

    line = FragmentLine(
        line_id="main.40",
        n="40",
        tokens=[
            FragmentToken(kind=FragmentLayerKind.RESTORATION, text="μα"),
            FragmentToken(kind=FragmentLayerKind.SURVIVING, text="κ"),
            FragmentToken(kind=FragmentLayerKind.SURVIVING, text="άρι"),
            FragmentToken(kind=FragmentLayerKind.RESTORATION, text="ός"),
        ],
    )
    tx = FragmentTranscription(
        id="fixture-654",
        source_id="dclp-poxy-654",
        manuscript_id="P.Oxy. 654",
        lines=[line],
    )
    gold = extract_surviving_letter_gold(tx, line_ids=["main.40"])
    assert_restorations_excluded_from_gold(gold)
    assert gold.surviving_letters == greek_letters_only("κάρι")
    assert gold.surviving_letters != greek_letters_only("μακάριός")
    # Restored prefix "μα" must not be required as part of surviving gold.
    assert not gold.surviving_letters.startswith(greek_letters_only("μα"))


def test_fragment_path_rejected_on_reconstruct_api() -> None:
    with pytest.raises(BlindedReconstructError, match=r"P\.Oxy"):
        assert_no_fragment_greek_in_scope(["data/sources/fragments/poxy-654.json"])


def test_blinded_thomas7_reconstruct_no_fragment_greek() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=False,
        include_word_for_word=True,
    )
    assert package.owned_by == "JAK-40"
    assert package.blinding["greek_consulted_during_reconstruction"] is False
    assert package.blinding["fragment_corpus_opened"] is False
    assert package.adopted_aligner is None
    assert set(package.phase2_active_aligners) == {"eflomal", "fast_align"}
    assert package.provenance["llm"] is False
    assert len(package.units) == 1
    unit = package.units[0]
    assert unit.unit_id == "thomas.7"
    assert unit.target_kind is BlindedTargetKind.THOMAS_PILOT
    assert "ⲙⲁⲕⲁⲣⲓⲟⲥ" in unit.coptic_surfaces
    # makarios loanword clue should surface under arm B
    want = greek_letters_only("μακαριος")
    assert any(
        s is not None and want in greek_letters_only(s)
        for s in unit.top1_greek_surfaces
    )
    blob = json.dumps(package.model_dump(mode="json"), ensure_ascii=False)
    assert "ἔσται. μα" not in blob
    assert package.word_for_word_baseline
    assert package.word_for_word_baseline[0].method == "train-cooccurrence-argmax-v1"


def test_nt_holdout_partition_only() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=True,
        include_word_for_word=False,
    )
    holdouts = [
        u for u in package.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    ]
    assert package.summary["nt_holdout_units"] == len(holdouts)
    assert len(holdouts) >= 30  # v3 has 41 nonempty; allow minor empties skip
    # No train-looking John 1 in holdout reconstruct set
    assert not any(u.unit_id.startswith("john.1.") for u in holdouts)
    assert not any(u.unit_id.startswith("john.8.") for u in holdouts)


def test_score_thomas7_gate_blocks_unreviewed() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=False,
        include_word_for_word=False,
    )
    with pytest.raises(EvalReviewGateError, match="human-checked"):
        score_thomas7_surviving(
            REPO_ROOT,
            package,
            require_human_checked=True,
        )
    blocked = try_score_thomas7_or_gate_report(REPO_ROOT, package)
    assert blocked.official is False
    assert blocked.blinding.get("gate_blocked") is True
    assert blocked.gold is None


def test_unofficial_surviving_letter_smoke_loads_fragment_only_at_score() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=False,
        include_word_for_word=False,
    )
    assert package.blinding["fragment_corpus_opened"] is False
    report = score_thomas7_surviving(
        REPO_ROOT,
        package,
        require_human_checked=False,
    )
    assert report.official is False
    assert report.gold is not None
    assert report.gold.line_ids == list(DEFAULT_THOMAS7_LINE_IDS)
    assert_restorations_excluded_from_gold(report.gold)
    assert report.scores and report.scores[0].gold_denominator > 0
    # Restorations longer than surviving on these lines — excluded count > 0
    assert len(report.gold.restoration_letters_excluded) > 0
    assert report.blinding["greek_loaded_at_scoring"] is True
    assert report.blinding["restorations_in_gold_denominator"] is False


def test_poxy_surviving_extract_matches_corpus_layers() -> None:
    corpus = load_fragments_at_eval_time(REPO_ROOT)
    tx = transcription_by_id(corpus, "poxy-654-dclp")
    gold = extract_surviving_letter_gold(tx, line_ids=DEFAULT_THOMAS7_LINE_IDS)
    # Hand-check main.40 surviving includes κ + άρι + space-bearing ἐστιν
    line40 = next(s for s in gold.spans if s.line_id == "main.40")
    assert "κ" in line40.surviving_text
    assert "άρι" in line40.surviving_text
    assert "μα" not in line40.surviving_text  # restoration prefix


def test_word_for_word_argmax_blind() -> None:
    model = LexicalBaselineModel()
    model.pair_counts[("ⲣⲱⲙⲉ", "ανθρωπος")] = 5
    model.pair_counts[("ⲣⲱⲙⲉ", "τις")] = 1
    model.left_counts["ⲣⲱⲙⲉ"] = 5
    model.right_counts["ανθρωπος"] = 5
    model.right_counts["τις"] = 1
    table = fit_argmax_table(model)
    assert predict_surfaces(table, ["ⲣⲱⲙⲉ", "xyz"]) == ["ανθρωπος", None]


def test_cli_reconstruct_thomas_only(tmp_path: Path) -> None:
    out = tmp_path / "recon.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "reconstruct",
            "--repo-root",
            str(REPO_ROOT),
            "--thomas-only",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    package = json.loads(out.read_text(encoding="utf-8"))
    assert package["owned_by"] == "JAK-40"
    assert package["summary"]["thomas_units"] == 1
    assert package["blinding"]["greek_consulted_during_reconstruction"] is False
    val = runner.invoke(app, ["eval", "validate-reconstruction", str(out)])
    assert val.exit_code == 0, val.output


def test_cli_score_gate_report(tmp_path: Path) -> None:
    recon = tmp_path / "recon.json"
    score = tmp_path / "score.json"
    runner = CliRunner()
    build = runner.invoke(
        app,
        [
            "eval",
            "reconstruct",
            "--repo-root",
            str(REPO_ROOT),
            "--thomas-only",
            "-o",
            str(recon),
        ],
    )
    assert build.exit_code == 0, build.output
    gated = runner.invoke(
        app,
        [
            "eval",
            "score-thomas7",
            "--repo-root",
            str(REPO_ROOT),
            "--gate-report",
            "-r",
            str(recon),
            "-o",
            str(score),
        ],
    )
    assert gated.exit_code == 0, gated.output
    report = json.loads(score.read_text(encoding="utf-8"))
    assert report["official"] is False
    assert report["blinding"]["gate_blocked"] is True


def test_committed_artifacts_exist_and_validate() -> None:
    recon_path = RECON_DIR / "blinded-reconstruction-v1.json"
    nt_path = RECON_DIR / "nt-holdout-blinded-reconstruction-v1.json"
    assert recon_path.is_file(), "run eval reconstruct to refresh committed artifact"
    assert nt_path.is_file()
    # Large-file hook: each committed JSON must stay ≤ 1024 KiB.
    assert recon_path.stat().st_size <= 1024 * 1024
    assert nt_path.stat().st_size <= 1024 * 1024
    package = json.loads(recon_path.read_text(encoding="utf-8"))
    assert package["owned_by"] == "JAK-40"
    assert package["blinding"]["greek_consulted_during_reconstruction"] is False
    assert package["summary"]["thomas_units"] == 1
    assert package["summary"]["nt_holdout_units"] == 0
    nt_pkg = json.loads(nt_path.read_text(encoding="utf-8"))
    assert nt_pkg["summary"]["nt_holdout_units"] >= 30
    assert nt_pkg["summary"]["thomas_units"] == 0
    runner = CliRunner()
    val = runner.invoke(app, ["eval", "validate-reconstruction", str(recon_path)])
    assert val.exit_code == 0, val.output
    val_nt = runner.invoke(app, ["eval", "validate-reconstruction", str(nt_path)])
    assert val_nt.exit_code == 0, val_nt.output
    score_path = RECON_DIR / "thomas-7-surviving-letter-eval-v1.json"
    assert score_path.is_file()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    assert score["owned_by"] == "JAK-40"
    # Either gate-blocked or (future) official — never silent invent human-checked
    assert score["review_status"] != "human-checked" or score["official"] is True


def test_decode_model_still_loads_without_opening_fragments_in_reconstruct() -> None:
    model = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    surfaces = coptic_surfaces_from_normalized("ⲙⲁⲕⲁⲣⲓⲟⲥ ⲣⲱⲙⲉ")
    unit = reconstruct_unit(
        model,
        unit_id="thomas.7",
        target_kind=BlindedTargetKind.THOMAS_PILOT,
        source_id="dilley-thomas",
        coptic_surfaces=surfaces,
        request_id="test-makarios",
    )
    assert unit.decode_result.provenance.get("llm") is False

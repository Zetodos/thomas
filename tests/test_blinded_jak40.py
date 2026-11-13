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
    holdout_units,
    load_thomas_corpus,
    reconstruct_blinded_package,
    reconstruct_unit,
    reconstruct_unit_from_text,
    thomas_unit,
)
from zetodos_thomas.eval.letters import (
    gap_aware_letter_score,
    greek_letters_only,
    token_aware_letter_score,
)
from zetodos_thomas.eval.review_gate import EvalReviewGateError
from zetodos_thomas.eval.spans import (
    DILLEY_SEGMENTATION,
    HORNER_SEGMENTATION,
    construction_patterns_from_model,
    horner_bound_group_tokens,
    resolve_covering_spans,
)
from zetodos_thomas.eval.surviving_letters import (
    DEFAULT_THOMAS7_LINE_IDS,
    BlindedScoreIntegrityError,
    assert_restorations_excluded_from_gold,
    extract_surviving_letter_gold,
    load_fragments_at_eval_time,
    package_content_sha256,
    score_thomas7_surviving,
    transcription_by_id,
    try_score_thomas7_or_gate_report,
    verify_official_freeze,
    write_freeze_record,
)
from zetodos_thomas.eval.validate_reconstruction import (
    ReconstructionValidationError,
    assert_valid_reconstruction,
    validate_reconstruction_package,
)
from zetodos_thomas.eval.word_for_word import fit_argmax_table, predict_surfaces
from zetodos_thomas.reconstruct.decoder import decode_request
from zetodos_thomas.reconstruct.model import load_decode_model
from zetodos_thomas.schemas.blinded import (
    BlindedFreezeRecord,
    BlindedTargetKind,
    UnclearLetterPolicy,
)
from zetodos_thomas.schemas.decode import DecodeRequest, DecodeSpanInput
from zetodos_thomas.schemas.fragment import (
    FragmentLayerKind,
    FragmentLine,
    FragmentToken,
)
from zetodos_thomas.schemas.mappings import AblationCondition

REPO_ROOT = Path(__file__).resolve().parents[1]
BLIND_PATH = REPO_ROOT / "data" / "evidence" / "eval" / "thomas-7-blinded-v1.json"
RECON_DIR = REPO_ROOT / "data" / "reconstructions" / "jak40"

# Jake review r4569004284: exact unsplit whitespace groups on Mark 4:4.
MARK_4_4_UNSPLIT = ("ⲛⲧⲉⲣⲉϥϫⲟ,", "ⲛϭⲓⲛϩⲁⲗⲁⲧⲉ")


def test_gap_aware_letter_score_ignores_accents_and_punct() -> None:
    metrics = gap_aware_letter_score("Μακάριος ἐστίν!", ["καρι εστιν"])
    assert metrics.gold_denominator == len(greek_letters_only("καρι εστιν"))
    assert metrics.true_positives >= 8
    assert metrics.precision is not None
    assert metrics.metric_kind == "gap-aware-ordered-runs-v1"


def test_global_lcs_not_used_as_precision_for_duplicate_harvest() -> None:
    """Matthew 5:3-style: duplicate οτι in hyp must not claim precision=1.0."""
    hyp = ["οτι", "οτι", "η", "βασιλεια"]
    gold = ["οτι", "η", "βασιλεια"]
    metrics = token_aware_letter_score(hyp, gold)
    assert metrics.precision is not None
    assert metrics.precision < 1.0
    assert metrics.incidental_global_lcs >= metrics.true_positives
    assert metrics.metric_kind == "token-positional-letter-v1"


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
    assert not gold.surviving_letters.startswith(greek_letters_only("μα"))


def test_unclear_certainty_excluded_from_primary_gold() -> None:
    from zetodos_thomas.schemas.fragment import FragmentTranscription

    line = FragmentLine(
        line_id="main.40",
        n="40",
        tokens=[
            FragmentToken(
                kind=FragmentLayerKind.SURVIVING, text="κ", certainty="unclear"
            ),
            FragmentToken(
                kind=FragmentLayerKind.SURVIVING, text="άρι", certainty="clear"
            ),
        ],
    )
    tx = FragmentTranscription(
        id="fixture-654",
        source_id="dclp-poxy-654",
        manuscript_id="P.Oxy. 654",
        lines=[line],
    )
    gold = extract_surviving_letter_gold(
        tx, line_ids=["main.40"], unclear_policy=UnclearLetterPolicy.EXCLUDE
    )
    assert gold.clear_letters == greek_letters_only("άρι")
    assert gold.unclear_letters == greek_letters_only("κ")
    assert gold.primary_gold_letters == greek_letters_only("άρι")


def test_poxy_unclear_tokens_on_thomas7_lines() -> None:
    corpus = load_fragments_at_eval_time(REPO_ROOT)
    tx = transcription_by_id(corpus, "poxy-654-dclp")
    gold = extract_surviving_letter_gold(tx, line_ids=DEFAULT_THOMAS7_LINE_IDS)
    unclear = {
        (span.line_id, tok.text)
        for span in gold.spans
        for tok in span.surviving_tokens
        if tok.certainty == "unclear"
    }
    assert ("main.40", "κ") in unclear
    assert ("main.41", "α") in unclear
    assert ("main.42", "ὃν") in unclear
    assert gold.unclear_letters == greek_letters_only("καον")
    assert len(gold.clear_letters) < len(gold.surviving_letters)
    assert gold.primary_gold_letters == gold.clear_letters


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
    assert unit.segmentation == DILLEY_SEGMENTATION
    assert unit.source_id == "dilley-thomas"
    assert "ⲙⲁⲕⲁⲣⲓⲟⲥ" in unit.coptic_surfaces
    want = greek_letters_only("μακαριος")
    assert any(
        s is not None and want in greek_letters_only(s)
        for s in unit.top1_greek_surfaces
    )
    # Provenance: real Dilley token ids, not decode-input.
    for span in unit.decode_result.spans:
        assert span.input_span is not None
        assert span.input_span.source_id == "dilley-thomas"
        assert span.input_span.token_ids
        assert all(
            tid.startswith("v1:dilley-thomas/") for tid in span.input_span.token_ids
        )
        assert "decode-input" not in span.span_id
    blob = json.dumps(package.model_dump(mode="json"), ensure_ascii=False)
    assert "ἔσται. μα" not in blob
    assert package.word_for_word_baseline
    assert package.word_for_word_baseline[0].method == "train-cooccurrence-argmax-v1"
    assert_valid_reconstruction(package, repo_root=REPO_ROOT)


def test_multi_token_jak38_construction_applied_on_holdout() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=True,
        include_word_for_word=False,
    )
    multi = [
        (u.unit_id, span)
        for u in package.units
        for span in u.decode_result.spans
        if len(span.surfaces) > 1
    ]
    assert multi, "expected at least one multi-token JAK-38 covering span"
    # Known hits from frozen JAK-38 patterns.
    unit_ids = {uid for uid, _ in multi}
    assert unit_ids & {"luke.8.11", "matthew.13.11", "luke.8.10"}
    for _uid, span in multi:
        assert span.construction_id
        assert span.input_span is not None
        assert len(span.input_span.token_ids) == len(span.surfaces) > 1


def test_nt_bound_group_tokenization_parity_mark_4_4() -> None:
    """Whitespace leaves Mark 4:4 unsplit; bound-group-v1 splits them."""
    holdouts = holdout_units(REPO_ROOT)
    mark = next(u for u in holdouts if u.unit_id == "mark.4.4")
    ws = mark.sahidic.normalized_text.split()
    for surface in MARK_4_4_UNSPLIT:
        assert surface in ws
    stream = horner_bound_group_tokens(mark.sahidic)
    norms = [t.normalized for t in stream.tokens]
    for surface in MARK_4_4_UNSPLIT:
        assert surface not in norms
    assert "ⲛⲧⲉⲣⲉ" in norms and "ϥϫⲟ," in norms
    assert "ⲛϭⲓ" in norms and "ⲛϩⲁⲗⲁⲧⲉ" in norms
    assert stream.segmentation == HORNER_SEGMENTATION
    assert stream.id_scheme_version == "2"
    assert all(t.id.startswith("v2:copsahhorner/mark.4.4/") for t in stream.tokens)

    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=True,
        include_word_for_word=True,
    )
    unit = next(u for u in package.units if u.unit_id == "mark.4.4")
    assert unit.segmentation == HORNER_SEGMENTATION
    for surface in MARK_4_4_UNSPLIT:
        assert surface not in unit.coptic_surfaces
    base = next(b for b in package.word_for_word_baseline if b.unit_id == "mark.4.4")
    assert base.segmentation == HORNER_SEGMENTATION
    for surface in MARK_4_4_UNSPLIT:
        assert surface not in base.coptic_surfaces


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
    assert len(holdouts) >= 30
    assert not any(u.unit_id.startswith("john.1.") for u in holdouts)
    assert not any(u.unit_id.startswith("john.8.") for u in holdouts)


def test_mismatched_token_id_rejected_by_decoder() -> None:
    model = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    request = DecodeRequest(
        id="bad-token",
        owned_by="JAK-40",
        ablation_condition=model.condition,
        unit_id="thomas.7",
        spans=[
            DecodeSpanInput(
                span_id="v1:dilley-thomas/thomas.7/s/t0-0",
                source_id="dilley-thomas",
                unit_id="thomas.7",
                token_ids=["v1:dilley-thomas/thomas.8/t/0"],  # wrong unit
                surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ"],
            )
        ],
    )
    result = decode_request(model, request)
    assert result.spans[0].status.value == "abstain"
    assert result.spans[0].abstention_reason is not None
    assert result.spans[0].abstention_reason.value == "input_id_inconsistent"
    assert result.spans[0].input_span is None


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


def test_freeze_pin_required_for_official_and_detects_tamper(tmp_path: Path) -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=False,
        include_word_for_word=False,
    )
    freeze_path = tmp_path / "freeze.json"
    freeze = write_freeze_record(package, freeze_path)
    verify_official_freeze(package, freeze)
    # Tamper top-1 after freeze.
    unit = package.units[0]
    tampered_top1 = list(unit.top1_greek_surfaces)
    if tampered_top1:
        tampered_top1[0] = "ταμπερ"
    tampered_unit = unit.model_copy(update={"top1_greek_surfaces": tampered_top1})
    tampered = package.model_copy(update={"units": [tampered_unit]})
    assert package_content_sha256(tampered) != freeze.reconstruction_sha256
    with pytest.raises(BlindedScoreIntegrityError, match="digest"):
        verify_official_freeze(tampered, freeze)


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
    assert report.scores[0].metric_kind == "gap-aware-ordered-runs-v1"
    assert len(report.gold.restoration_letters_excluded) > 0
    assert report.blinding["greek_loaded_at_scoring"] is True
    assert report.blinding["restorations_in_gold_denominator"] is False
    assert report.blinding["unclear_in_primary_gold"] is False


def test_poxy_surviving_extract_matches_corpus_layers() -> None:
    corpus = load_fragments_at_eval_time(REPO_ROOT)
    tx = transcription_by_id(corpus, "poxy-654-dclp")
    gold = extract_surviving_letter_gold(tx, line_ids=DEFAULT_THOMAS7_LINE_IDS)
    line40 = next(s for s in gold.spans if s.line_id == "main.40")
    assert "κ" in line40.surviving_text
    assert "άρι" in line40.surviving_text
    assert "μα" not in line40.surviving_text


def test_word_for_word_argmax_blind() -> None:
    model = LexicalBaselineModel()
    model.pair_counts[("ⲣⲱⲙⲉ", "ανθρωπος")] = 5
    model.pair_counts[("ⲣⲱⲙⲉ", "τις")] = 1
    model.left_counts["ⲣⲱⲙⲉ"] = 5
    model.right_counts["ανθρωπος"] = 5
    model.right_counts["τις"] = 1
    table = fit_argmax_table(model)
    assert predict_surfaces(table, ["ⲣⲱⲙⲉ", "xyz"]) == ["ανθρωπος", None]


def test_validate_rejects_top1_tamper_and_train_as_holdout() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=False,
        include_word_for_word=False,
    )
    assert_valid_reconstruction(package, repo_root=REPO_ROOT)
    unit = package.units[0]
    bad_top1 = list(unit.top1_greek_surfaces)
    if bad_top1:
        bad_top1[0] = "not-from-candidates"
    tampered = package.model_copy(
        update={"units": [unit.model_copy(update={"top1_greek_surfaces": bad_top1})]}
    )
    problems = validate_reconstruction_package(tampered, repo_root=REPO_ROOT)
    assert any("top1" in p for p in problems)
    with pytest.raises(ReconstructionValidationError):
        assert_valid_reconstruction(tampered, repo_root=REPO_ROOT)

    # Train verse labeled as holdout must fail split membership.
    holdout_pkg = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=True,
        include_word_for_word=False,
    )
    trainish = next(
        u for u in holdout_pkg.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    )
    fake = trainish.model_copy(update={"unit_id": "john.1.1"})
    # Keep decode ids pointing at john.1.1 so unit_id checks fire on split role.
    fake_pkg = holdout_pkg.model_copy(update={"units": [fake]})
    split_problems = validate_reconstruction_package(fake_pkg, repo_root=REPO_ROOT)
    assert any("john.1.1" in p and "holdout" in p for p in split_problems)


def test_validate_rejects_decode_input_source() -> None:
    package = reconstruct_blinded_package(
        REPO_ROOT,
        include_nt_holdouts=False,
        include_word_for_word=False,
    )
    unit = package.units[0]
    span = unit.decode_result.spans[0]
    bad_span = span.model_copy(
        update={
            "span_id": "v1:decode-input/thomas.7/s/0",
            "input_span": (
                span.input_span.model_copy(update={"source_id": "decode-input"})
                if span.input_span
                else None
            ),
        }
    )
    bad_result = unit.decode_result.model_copy(
        update={"spans": [bad_span, *unit.decode_result.spans[1:]]}
    )
    bad_unit = unit.model_copy(
        update={"source_id": "decode-input", "decode_result": bad_result}
    )
    bad_pkg = package.model_copy(update={"units": [bad_unit]})
    problems = validate_reconstruction_package(
        bad_pkg, repo_root=REPO_ROOT, check_digests=False, check_splits=False
    )
    assert any("decode-input" in p or "forbidden" in p for p in problems)


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
    freeze_path = RECON_DIR / "thomas-7-blinded-freeze-v1.json"
    assert recon_path.is_file(), "run eval reconstruct to refresh committed artifact"
    assert nt_path.is_file()
    assert freeze_path.is_file()
    assert recon_path.stat().st_size <= 1024 * 1024
    assert nt_path.stat().st_size <= 1024 * 1024
    package = json.loads(recon_path.read_text(encoding="utf-8"))
    assert package["owned_by"] == "JAK-40"
    assert package["blinding"]["greek_consulted_during_reconstruction"] is False
    assert package["summary"]["thomas_units"] == 1
    assert package["summary"]["nt_holdout_units"] == 0
    # No decode-input provenance in committed Thomas artifact.
    blob = recon_path.read_text(encoding="utf-8")
    assert "decode-input" not in blob
    freeze = BlindedFreezeRecord.model_validate(
        json.loads(freeze_path.read_text(encoding="utf-8"))
    )
    from zetodos_thomas.schemas.blinded import BlindedReconstructionPackage

    pkg = BlindedReconstructionPackage.model_validate(package)
    verify_official_freeze(pkg, freeze)
    nt_pkg = json.loads(nt_path.read_text(encoding="utf-8"))
    assert nt_pkg["summary"]["nt_holdout_units"] >= 30
    assert nt_pkg["summary"]["thomas_units"] == 0
    assert "decode-input" not in nt_path.read_text(encoding="utf-8")
    # Mark 4:4 must not retain whitespace-unsplit groups.
    mark = next(u for u in nt_pkg["units"] if u["unit_id"] == "mark.4.4")
    for surface in MARK_4_4_UNSPLIT:
        assert surface not in mark["coptic_surfaces"]
    runner = CliRunner()
    val = runner.invoke(app, ["eval", "validate-reconstruction", str(recon_path)])
    assert val.exit_code == 0, val.output
    val_nt = runner.invoke(app, ["eval", "validate-reconstruction", str(nt_path)])
    assert val_nt.exit_code == 0, val_nt.output
    score_path = RECON_DIR / "thomas-7-surviving-letter-eval-v1.json"
    assert score_path.is_file()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    assert score["owned_by"] == "JAK-40"
    assert score["review_status"] != "human-checked" or score["official"] is True


def test_decode_model_still_loads_without_opening_fragments_in_reconstruct() -> None:
    model = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    corpus = load_thomas_corpus(REPO_ROOT)
    unit = thomas_unit(corpus, "thomas.7")
    rebuilt = reconstruct_unit_from_text(
        model,
        text_unit=unit,
        target_kind=BlindedTargetKind.THOMAS_PILOT,
        request_id="test-makarios",
    )
    assert rebuilt.decode_result.provenance.get("llm") is False
    assert rebuilt.segmentation == DILLEY_SEGMENTATION
    # Ad-hoc surfaces path still works for Dilley.
    smoke = reconstruct_unit(
        model,
        unit_id="thomas.7",
        target_kind=BlindedTargetKind.THOMAS_PILOT,
        source_id="dilley-thomas",
        coptic_surfaces=["ⲙⲁⲕⲁⲣⲓⲟⲥ", "ⲣⲱⲙⲉ"],
        request_id="test-surfaces",
    )
    assert smoke.decode_result.provenance.get("llm") is False


def test_covering_spans_prefer_longer_construction() -> None:
    model = load_decode_model(
        REPO_ROOT,
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
    )
    patterns = construction_patterns_from_model(model.constructions_by_pattern)
    holdouts = holdout_units(REPO_ROOT)
    luke = next(u for u in holdouts if u.unit_id == "luke.8.11")
    stream = horner_bound_group_tokens(luke.sahidic)
    windows = resolve_covering_spans(stream, patterns)
    multi = [w for w in windows if w.end - w.start > 1]
    assert multi
    assert any(w.surfaces == ("ⲡ", "ϣⲁϫⲉ") for w in multi)
    # Coverage: every token in exactly one window.
    covered = []
    for w in windows:
        covered.extend(range(w.start, w.end))
    assert covered == list(range(len(stream.tokens)))

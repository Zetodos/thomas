# SPDX-License-Identifier: Apache-2.0
"""JAK-28: splits, structural validation, lexical baseline, benchmark offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.align.benchmark import run_benchmark
from zetodos_thomas.align.eflomal_probe import probe_eflomal
from zetodos_thomas.align.lexical_baseline import (
    TokenLink,
    fit_lexical_baseline,
    gold_links_from_reference,
    null_alignment_rate,
    predict_links,
    score_links,
)
from zetodos_thomas.align.paratext import (
    filter_alignable_tokens,
    is_mark_1_1_heading_token,
    unit_has_mark_heading,
)
from zetodos_thomas.align.splits import (
    DEFAULT_SPLITS,
    PassageRange,
    PilotSplits,
    SplitRole,
    load_splits,
    nonempty_positive_units,
    partition_corpus,
)
from zetodos_thomas.align.tokenize import whitespace_tokens
from zetodos_thomas.align.validate import validate_pilot_corpus
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.normalize import (
    normalize_coptic,
    normalize_greek,
    recompute_normalized,
    strip_editorial_marks,
)
from zetodos_thomas.ingest.ntvmr_vaticanus import parse_teiraw
from zetodos_thomas.ingest.pilot import (
    EXPECTED_NONEMPTY_POSITIVE,
    EXPECTED_OMISSION_UNITS,
    EXPECTED_UNIT_COUNT,
    load_pilot_corpus,
)
from zetodos_thomas.schemas.alignment import AlignmentDocument, ParallelRelation

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
SPLITS_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "splits.json"
GOLD_PATH = (
    REPO_ROOT
    / "data"
    / "sources"
    / "alignment"
    / "reference"
    / "horner-vaticanus-gold-sample-v1.json"
)
NTVMR_FIXTURES = Path(__file__).parent / "fixtures" / "ntvmr"


def test_pilot_expanded_counts_and_sources() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    assert corpus.id == "horner-vaticanus-nt-pilot-v2"
    assert corpus.greek_source_id == "ntvmr-vaticanus-b"
    assert corpus.sahidic_source_id == "copsahhorner"
    assert len(corpus.units) == EXPECTED_UNIT_COUNT
    splits = load_splits(SPLITS_PATH)
    report = validate_pilot_corpus(corpus, splits=splits)
    assert report.ok, report.errors
    assert report.nonempty_positive == EXPECTED_NONEMPTY_POSITIVE
    assert report.omission_units == EXPECTED_OMISSION_UNITS


def test_splits_cover_every_unit_without_overlap() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    splits = load_splits(SPLITS_PATH)
    buckets = partition_corpus(corpus, splits)
    seen: list[str] = []
    for role, units in buckets.items():
        for unit in units:
            assert splits.role_for(unit.unit_id) is role
            seen.append(unit.unit_id)
    assert len(seen) == len(set(seen)) == len(corpus.units)
    assert [u.unit_id for u in buckets[SplitRole.HOLDOUT]] == [
        "matthew.13.31",
        "matthew.13.32",
    ]
    assert [u.unit_id for u in buckets[SplitRole.OMISSION]] == [
        f"john.8.{v}" for v in range(1, 12)
    ]
    assert any("7/17/30/77" in x for x in splits.excluded_from_training_stack)


def test_normalization_recompute_and_editorial_marks() -> None:
    assert strip_editorial_marks("τ[η]ν") == "την"
    assert normalize_greek("εν  [η]ν  ") == "εν ην"
    assert "<" not in normalize_coptic("ⲧⲁⲣ<ⲟⲩ>ⲟⲩⲱⲛ")
    assert normalize_coptic("ⲧⲁⲣ<ⲟⲩ>ⲟⲩⲱⲛ") == "ⲧⲁⲣⲟⲩⲟⲩⲱⲛ"
    corpus = load_pilot_corpus(PILOT_PATH)
    matt = next(u for u in corpus.units if u.unit_id == "matthew.7.7")
    assert "<" in matt.sahidic.original_text or "ⲟⲩ" in matt.sahidic.original_text
    assert (
        recompute_normalized(
            language="coptic", original_text=matt.sahidic.original_text
        )
        == matt.sahidic.normalized_text
    )
    john15 = next(u for u in corpus.units if u.unit_id == "john.1.15")
    assert "[" in john15.greek.original_text
    assert "[" not in john15.greek.normalized_text
    assert (
        recompute_normalized(language="greek", original_text=john15.greek.original_text)
        == john15.greek.normalized_text
    )


def test_mark_heading_paratext_excluded_from_alignable() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    mark = next(u for u in corpus.units if u.unit_id == "mark.1.1")
    assert unit_has_mark_heading(mark.sahidic)
    assert "paratext" in mark.notes.lower() or "heading" in mark.notes.lower()
    tokens = whitespace_tokens(mark.sahidic)
    assert is_mark_1_1_heading_token(tokens[0])
    alignable = filter_alignable_tokens(tokens)
    assert tokens[0] not in alignable
    assert alignable[0].normalized == "ⲧⲁⲣⲭⲏ"


def test_new_tei_fixtures_parse_nonempty() -> None:
    for name, expected_n in (
        ("Matt_5_14-16.teiraw.xml", 3),
        ("Matt_7_7-8.teiraw.xml", 2),
        ("Matt_13_31-32.teiraw.xml", 2),
        ("Luke_6_20-21.teiraw.xml", 2),
    ):
        units = parse_teiraw(NTVMR_FIXTURES / name)
        assert len(units) == expected_n
        assert all(u.original_text for u in units)


def test_gold_reference_valid_and_omission_shared_absence() -> None:
    doc = AlignmentDocument.model_validate_json(GOLD_PATH.read_text(encoding="utf-8"))
    assert doc.tokens
    assert "Provisional" in doc.title
    assert "provisional" in doc.notes.lower()
    assert not doc.title.startswith("Human-")
    assert "do not cite as Human-checked/Human-verified" in doc.notes
    assert any(p.relation is ParallelRelation.PHRASE for p in doc.parallels)
    assert any(
        p.provenance == "provisional-draft-jak28-unreviewed" for p in doc.parallels
    )
    assert any(p.agreement.value == "shared-absence" for p in doc.parallels)
    # Heading span exists and is not lexically parallelled to a Greek content span
    heading = next(s for s in doc.spans if s.id.endswith("/mark.1.1/s/book-heading"))
    assert heading.token_ids
    gold = gold_links_from_reference(doc)
    assert not any(g.unit_id == "john.8.1" for g in gold)
    assert any(g.unit_id == "matthew.13.31" for g in gold)
    # Holdout smoke-test scale (not a generalizable F1 sample).
    assert sum(1 for g in gold if g.unit_id == "matthew.13.31") == 7


def test_lexical_baseline_fit_excludes_holdout_and_omission() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    splits = load_splits(SPLITS_PATH)
    buckets = partition_corpus(corpus, splits)
    train = nonempty_positive_units(buckets[SplitRole.TRAIN])
    holdout_ids = {u.unit_id for u in buckets[SplitRole.HOLDOUT]}
    model = fit_lexical_baseline(train)
    assert model.train_unit_ids
    assert not (set(model.train_unit_ids) & holdout_ids)
    assert "john.8.1" not in model.train_unit_ids
    doc = AlignmentDocument.model_validate_json(GOLD_PATH.read_text(encoding="utf-8"))
    gold = gold_links_from_reference(doc)
    predicted = predict_links(model, train[:3])
    scores = score_links(predicted, gold)
    assert scores.predicted >= 0
    assert scores.gold >= 0
    assert "No pass/fail threshold" in scores.notes


def test_benchmark_report_has_denominators_no_threshold() -> None:
    report = run_benchmark(
        pilot_path=PILOT_PATH,
        splits=load_splits(SPLITS_PATH),
        reference_paths=[GOLD_PATH],
    )
    assert report["pass_threshold"] is None
    assert report["validation_ok"] is True
    overall = report["lexical_baseline"]["overall"]
    assert overall["predicted_denominator"] > 0
    assert overall["gold_denominator"] > 0
    assert overall["true_positives"] >= 0
    assert report["omission_units"] == 11
    assert report["eflomal"]["selected_for_default_path"] is False
    assert not report["omission_leak_problems"]
    # Holdout must not appear in train fit list
    assert "matthew.13.31" not in report["lexical_baseline"]["train_units_fit"]
    ref = report["human_reference"]
    assert ref["status"] == "provisional"
    assert ref["provenance_tag"] == "provisional-draft-jak28-unreviewed"
    assert (
        "Cartesian" in ref["phrase_scoring"]
        or "link-expansion" in ref["phrase_scoring"]
    )
    assert "7 links" in ref["holdout_scale"]
    nulls = report["lexical_baseline"]["null_alignments"]
    assert nulls["scoring_min_dice"] == 0.0
    assert nulls["forced_prediction_coverage"] == 1.0
    assert "null_rate" not in nulls  # do not present forced coverage as empirical nulls
    assert nulls["null_rate_at_abstention_threshold"] is not None
    assert any(
        "Cartesian" in lim or "phrase" in lim.lower() for lim in report["limitations"]
    )
    assert any(
        "smoke" in lim.lower() or "7 links" in lim for lim in report["limitations"]
    )


def test_null_alignment_rate_distinguishes_forced_vs_abstention() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    unit = next(u for u in corpus.units if u.unit_id == "john.1.1")
    model = fit_lexical_baseline([unit])
    forced = predict_links(model, [unit], min_dice=0.0)
    stats = null_alignment_rate(
        [unit], forced, scoring_min_dice=0.0, abstention_threshold=0.05
    )
    assert stats["forced_prediction_coverage"] == 1.0
    assert stats["missing_predictions"] == 0
    assert 0.0 <= float(stats["null_rate_at_abstention_threshold"] or 0.0) <= 1.0


def test_eflomal_probe_does_not_raise() -> None:
    result = probe_eflomal()
    assert result.selected_for_default_path is False
    assert isinstance(result.detail, str) and result.detail


def test_cli_align_validate_and_benchmark() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["align", "validate", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.stdout + result.stderr
    out = REPO_ROOT / "data" / "sources" / "alignment" / "benchmark-report.json"
    result = runner.invoke(
        app,
        [
            "align",
            "benchmark",
            "--repo-root",
            str(REPO_ROOT),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pass_threshold"] is None
    assert data["greek_source_id"] == "ntvmr-vaticanus-b"


def test_default_splits_match_committed_file() -> None:
    committed = load_splits(SPLITS_PATH)
    assert committed.id == DEFAULT_SPLITS.id
    assert committed.model_dump() == DEFAULT_SPLITS.model_dump()


def test_dice_counts_each_type_once_per_verse() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    john = next(u for u in corpus.units if u.unit_id == "john.1.1")
    model = fit_lexical_baseline([john])
    # Repeated λογος / ⲡϣⲁϫⲉ surfaces must not yield Dice > 1.
    assert model.dice("ⲡϣⲁϫⲉ", "λογος") <= 1.0
    assert model.left_counts["ⲁⲩⲱ"] == 1  # two surfaces, one type-event
    assert model.right_counts["λογος"] == 1
    assert model.pair_counts[("ⲡϣⲁϫⲉ", "λογος")] == 1


def test_score_links_restricts_to_annotated_left_coverage() -> None:
    gold = [
        TokenLink(
            unit_id="u.1.1",
            left_token_id="L0",
            right_token_id="R0",
            left_norm="a",
            right_norm="α",
        )
    ]
    predicted = [
        TokenLink(
            unit_id="u.1.1",
            left_token_id="L0",
            right_token_id="R0",
            left_norm="a",
            right_norm="α",
        ),
        TokenLink(
            unit_id="u.1.1",
            left_token_id="L_unannotated",
            right_token_id="R9",
            left_norm="noise",
            right_norm="ν",
        ),
    ]
    covered = score_links(predicted, gold, restrict_to_annotated_left=True)
    assert covered.true_positives == 1
    assert covered.predicted == 1
    assert covered.precision == 1.0
    raw = score_links(predicted, gold, restrict_to_annotated_left=False)
    assert raw.predicted == 2
    assert raw.precision == 0.5


def test_score_links_f1_zero_when_denominators_nonzero() -> None:
    gold = [
        TokenLink(
            unit_id="u.1.1",
            left_token_id="L0",
            right_token_id="R0",
            left_norm="a",
            right_norm="α",
        )
    ]
    predicted = [
        TokenLink(
            unit_id="u.1.1",
            left_token_id="L0",
            right_token_id="R_wrong",
            left_norm="a",
            right_norm="β",
        )
    ]
    scores = score_links(predicted, gold)
    assert scores.true_positives == 0
    assert scores.predicted == 1
    assert scores.gold == 1
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.f1 == 0.0


def test_pilot_splits_reject_overlapping_ranges() -> None:
    with pytest.raises(ValueError, match="overlapping split ranges"):
        PilotSplits(
            id="bad-overlap",
            title="overlap",
            greek_source_id="ntvmr-vaticanus-b",
            sahidic_source_id="copsahhorner",
            passages=[
                PassageRange(
                    book="matthew",
                    chapter=13,
                    verse_from=31,
                    verse_to=32,
                    role=SplitRole.TRAIN,
                ),
                PassageRange(
                    book="matthew",
                    chapter=13,
                    verse_from=31,
                    verse_to=31,
                    role=SplitRole.HOLDOUT,
                ),
            ],
        )


def test_cli_explicit_missing_splits_fails() -> None:
    runner = CliRunner()
    missing = REPO_ROOT / "data" / "sources" / "pilot" / "does-not-exist-splits.json"
    result = runner.invoke(
        app,
        [
            "align",
            "validate",
            "--repo-root",
            str(REPO_ROOT),
            "--splits",
            str(missing),
        ],
    )
    assert result.exit_code != 0
    combined = (result.stdout + result.stderr).lower()
    assert "does-not-exist" in combined or "exist" in combined or "invalid" in combined

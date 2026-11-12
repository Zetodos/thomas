# SPDX-License-Identifier: Apache-2.0
"""JAK-36: bitext prep, Pharaoh symmetrize, statistical eval scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from zetodos_thomas.align.bitext import build_bitext, subsample_train
from zetodos_thomas.align.lexical_baseline import TokenLink
from zetodos_thomas.align.splits import SplitRole, load_canonical_splits
from zetodos_thomas.align.statistical import (
    assert_fit_score_disjoint,
    links_to_token_links,
    score_intrinsic,
    score_norm_gold,
)
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.integrations.aligners.discovery import discover_aligners
from zetodos_thomas.integrations.aligners.pharaoh import (
    format_pharaoh_line,
    parse_pharaoh_line,
    symmetrize_grow_diag_final_and,
)
from zetodos_thomas.integrations.aligners.pins import (
    EFLOMAL_COMMIT_SHA,
    FAST_ALIGN_COMMIT_SHA,
    pinned_versions_payload,
)
from zetodos_thomas.integrations.aligners.types import (
    AlignerName,
    AlignmentLinks,
    BitextSentence,
    DirectionalAlignment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = REPO_ROOT / "data" / "sources" / "pilot" / "horner-vaticanus-nt-pilot.json"
SPLITS_PATH = (
    REPO_ROOT / "data" / "training" / "splits" / "horner-vaticanus-nt-split-v3.json"
)
OMISSION_PATH = (
    REPO_ROOT / "data" / "evidence" / "omissions" / "john-8-pa-omission-fixture-v3.json"
)


def _splits():
    return load_canonical_splits(SPLITS_PATH, OMISSION_PATH)


def test_pharaoh_roundtrip_and_symmetrize() -> None:
    pairs = ((0, 0), (1, 2), (3, 3))
    assert parse_pharaoh_line(format_pharaoh_line(pairs)) == pairs
    assert parse_pharaoh_line("0-0 1-1 ||| -1.23") == ((0, 0), (1, 1))
    fwd = ((0, 0), (1, 1), (2, 3))
    rev = ((0, 0), (1, 2), (2, 3))
    sym = symmetrize_grow_diag_final_and(fwd, rev, src_len=4, trg_len=4)
    assert (0, 0) in sym
    assert (2, 3) in sym


def test_bitext_uses_v3_bound_group_and_excludes_omission() -> None:
    corpus = load_pilot_corpus(PILOT_PATH)
    splits = _splits()
    assert corpus.id == "horner-vaticanus-nt-pilot-v3"
    assert splits.id == "horner-vaticanus-nt-split-v3"
    bitext = build_bitext(corpus, splits)
    assert bitext.coptic_segmentation == "bound-group-v1"
    assert bitext.id_scheme_segmented == "2"
    assert len(bitext.sentences_for(SplitRole.OMISSION)) == 0
    train = bitext.sentences_for(SplitRole.TRAIN)
    dev = bitext.sentences_for(SplitRole.DEV)
    holdout = bitext.sentences_for(SplitRole.HOLDOUT)
    assert len(train) == 304
    assert len(dev) == 51
    assert len(holdout) == 41
    # Segmented Coptic should often exceed Greek whitespace count on John 1:1.
    john = next(s for s in train if s.unit_id == "john.1.1")
    assert john.source_token_ids[0].startswith("v2:")
    assert john.target_token_ids[0].startswith("v1:")
    assert john.source_parent_v1_ids
    assert all(pid.startswith("v1:") for pid in john.source_parent_v1_ids if pid)
    assert len(john.source_parent_v1_ids) == len(john.source_tokens)
    assert any(
        "poxy-654" in ex.lower() or "654" in ex
        for ex in bitext.excluded_from_training_stack
    )
    assert not any(s.unit_id.startswith("thomas.") for s in train)
    assert bitext.checksum_sha256


def test_subsample_train_is_deterministic() -> None:
    sentences = [
        BitextSentence(
            unit_id=f"u.{i}",
            source_tokens=("a",),
            target_tokens=("b",),
            source_token_ids=(f"v2:s/u.{i}/t/0",),
            target_token_ids=(f"v1:g/u.{i}/t/0",),
            split_role="train",
        )
        for i in range(20)
    ]
    a = subsample_train(sentences, n=5, seed=36)
    b = subsample_train(sentences, n=5, seed=36)
    c = subsample_train(sentences, n=5, seed=37)
    assert [s.unit_id for s in a] == [s.unit_id for s in b]
    assert [s.unit_id for s in a] != [s.unit_id for s in c]


def test_intrinsic_and_norm_gold_scoring() -> None:
    sentences = [
        BitextSentence(
            unit_id="john.1.1",
            source_tokens=("ϩⲛ", "ⲧⲉϩⲟⲩⲉⲓⲧⲉ"),
            target_tokens=("εν", "αρχη"),
            source_token_ids=("v2:c/john.1.1/t/0", "v2:c/john.1.1/t/1"),
            target_token_ids=("v1:g/john.1.1/t/0", "v1:g/john.1.1/t/1"),
            split_role="dev",
            source_parent_v1_ids=("v1:c/john.1.1/t/0", "v1:c/john.1.1/t/0"),
            source_char_spans=((0, 2), (2, 11)),
        )
    ]
    links = AlignmentLinks(
        aligner=AlignerName.EFLOMAL,
        sentences=sentences,
        forward=[DirectionalAlignment("john.1.1", ((0, 0), (1, 1)))],
        reverse=[DirectionalAlignment("john.1.1", ((0, 0), (1, 1)))],
        symmetrized=[DirectionalAlignment("john.1.1", ((0, 0), (1, 1)))],
    )
    intrinsic = score_intrinsic(links, eval_unit_ids={"john.1.1"})
    assert intrinsic.sentence_count == 1
    assert intrinsic.mean_fwd_rev_agreement == 1.0
    pred = links_to_token_links(links)
    # Both bound-group pieces project onto the same parent-v1 id.
    gold = [
        TokenLink(
            unit_id="john.1.1",
            left_token_id="v1:c/john.1.1/t/0",
            right_token_id="v1:g/john.1.1/t/0",
            left_norm="ϩⲛⲧⲉϩⲟⲩⲉⲓⲧⲉ",
            right_norm="εν",
        )
    ]
    scored = score_norm_gold(pred, gold, unit_ids={"john.1.1"})
    assert scored["true_positives"] == 1
    assert scored["recall"] == 1.0
    # Two preds share one parent id; only the εν match counts as TP, αρχη is FP.
    assert scored["predicted_denominator"] == 2
    assert scored["precision"] == 0.5


def test_repeated_coptic_pi_does_not_cross_match() -> None:
    """Segmented ``ⲡ`` must not TP against a different parent that merely contains ⲡ."""
    sentences = [
        BitextSentence(
            unit_id="toy.pi",
            source_tokens=("ⲡ", "ϣⲁϫⲉ", "ⲡ", "ⲛⲟⲩⲧⲉ"),
            target_tokens=("ο", "λογος", "ο", "θεος"),
            source_token_ids=(
                "v2:c/toy.pi/t/0",
                "v2:c/toy.pi/t/1",
                "v2:c/toy.pi/t/2",
                "v2:c/toy.pi/t/3",
            ),
            target_token_ids=(
                "v1:g/toy.pi/t/0",
                "v1:g/toy.pi/t/1",
                "v1:g/toy.pi/t/2",
                "v1:g/toy.pi/t/3",
            ),
            split_role="dev",
            source_parent_v1_ids=(
                "v1:c/toy.pi/t/0",
                "v1:c/toy.pi/t/1",
                "v1:c/toy.pi/t/2",
                "v1:c/toy.pi/t/3",
            ),
            source_char_spans=((0, 1), (1, 5), (6, 7), (7, 12)),
        )
    ]
    gold = [
        TokenLink(
            unit_id="toy.pi",
            left_token_id="v1:c/toy.pi/t/0",
            right_token_id="v1:g/toy.pi/t/0",
            left_norm="ⲡ",
            right_norm="ο",
        ),
        TokenLink(
            unit_id="toy.pi",
            left_token_id="v1:c/toy.pi/t/2",
            right_token_id="v1:g/toy.pi/t/3",
            left_norm="ⲡ",
            right_norm="θεος",
        ),
    ]
    # Wrong under old substring rule: first ⲡ → θεος (shared surface letter + right).
    links_wrong = AlignmentLinks(
        aligner=AlignerName.EFLOMAL,
        sentences=sentences,
        forward=[DirectionalAlignment("toy.pi", ((0, 3),))],
        reverse=[DirectionalAlignment("toy.pi", ((0, 3),))],
        symmetrized=[DirectionalAlignment("toy.pi", ((0, 3),))],
    )
    scored_wrong = score_norm_gold(
        links_to_token_links(links_wrong), gold, unit_ids={"toy.pi"}
    )
    assert scored_wrong["true_positives"] == 0
    assert scored_wrong["predicted_denominator"] == 1
    # Correct positional links.
    links_ok = AlignmentLinks(
        aligner=AlignerName.EFLOMAL,
        sentences=sentences,
        forward=[DirectionalAlignment("toy.pi", ((0, 0), (2, 3)))],
        reverse=[DirectionalAlignment("toy.pi", ((0, 0), (2, 3)))],
        symmetrized=[DirectionalAlignment("toy.pi", ((0, 0), (2, 3)))],
    )
    scored_ok = score_norm_gold(
        links_to_token_links(links_ok), gold, unit_ids={"toy.pi"}
    )
    assert scored_ok["true_positives"] == 2
    assert scored_ok["recall"] == 1.0


def test_oos_fit_score_unit_ids_disjoint() -> None:
    fit = [
        BitextSentence(
            unit_id="train.1",
            source_tokens=("ⲁ",),
            target_tokens=("a",),
            source_token_ids=("v2:c/train.1/t/0",),
            target_token_ids=("v1:g/train.1/t/0",),
            split_role="train",
        )
    ]
    score = [
        BitextSentence(
            unit_id="dev.1",
            source_tokens=("ⲃ",),
            target_tokens=("b",),
            source_token_ids=("v2:c/dev.1/t/0",),
            target_token_ids=("v1:g/dev.1/t/0",),
            split_role="dev",
        )
    ]
    assert_fit_score_disjoint(fit, score)
    with pytest.raises(AssertionError, match="fit/score unit_id overlap"):
        assert_fit_score_disjoint(fit, fit)


def test_aligner_pins_are_immutable_shas() -> None:
    pins = pinned_versions_payload()
    assert pins["eflomal"]["commit_sha"] == EFLOMAL_COMMIT_SHA
    assert pins["fast_align"]["commit_sha"] == FAST_ALIGN_COMMIT_SHA
    assert len(EFLOMAL_COMMIT_SHA) == 40
    assert len(FAST_ALIGN_COMMIT_SHA) == 40


def test_jake_decision_keep_both_is_recorded() -> None:
    from zetodos_thomas.align.statistical import _jake_decision_scaffold
    from zetodos_thomas.integrations.aligners.discovery import AlignerAvailability
    from zetodos_thomas.integrations.aligners.types import AlignerName

    availability = {
        "eflomal": AlignerAvailability(
            name=AlignerName.EFLOMAL,
            available=True,
            binary_path="/tmp/eflomal",
            python_module=True,
            detail="ok",
        ),
        "fast_align": AlignerAvailability(
            name=AlignerName.FAST_ALIGN,
            available=True,
            binary_path="/tmp/fast_align",
            python_module=False,
            detail="ok",
        ),
    }
    full = {
        "dev": {
            "eflomal": {
                "available": True,
                "intrinsic": {"mean_fwd_rev_agreement": 0.5},
            },
            "fast_align": {
                "available": True,
                "intrinsic": {"mean_fwd_rev_agreement": 0.1},
            },
        }
    }
    curves = [
        {
            "fraction": 0.25,
            "aligners": {
                "eflomal": {
                    "available": True,
                    "intrinsic": {"mean_fwd_rev_agreement": 0.4},
                },
                "fast_align": {
                    "available": True,
                    "intrinsic": {"mean_fwd_rev_agreement": 0.1},
                },
            },
        },
        {
            "fraction": 1.0,
            "aligners": {
                "eflomal": {
                    "available": True,
                    "intrinsic": {"mean_fwd_rev_agreement": 0.5},
                },
                "fast_align": {
                    "available": True,
                    "intrinsic": {"mean_fwd_rev_agreement": 0.1},
                },
            },
        },
    ]
    decision = _jake_decision_scaffold(full, curves, availability)
    assert decision["status"] == "decided"
    assert decision["decision"] == "keep-both"
    assert decision["recommendation"] == "keep-both"
    assert decision["adopted_aligner"] is None
    assert decision["phase2_active_aligners"] == ["eflomal", "fast_align"]
    assert decision["decided_by"] == "Jake"
    assert decision["question_for_jake"] is None


def test_symmetrize_final_and_is_order_stable() -> None:
    # Competing isolated links: forward-only should win over reverse-only.
    fwd = ((0, 0), (1, 2))
    rev = ((0, 0), (2, 1))
    a = symmetrize_grow_diag_final_and(fwd, rev, src_len=3, trg_len=3)
    b = symmetrize_grow_diag_final_and(fwd, rev, src_len=3, trg_len=3)
    assert a == b
    assert (1, 2) in a  # forward-only preferred in final-and


def test_discover_aligners_does_not_raise() -> None:
    report = discover_aligners(REPO_ROOT)
    assert "eflomal" in report
    assert "fast_align" in report
    assert report["eflomal"].name.value == "eflomal"


def test_aligner_probe_cli() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["align", "aligner-probe", "--repo-root", str(REPO_ROOT)]
    )
    assert result.exit_code == 0
    assert "eflomal" in result.stdout
    assert "fast_align" in result.stdout


@pytest.mark.skipif(
    not discover_aligners(REPO_ROOT)["fast_align"].available,
    reason="fast_align binary not set up in this environment",
)
def test_fast_align_smoke_on_tiny_bitext() -> None:
    from zetodos_thomas.integrations.aligners.fast_align_adapter import align_fast_align
    from zetodos_thomas.integrations.aligners.types import AlignerConfig

    sentences = [
        BitextSentence(
            unit_id="toy.1",
            source_tokens=("ⲡ", "ϣⲁϫⲉ"),
            target_tokens=("ο", "λογος"),
            source_token_ids=("v2:c/toy.1/t/0", "v2:c/toy.1/t/1"),
            target_token_ids=("v1:g/toy.1/t/0", "v1:g/toy.1/t/1"),
            split_role="train",
        ),
        BitextSentence(
            unit_id="toy.2",
            source_tokens=("ⲡ", "ϣⲁϫⲉ"),
            target_tokens=("ο", "λογος"),
            source_token_ids=("v2:c/toy.2/t/0", "v2:c/toy.2/t/1"),
            target_token_ids=("v1:g/toy.2/t/0", "v1:g/toy.2/t/1"),
            split_role="dev",
        ),
    ]
    links = align_fast_align(
        sentences,
        config=AlignerConfig(seed=36, fast_align_iterations=2),
        repo_root=REPO_ROOT,
    )
    assert links.aligner is AlignerName.FAST_ALIGN
    assert len(links.forward) == 2
    assert len(links.symmetrized) == 2


@pytest.mark.skipif(
    not discover_aligners(REPO_ROOT)["eflomal"].python_module,
    reason="eflomal Python package not installed in this environment",
)
def test_eflomal_smoke_on_tiny_bitext() -> None:
    from zetodos_thomas.integrations.aligners.eflomal_adapter import align_eflomal
    from zetodos_thomas.integrations.aligners.types import AlignerConfig

    sentences = [
        BitextSentence(
            unit_id=f"toy.{i}",
            source_tokens=("ⲡ", "ϣⲁϫⲉ"),
            target_tokens=("ο", "λογος"),
            source_token_ids=(f"v2:c/toy.{i}/t/0", f"v2:c/toy.{i}/t/1"),
            target_token_ids=(f"v1:g/toy.{i}/t/0", f"v1:g/toy.{i}/t/1"),
            split_role="train" if i == 0 else "dev",
        )
        for i in range(4)
    ]
    links = align_eflomal(
        sentences,
        config=AlignerConfig(seed=36, eflomal_n_samplers=1),
        repo_root=REPO_ROOT,
    )
    assert links.aligner is AlignerName.EFLOMAL
    assert len(links.forward) == 4

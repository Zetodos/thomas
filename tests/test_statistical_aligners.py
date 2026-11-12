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
            source_tokens=("ϩⲛⲧⲉ", "ϩⲟⲩⲉⲓⲧⲉ"),
            target_tokens=("εν", "αρχη"),
            source_token_ids=("v2:c/john.1.1/t/0", "v2:c/john.1.1/t/1"),
            target_token_ids=("v1:g/john.1.1/t/0", "v1:g/john.1.1/t/1"),
            split_role="dev",
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
    # Whitespace gold left is the fused form; segmented pred peels project onto it.
    gold = [
        TokenLink(
            unit_id="john.1.1",
            left_token_id="v1:ignored",
            right_token_id="v1:ignored",
            left_norm="ϩⲛⲧⲉϩⲟⲩⲉⲓⲧⲉ",
            right_norm="εν",
        )
    ]
    scored = score_norm_gold(pred, gold, unit_ids={"john.1.1"})
    assert scored["true_positives"] == 1
    assert scored["recall"] == 1.0
    # Both segmented pieces project onto the fused gold left → both kept.
    assert scored["predicted_denominator"] == 2
    assert scored["precision"] == 0.5


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

# SPDX-License-Identifier: Apache-2.0
"""Gospel re-fit keeps v3 holdouts out of training and unions aligner links."""

from __future__ import annotations

import pytest

from zetodos_thomas.align.splits import SplitRole
from zetodos_thomas.dictionary.build import DictionaryBuildError
from zetodos_thomas.dictionary.diagnose import load_nt_splits
from zetodos_thomas.dictionary.wider import (
    observed_links_from_alignments,
    role_for_unit,
    stable_eflomal_links,
    wider_splits,
)
from zetodos_thomas.ingest.paths import find_repo_root
from zetodos_thomas.integrations.aligners.types import (
    AlignerName,
    AlignmentLinks,
    BitextSentence,
    DirectionalAlignment,
)


def _links(name: AlignerName, pairs: tuple[tuple[int, int], ...]) -> AlignmentLinks:
    sentence = BitextSentence(
        unit_id="matthew.2.1",
        source_tokens=("ⲁ", "ⲃ"),
        target_tokens=("και", "ο"),
        source_token_ids=("c0", "c1"),
        target_token_ids=("g0", "g1"),
        split_role="train",
    )
    return AlignmentLinks(
        aligner=name,
        sentences=[sentence],
        forward=[],
        reverse=[],
        symmetrized=[DirectionalAlignment(unit_id=sentence.unit_id, pairs=pairs)],
    )


def test_link_union_counts_once_when_both_aligners_agree() -> None:
    observed, context = observed_links_from_alignments(
        {
            "eflomal": _links(AlignerName.EFLOMAL, ((1, 0),)),
            "fast_align": _links(AlignerName.FAST_ALIGN, ((1, 0),)),
        }
    )
    assert context["ⲃ"]["και"]["ⲁ"] == 2
    assert len(observed) == 1
    assert observed[0].observation_count == 1
    assert observed[0].count_eflomal == 1
    assert observed[0].count_fast_align == 1


def test_eflomal_sample_intersection_drops_flips() -> None:
    first = _links(AlignerName.EFLOMAL, ((0, 0), (1, 1)))
    second = _links(AlignerName.EFLOMAL, ((0, 0), (1, 0)))
    stable, stats = stable_eflomal_links(first, second)
    assert stable.symmetrized[0].pairs == ((0, 0),)
    assert stats == {"stable_links": 1, "dropped_from_first": 1}


def test_single_aligner_batch_is_refused() -> None:
    with pytest.raises(DictionaryBuildError):
        observed_links_from_alignments(
            {"eflomal": _links(AlignerName.EFLOMAL, ((0, 0),))}
        )


def test_v3_holdout_is_not_relabeled_as_train() -> None:
    base = load_nt_splits(find_repo_root())
    holdout = base.unit_ids_for(SplitRole.HOLDOUT)
    assert holdout
    splits = wider_splits([*holdout, "matthew.2.1"], base)
    train_ids = set(splits.unit_ids_for(SplitRole.TRAIN))
    assert "matthew.2.1" in train_ids
    assert not (set(holdout) & train_ids)
    assert role_for_unit(holdout[0], base) is SplitRole.HOLDOUT
    with pytest.raises(DictionaryBuildError):
        role_for_unit("thomas.7", base)

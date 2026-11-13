# SPDX-License-Identifier: Apache-2.0
"""Elementary train-only word-for-word baseline (JAK-28 lineage; JAK-40/41).

Predicts Greek from Coptic using train co-occurrence argmax — never opens the
holdout / Thomas Greek witness. Decoder comparison: ``eval.feasibility`` (JAK-41).
"""

from __future__ import annotations

from collections import defaultdict

from zetodos_thomas.align.lexical_baseline import LexicalBaselineModel
from zetodos_thomas.schemas.blinded import BlindedTargetKind, WordForWordUnitBaseline


def fit_argmax_table(
    model: LexicalBaselineModel,
) -> dict[str, str]:
    """Map each Coptic type to its highest Dice Greek type (train only)."""
    best: dict[str, tuple[float, str]] = {}
    # Group pair counts by left type.
    rights_by_left: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (left, right), count in model.pair_counts.items():
        rights_by_left[left].append((right, count))
    for left, rights in rights_by_left.items():
        ranked: list[tuple[float, str]] = []
        for right, _count in rights:
            ranked.append((model.dice(left, right), right))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if ranked and ranked[0][0] > 0:
            best[left] = (ranked[0][0], ranked[0][1])
    return {left: greek for left, (_score, greek) in best.items()}


def predict_surfaces(
    argmax_table: dict[str, str],
    coptic_surfaces: list[str],
) -> list[str | None]:
    """Blinded per-surface prediction; None when unseen in train."""
    return [argmax_table.get(surface) for surface in coptic_surfaces]


def baseline_for_unit(
    *,
    unit_id: str,
    target_kind: BlindedTargetKind,
    source_id: str,
    coptic_surfaces: list[str],
    argmax_table: dict[str, str],
    segmentation: str = "",
) -> WordForWordUnitBaseline:
    predicted = predict_surfaces(argmax_table, coptic_surfaces)
    n_hit = sum(1 for p in predicted if p is not None)
    return WordForWordUnitBaseline(
        unit_id=unit_id,
        target_kind=target_kind,
        source_id=source_id,
        coptic_surfaces=list(coptic_surfaces),
        predicted_greek=predicted,
        segmentation=segmentation,
        notes=[
            "Train-cooccurrence argmax; target Greek never consulted.",
            f"covered_surfaces={n_hit}/{len(coptic_surfaces)}",
            f"segmentation={segmentation or 'unspecified'}",
            "Compared to decoder in JAK-41 feasibility metrics on identical inputs.",
        ],
    )

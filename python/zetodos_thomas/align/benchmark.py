# SPDX-License-Identifier: Apache-2.0
"""Run the JAK-28 alignment pilot benchmark offline and return a report dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zetodos_thomas.align.eflomal_probe import probe_eflomal
from zetodos_thomas.align.lexical_baseline import (
    assert_no_omission_in_links,
    fit_lexical_baseline,
    gold_links_from_reference,
    null_alignment_rate,
    predict_links,
    reference_parallel_inventory,
    score_links,
)
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    nonempty_positive_units,
    partition_corpus,
)
from zetodos_thomas.align.validate import validate_pilot_corpus
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.schemas.alignment import AlignmentDocument


def load_reference_docs(paths: list[Path]) -> list[AlignmentDocument]:
    """Load and validate human reference AlignmentDocuments."""
    docs: list[AlignmentDocument] = []
    for path in paths:
        docs.append(
            AlignmentDocument.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return docs


def run_benchmark(
    *,
    pilot_path: Path,
    splits: PilotSplits,
    reference_paths: list[Path],
) -> dict[str, Any]:
    """Execute validation + lexical baseline eval; return JSON-serializable report."""
    corpus = load_pilot_corpus(pilot_path)
    validation = validate_pilot_corpus(corpus, splits=splits)
    buckets = partition_corpus(corpus, splits)
    train = nonempty_positive_units(buckets[SplitRole.TRAIN])
    dev = nonempty_positive_units(buckets[SplitRole.DEV])
    holdout = nonempty_positive_units(buckets[SplitRole.HOLDOUT])
    omission = buckets[SplitRole.OMISSION]

    model = fit_lexical_baseline(train)
    refs = load_reference_docs(reference_paths)
    gold = [link for doc in refs for link in gold_links_from_reference(doc)]
    gold_units = {g.unit_id for g in gold}

    # Score only against units that have human gold AND are not omission.
    omission_ids = {u.unit_id for u in omission}
    eval_pool = [
        u
        for u in [*train, *dev, *holdout]
        if u.unit_id in gold_units and u.unit_id not in omission_ids
    ]
    predicted = predict_links(model, eval_pool)
    leak = assert_no_omission_in_links(predicted, omission_ids)
    leak += assert_no_omission_in_links(gold, omission_ids)

    overall = score_links(predicted, gold, unit_ids=gold_units - omission_ids)
    by_split: dict[str, object] = {}
    for role in (SplitRole.TRAIN, SplitRole.DEV, SplitRole.HOLDOUT):
        role_ids = set(splits.unit_ids_for(role)) & gold_units
        if not role_ids:
            by_split[role.value] = {
                "skipped": True,
                "reason": "no human gold links for this split in reference fixtures",
            }
            continue
        by_split[role.value] = score_links(predicted, gold, unit_ids=role_ids).as_dict()

    eflomal = probe_eflomal()
    failure_examples = _failure_examples(predicted, gold, limit=8)

    report: dict[str, Any] = {
        "id": "horner-vaticanus-alignment-benchmark-v1",
        "title": "Greek-Sahidic alignment pilot benchmark (JAK-28)",
        "greek_source_id": corpus.greek_source_id,
        "sahidic_source_id": corpus.sahidic_source_id,
        "pilot_id": corpus.id,
        "pilot_units": validation.unit_count,
        "nonempty_positive_units": validation.nonempty_positive,
        "omission_units": validation.omission_units,
        "validation_ok": validation.ok,
        "validation_issues": [
            {
                "code": i.code,
                "severity": i.severity,
                "unit_id": i.unit_id,
                "message": i.message,
            }
            for i in validation.issues
        ],
        "splits": {
            "train_units": len(buckets[SplitRole.TRAIN]),
            "dev_units": len(buckets[SplitRole.DEV]),
            "holdout_units": len(buckets[SplitRole.HOLDOUT]),
            "omission_units": len(buckets[SplitRole.OMISSION]),
            "train_nonempty": len(train),
            "dev_nonempty": len(dev),
            "holdout_nonempty": len(holdout),
            "excluded_from_training_stack": splits.excluded_from_training_stack,
        },
        "lexical_baseline": {
            "algorithm": "verse-bag co-occurrence Dice + mild positional prior",
            "train_units_fit": model.train_unit_ids,
            "pair_types": len(model.pair_counts),
            "overall": overall.as_dict(),
            "by_split": by_split,
            "null_alignments": null_alignment_rate(eval_pool, predicted),
            "failure_examples": failure_examples,
        },
        "human_reference": {
            "documents": [p.name for p in reference_paths],
            "gold_link_count": len(gold),
            "parallel_inventory": {
                path.name: reference_parallel_inventory(doc)
                for path, doc in zip(reference_paths, refs, strict=True)
            },
        },
        "eflomal": {
            "available": eflomal.available,
            "selected_for_default_path": eflomal.selected_for_default_path,
            "detail": eflomal.detail,
        },
        "limitations": [
            "No numeric pass threshold - baseline only (PLAN.md Phase 2).",
            "Whitespace tokenization is not morphology; Coptic groups stay fused.",
            "Verse-bag Dice over-generates co-occurrence mass; phrase order is weak.",
            "Human gold is a small sample, not exhaustive verse annotation.",
            "John 8:1-11 omission fixtures excluded from positive denominators.",
            "Matthew 13:31-32 holdout scored only when present in gold fixtures.",
            "Greek = NTVMR Vaticanus B* (GA 03 / docID 20003) only; no WH/SBLGNT mix.",
            "eflomal unavailable without native build; not on default path.",
            "Thomas Greek fragments (7/17/30/77) are outside this NT training stack.",
        ],
        "omission_leak_problems": leak,
        "pass_threshold": None,
    }
    return report


def _failure_examples(
    predicted: list,
    gold: list,
    *,
    limit: int,
) -> list[dict[str, object]]:
    pred_map = {(p.unit_id, p.left_token_id): p for p in predicted}
    examples: list[dict[str, object]] = []
    for g in gold:
        p = pred_map.get((g.unit_id, g.left_token_id))
        if p is None:
            examples.append(
                {
                    "type": "false_negative",
                    "unit_id": g.unit_id,
                    "left": g.left_norm,
                    "gold_right": g.right_norm,
                    "predicted_right": None,
                }
            )
        elif p.right_token_id != g.right_token_id:
            examples.append(
                {
                    "type": "mismatch",
                    "unit_id": g.unit_id,
                    "left": g.left_norm,
                    "gold_right": g.right_norm,
                    "predicted_right": p.right_norm,
                    "predicted_score": p.score,
                }
            )
        if len(examples) >= limit:
            break
    return examples


def write_benchmark_json(report: dict[str, Any], path: Path) -> None:
    """Write benchmark JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

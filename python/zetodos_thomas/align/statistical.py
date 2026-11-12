# SPDX-License-Identifier: Apache-2.0
"""Statistical aligner comparison + learning curves on v3 splits (JAK-36).

Out-of-sample protocol: fit on train (subsets) only, apply to fixed unseen
dev/holdout. Never treats PA omission fixtures as positive pairs. No LLM.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zetodos_thomas.align.bitext import build_bitext, subsample_train
from zetodos_thomas.align.lexical_baseline import (
    TokenLink,
    fit_lexical_baseline,
    gold_links_from_reference,
    predict_links,
    score_links,
)
from zetodos_thomas.align.paratext import filter_alignable_tokens
from zetodos_thomas.align.project_spans import parent_whitespace_token
from zetodos_thomas.align.segment_coptic import segmented_coptic_tokens
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    nonempty_positive_units,
    partition_corpus,
)
from zetodos_thomas.align.tokenize import whitespace_tokens
from zetodos_thomas.integrations.aligners.discovery import discover_aligners
from zetodos_thomas.integrations.aligners.eflomal_adapter import (
    EflomalNotAvailableError,
    align_eflomal,
    write_priors_from_links,
)
from zetodos_thomas.integrations.aligners.fast_align_adapter import (
    FastAlignNotAvailableError,
    align_fast_align,
)
from zetodos_thomas.integrations.aligners.pharaoh import symmetrize_batch
from zetodos_thomas.integrations.aligners.pins import pinned_versions_payload
from zetodos_thomas.integrations.aligners.types import (
    AlignerConfig,
    AlignerName,
    AlignmentLinks,
    BitextSentence,
    DirectionalAlignment,
)
from zetodos_thomas.schemas.alignment import AlignmentDocument
from zetodos_thomas.schemas.source import ParallelUnit, PilotCorpus

DEFAULT_CURVE_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class IntrinsicScores:
    """Intrinsic bidirectional quality (no gold required)."""

    sentence_count: int
    mean_fwd_rev_agreement: float | None
    mean_alignment_density: float | None
    mean_source_fertility: float | None
    total_sym_links: int

    def as_dict(self) -> dict[str, object]:
        return {
            "sentence_count": self.sentence_count,
            "mean_fwd_rev_agreement": self.mean_fwd_rev_agreement,
            "mean_alignment_density": self.mean_alignment_density,
            "mean_source_fertility": self.mean_source_fertility,
            "total_sym_links": self.total_sym_links,
            "notes": (
                "fwd_rev_agreement = |fwd&rev| / |fwd U rev| per sentence, "
                "averaged. density = symmetrized links / (src_len * trg_len). "
                "fertility = symmetrized links / src_len."
            ),
        }


def _index_alignments(
    alignments: Sequence[DirectionalAlignment],
) -> dict[str, set[tuple[int, int]]]:
    return {a.unit_id: set(a.pairs) for a in alignments}


def score_intrinsic(
    links: AlignmentLinks,
    *,
    eval_unit_ids: set[str],
) -> IntrinsicScores:
    """Score bidirectional agreement / density on eval units only."""
    fwd = _index_alignments(links.forward)
    rev = _index_alignments(links.reverse)
    sym = _index_alignments(links.symmetrized)
    by_unit = {s.unit_id: s for s in links.sentences}
    agreements: list[float] = []
    densities: list[float] = []
    fertilities: list[float] = []
    total_sym = 0
    counted = 0
    for unit_id in sorted(eval_unit_ids):
        sentence = by_unit.get(unit_id)
        if sentence is None:
            continue
        fset = fwd.get(unit_id, set())
        rset = rev.get(unit_id, set())
        sset = sym.get(unit_id, set())
        union = fset | rset
        agreements.append((len(fset & rset) / len(union)) if union else 1.0)
        src_n = max(len(sentence.source_tokens), 1)
        trg_n = max(len(sentence.target_tokens), 1)
        densities.append(len(sset) / (src_n * trg_n))
        fertilities.append(len(sset) / src_n)
        total_sym += len(sset)
        counted += 1
    if counted == 0:
        return IntrinsicScores(0, None, None, None, 0)
    return IntrinsicScores(
        sentence_count=counted,
        mean_fwd_rev_agreement=sum(agreements) / counted,
        mean_alignment_density=sum(densities) / counted,
        mean_source_fertility=sum(fertilities) / counted,
        total_sym_links=total_sym,
    )


def links_to_token_links(
    links: AlignmentLinks,
    *,
    unit_ids: set[str] | None = None,
    use_symmetrized: bool = True,
) -> list[TokenLink]:
    """Map Pharaoh index pairs onto TokenLink rows via parent-v1 projection.

    Bound-group (v2) source indices are projected onto whitespace parent-v1
    token ids using ``BitextSentence.source_parent_v1_ids`` (char-span based).
    Predictions that cannot be projected are dropped.
    """
    chosen = links.symmetrized if use_symmetrized else links.forward
    by_pairs = {a.unit_id: a.pairs for a in chosen}
    out: list[TokenLink] = []
    for sentence in links.sentences:
        if unit_ids is not None and sentence.unit_id not in unit_ids:
            continue
        parents = sentence.source_parent_v1_ids
        for i, j in by_pairs.get(sentence.unit_id, ()):
            if i < 0 or j < 0:
                continue
            if i >= len(sentence.source_token_ids) or j >= len(
                sentence.target_token_ids
            ):
                continue
            if i >= len(parents) or not parents[i]:
                continue
            out.append(
                TokenLink(
                    unit_id=sentence.unit_id,
                    left_token_id=parents[i],
                    right_token_id=sentence.target_token_ids[j],
                    left_norm=sentence.source_tokens[i],
                    right_norm=sentence.target_tokens[j],
                    score=1.0,
                )
            )
    return out


def project_token_links_to_parent_v1(
    predicted: list[TokenLink],
    units: Sequence[ParallelUnit],
) -> list[TokenLink]:
    """Rewrite segmented left_token_ids onto parent whitespace v1 ids via spans."""
    by_unit = {u.unit_id: u for u in units}
    out: list[TokenLink] = []
    for link in predicted:
        unit = by_unit.get(link.unit_id)
        if unit is None:
            continue
        segmented = filter_alignable_tokens(segmented_coptic_tokens(unit.sahidic))
        whitespace = filter_alignable_tokens(whitespace_tokens(unit.sahidic))
        by_seg = {t.id: t for t in segmented}
        seg = by_seg.get(link.left_token_id)
        if seg is None:
            # Already a v1 id, or unknown — keep only if it matches a whitespace id.
            if any(t.id == link.left_token_id for t in whitespace):
                out.append(link)
            continue
        parent = parent_whitespace_token(seg, whitespace)
        if parent is None:
            continue
        out.append(
            TokenLink(
                unit_id=link.unit_id,
                left_token_id=parent.id,
                right_token_id=link.right_token_id,
                left_norm=link.left_norm,
                right_norm=link.right_norm,
                score=link.score,
            )
        )
    return out


def score_norm_gold(
    predicted: list[TokenLink],
    gold: list[TokenLink],
    *,
    unit_ids: set[str] | None = None,
) -> dict[str, object]:
    """Gold score via positional parent-v1 identity (no substring matching).

    Predictions must already carry whitespace parent-v1 ``left_token_id`` values
    (see ``links_to_token_links`` / ``project_token_links_to_parent_v1``). A true
    positive is an exact ``(unit_id, left_token_id, right_norm)`` match against
    gold. Repeated identical surfaces (e.g. multiple ``ⲡ``) do not cross-match
    across different parent token ids / Greek rights.
    """
    gld = [g for g in gold if unit_ids is None or g.unit_id in unit_ids]
    pred = [p for p in predicted if unit_ids is None or p.unit_id in unit_ids]
    annotated_left = {(g.unit_id, g.left_token_id) for g in gld}
    pred = [p for p in pred if (p.unit_id, p.left_token_id) in annotated_left]

    def _key(link: TokenLink) -> tuple[str, str, str]:
        return (link.unit_id, link.left_token_id, link.right_norm)

    # Multiset-aware matching so duplicate keys do not over-count.
    gold_bag: Counter[tuple[str, str, str]] = Counter(_key(g) for g in gld)
    pred_bag: Counter[tuple[str, str, str]] = Counter(_key(p) for p in pred)
    tp = sum((pred_bag & gold_bag).values())
    pred_n = sum(pred_bag.values())
    gold_n = sum(gold_bag.values())
    precision = (tp / pred_n) if pred_n else None
    recall = (tp / gold_n) if gold_n else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "predicted_denominator": pred_n,
        "gold_denominator": gold_n,
        "matched_predictions": tp,
        "unit_ids_scored": sorted(
            {*(p.unit_id for p in pred), *(g.unit_id for g in gld)}
        ),
        "notes": (
            "Positional parent-v1 identity: predicted bound-group tokens are "
            "projected onto whitespace parent ids via char spans; match key is "
            "(unit_id, left_token_id, right_norm). No substring membership. "
            "Repeated parents (e.g. ⲡ) cannot cross-match. Provisional gold — "
            "intrinsic agreement remains the primary bake-off metric."
        ),
    }


def assert_fit_score_disjoint(
    fit_sents: Sequence[BitextSentence],
    score_sents: Sequence[BitextSentence],
) -> None:
    """Raise if any unit_id appears in both fit and score sets (OOS guard)."""
    fit_ids = {s.unit_id for s in fit_sents}
    score_ids = {s.unit_id for s in score_sents}
    overlap = fit_ids & score_ids
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        msg = f"fit/score unit_id overlap (OOS violated): {sample}"
        raise AssertionError(msg)


def _score_links_block(
    name: AlignerName,
    links: AlignmentLinks,
    *,
    eval_ids: set[str],
    train_size: int,
    eval_size: int,
    protocol: str,
    gold: list[TokenLink] | None,
    fit_unit_ids: list[str] | None = None,
    score_unit_ids: list[str] | None = None,
) -> dict[str, Any]:
    intrinsic = score_intrinsic(links, eval_unit_ids=eval_ids)
    pred_links = links_to_token_links(links, unit_ids=eval_ids)
    result: dict[str, Any] = {
        "aligner": name.value,
        "available": True,
        "train_size": train_size,
        "eval_size": eval_size,
        "apply_size": len(links.sentences),
        "protocol": protocol,
        "fit_unit_ids": fit_unit_ids,
        "score_unit_ids": score_unit_ids,
        "intrinsic": intrinsic.as_dict(),
        "config": links.config,
        "notes": links.notes,
        "_links": links,  # retained for exact artifact export; stripped before JSON
    }
    if gold:
        result["norm_gold"] = score_norm_gold(pred_links, gold, unit_ids=eval_ids)
    return result


def _best_align_greedy(
    src_tokens: Sequence[str],
    trg_tokens: Sequence[str],
    pair_counts: Counter[tuple[str, str]],
) -> tuple[tuple[int, int], ...]:
    """1-best unused target per source from frozen co-occurrence counts."""
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    n_src = max(len(src_tokens), 1)
    n_trg = max(len(trg_tokens), 1)
    for i, src in enumerate(src_tokens):
        best_j = -1
        best_score = -1.0
        for j, trg in enumerate(trg_tokens):
            if j in used:
                continue
            count = float(pair_counts[(src, trg)])
            pos = 1.0 - abs(i / n_src - j / n_trg)
            score = count * 10.0 + pos
            if score > best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j))
    return tuple(pairs)


def apply_frozen_lexical_table(
    train_links: AlignmentLinks,
    eval_sents: Sequence[BitextSentence],
    *,
    symmetrization: str = "grow-diag-final-and",
) -> AlignmentLinks:
    """Apply a frozen train co-occurrence table to unseen sentences (OOS)."""
    fwd_counts: Counter[tuple[str, str]] = Counter()
    rev_counts: Counter[tuple[str, str]] = Counter()
    by_sym = {a.unit_id: a.pairs for a in train_links.symmetrized}
    for sent in train_links.sentences:
        for i, j in by_sym.get(sent.unit_id, ()):
            if i < 0 or i >= len(sent.source_tokens):
                continue
            if j < 0 or j >= len(sent.target_tokens):
                continue
            src = sent.source_tokens[i]
            trg = sent.target_tokens[j]
            fwd_counts[(src, trg)] += 1
            rev_counts[(trg, src)] += 1

    forward: list[DirectionalAlignment] = []
    reverse: list[DirectionalAlignment] = []
    src_lens: list[int] = []
    trg_lens: list[int] = []
    for sent in eval_sents:
        fwd_pairs = _best_align_greedy(
            sent.source_tokens, sent.target_tokens, fwd_counts
        )
        rev_raw = _best_align_greedy(sent.target_tokens, sent.source_tokens, rev_counts)
        rev_pairs = tuple((src_i, trg_j) for trg_j, src_i in rev_raw)
        forward.append(DirectionalAlignment(unit_id=sent.unit_id, pairs=fwd_pairs))
        reverse.append(DirectionalAlignment(unit_id=sent.unit_id, pairs=rev_pairs))
        src_lens.append(len(sent.source_tokens))
        trg_lens.append(len(sent.target_tokens))

    sym = symmetrize_batch(
        forward,
        reverse,
        src_lens,
        trg_lens,
        method=symmetrization,
    )
    return AlignmentLinks(
        aligner=AlignerName.FAST_ALIGN,
        sentences=list(eval_sents),
        forward=forward,
        reverse=reverse,
        symmetrized=sym,
        symmetrization=symmetrization,
        config={
            "protocol": "oos-frozen-lexical-table",
            "train_sentences": len(train_links.sentences),
            "pair_types": len(fwd_counts),
        },
        notes=[
            "OOS apply: frozen train co-occurrence table; greedy 1-best + symmetrize.",
            "Eval sentences were not in the fit set used to build the table.",
        ],
    )


def _oos_align_and_score(
    name: AlignerName,
    train: list[BitextSentence],
    eval_sents: list[BitextSentence],
    *,
    config: AlignerConfig,
    repo_root: Path | None,
    gold: list[TokenLink] | None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Out-of-sample: fit on train only, apply to unseen eval sentences."""
    assert_fit_score_disjoint(train, eval_sents)
    fit_ids = [s.unit_id for s in train]
    score_ids = [s.unit_id for s in eval_sents]
    eval_id_set = set(score_ids)
    if not train:
        return {
            "aligner": name.value,
            "available": False,
            "error": "empty fit set",
            "train_size": 0,
            "eval_size": len(eval_sents),
            "protocol": "out-of-sample-train-only",
            "fit_unit_ids": fit_ids,
            "score_unit_ids": score_ids,
        }
    if not eval_sents:
        return {
            "aligner": name.value,
            "available": False,
            "error": "empty score set",
            "train_size": len(train),
            "eval_size": 0,
            "protocol": "out-of-sample-train-only",
            "fit_unit_ids": fit_ids,
            "score_unit_ids": score_ids,
        }

    try:
        if name is AlignerName.EFLOMAL:
            train_links = align_eflomal(train, config=config, repo_root=repo_root)
            priors_dir = work_dir or Path(
                tempfile.mkdtemp(prefix="thomas-eflomal-priors-")
            )
            priors_dir.mkdir(parents=True, exist_ok=True)
            priors_path = priors_dir / f"train_{len(train)}.priors"
            write_priors_from_links(train_links, priors_path)
            score_links = align_eflomal(
                eval_sents,
                config=config,
                repo_root=repo_root,
                priors_path=priors_path,
            )
            score_links.notes = [
                *score_links.notes,
                "OOS: priors estimated on train only; applied to unseen score set.",
            ]
            protocol = (
                "out-of-sample: eflomal train-only → Dirichlet priors → "
                "score-set align with frozen priors"
            )
        elif name is AlignerName.FAST_ALIGN:
            train_links = align_fast_align(train, config=config, repo_root=repo_root)
            score_links = apply_frozen_lexical_table(
                train_links,
                eval_sents,
                symmetrization=config.symmetrization,
            )
            protocol = (
                "out-of-sample: fast_align train-only → frozen lexical table → "
                "greedy apply on score set"
            )
        else:
            msg = f"unsupported aligner {name}"
            raise ValueError(msg)
    except (EflomalNotAvailableError, FastAlignNotAvailableError) as exc:
        return {
            "aligner": name.value,
            "available": False,
            "error": str(exc),
            "train_size": len(train),
            "eval_size": len(eval_sents),
            "protocol": "out-of-sample-train-only",
            "fit_unit_ids": fit_ids,
            "score_unit_ids": score_ids,
        }

    return _score_links_block(
        name,
        score_links,
        eval_ids=eval_id_set,
        train_size=len(train),
        eval_size=len(eval_sents),
        protocol=protocol,
        gold=gold,
        fit_unit_ids=fit_ids,
        score_unit_ids=score_ids,
    )


def _lexical_curve_point(
    train_units: list[ParallelUnit],
    eval_units: list[ParallelUnit],
    gold: list[TokenLink],
    *,
    train_size: int,
) -> dict[str, Any]:
    model = fit_lexical_baseline(
        train_units[:train_size] if train_size < len(train_units) else train_units,
        coptic_segmentation="bound-group-v1",
    )
    # Match statistical subsample size by unit_id set when possible.
    predicted = predict_links(
        model, eval_units, min_dice=0.0, coptic_segmentation="bound-group-v1"
    )
    projected = project_token_links_to_parent_v1(predicted, eval_units)
    eval_ids = {u.unit_id for u in eval_units}
    return {
        "aligner": "lexical_dice_baseline",
        "available": True,
        "train_size": len(model.train_unit_ids),
        "eval_size": len(eval_units),
        "protocol": "out-of-sample-train-only",
        "token_id_gold": score_links(projected, gold, unit_ids=eval_ids).as_dict()
        if gold
        else None,
        "norm_gold": score_norm_gold(projected, gold, unit_ids=eval_ids)
        if gold
        else None,
        "notes": [
            "Control baseline with bound-group-v1 Coptic segmentation.",
            "Predictions projected to parent-v1 ids before gold scoring.",
        ],
    }


def run_statistical_evaluation(
    *,
    corpus: PilotCorpus,
    splits: PilotSplits,
    config: AlignerConfig | None = None,
    repo_root: Path | None = None,
    reference_paths: list[Path] | None = None,
    curve_fractions: Sequence[float] = DEFAULT_CURVE_FRACTIONS,
    include_holdout: bool = True,
) -> dict[str, Any]:
    """Compare eflomal vs fast_align with learning curves on v3 splits."""
    cfg = config or AlignerConfig()
    bitext = build_bitext(corpus, splits)
    train = bitext.sentences_for(SplitRole.TRAIN)
    dev = bitext.sentences_for(SplitRole.DEV)
    holdout = bitext.sentences_for(SplitRole.HOLDOUT)
    omission = bitext.sentences_for(SplitRole.OMISSION)
    if omission:
        msg = "internal error: omission sentences must be empty in bitext"
        raise RuntimeError(msg)

    gold: list[TokenLink] = []
    gold_docs: list[str] = []
    if reference_paths:
        for path in reference_paths:
            doc = AlignmentDocument.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            gold.extend(gold_links_from_reference(doc))
            gold_docs.append(path.name)

    availability = discover_aligners(repo_root)
    aligners = [AlignerName.EFLOMAL, AlignerName.FAST_ALIGN]

    # ParallelUnit lists for lexical baseline control (same corpus partition).
    buckets = partition_corpus(corpus, splits)
    train_units = nonempty_positive_units(buckets[SplitRole.TRAIN])
    # Order train_units to match subsample unit sets via id map.
    train_unit_by_id = {u.unit_id: u for u in train_units}
    dev_units = nonempty_positive_units(buckets[SplitRole.DEV])

    curve_points: list[dict[str, Any]] = []
    for fraction in curve_fractions:
        n = max(1, round(len(train) * fraction))
        train_subset = subsample_train(train, n=n, seed=cfg.seed)
        assert_fit_score_disjoint(train_subset, dev)
        point: dict[str, Any] = {
            "fraction": fraction,
            "train_size_requested": n,
            "train_size_actual": len(train_subset),
            "train_unit_ids": [s.unit_id for s in train_subset],
            "score_unit_ids": [s.unit_id for s in dev],
            "aligners": {},
        }
        for name in aligners:
            point["aligners"][name.value] = _oos_align_and_score(
                name,
                train_subset,
                dev,
                config=cfg,
                repo_root=repo_root,
                gold=gold,
            )
        subset_units = [
            train_unit_by_id[s.unit_id]
            for s in train_subset
            if s.unit_id in train_unit_by_id
        ]
        point["aligners"]["lexical_dice_baseline"] = _lexical_curve_point(
            subset_units, dev_units, gold, train_size=len(subset_units)
        )
        curve_points.append(point)

    full_comparison: dict[str, Any] = {"dev": {}, "holdout": {}}
    retained_links: dict[str, AlignmentLinks] = {}
    for name in aligners:
        dev_block = _oos_align_and_score(
            name, train, dev, config=cfg, repo_root=repo_root, gold=gold
        )
        links_obj = dev_block.pop("_links", None)
        if isinstance(links_obj, AlignmentLinks):
            retained_links[name.value] = links_obj
        full_comparison["dev"][name.value] = dev_block
        if include_holdout:
            hold_block = _oos_align_and_score(
                name,
                train,
                holdout,
                config=cfg,
                repo_root=repo_root,
                gold=gold,
            )
            hold_block.pop("_links", None)
            full_comparison["holdout"][name.value] = hold_block

    # Full-train lexical control on dev.
    full_comparison["dev"]["lexical_dice_baseline"] = _lexical_curve_point(
        train_units, dev_units, gold, train_size=len(train_units)
    )

    for point in curve_points:
        for block in point["aligners"].values():
            if isinstance(block, dict):
                block.pop("_links", None)

    decision = _jake_decision_scaffold(full_comparison, curve_points, availability)

    return {
        "id": "horner-vaticanus-statistical-aligner-eval-v1",
        "title": (
            "Greek-Sahidic statistical aligner comparison + learning curves (JAK-36)"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "owned_by": "JAK-36",
        "greek_source_id": bitext.greek_source_id,
        "sahidic_source_id": bitext.sahidic_source_id,
        "pilot_corpus_id": bitext.pilot_corpus_id,
        "split_id": bitext.split_id,
        "bitext": {
            "checksum_sha256": bitext.checksum_sha256,
            "coptic_segmentation": bitext.coptic_segmentation,
            "greek_tokenization": bitext.greek_tokenization,
            "id_scheme_segmented": bitext.id_scheme_segmented,
            "train_nonempty": len(train),
            "dev_nonempty": len(dev),
            "holdout_nonempty": len(holdout),
            "omission_positive_pairs": 0,
            "excluded_from_training_stack": bitext.excluded_from_training_stack,
            "notes": bitext.notes,
        },
        "config": cfg.as_dict(),
        "aligner_pins": pinned_versions_payload(),
        "availability": {k: v.as_dict() for k, v in availability.items()},
        "human_reference": {
            "documents": gold_docs,
            "gold_link_count": len(gold),
            "status": "provisional-draft",
            "scoring": (
                "positional parent-v1 identity via char-span projection; "
                "match (unit_id, left_token_id, right_norm); no substring"
            ),
        },
        "learning_curve": {
            "fractions": list(curve_fractions),
            "eval_split": "dev",
            "protocol": "out-of-sample-train-only",
            "points": curve_points,
        },
        "full_train_comparison": full_comparison,
        "jake_decision": decision,
        "retained_dev_links": retained_links,
        "limitations": [
            "Provisional gold is tiny / unreviewed; intrinsic agreement is primary.",
            "eflomal C sampler is stochastic; seed recorded but not fully pinning.",
            "Dev/holdout curves are out-of-sample: fit on train only, apply to "
            "unseen score sets (eflomal priors; fast_align frozen lexical table).",
            "Upstream aligner commits are pinned in aligner_pins / versions.json.",
            "No LLM. Greek = Vaticanus B* only. PA omission never positive pairs.",
            "Thomas 7 / P.Oxy. 654 excluded from training stack.",
        ],
    }


def _jake_decision_scaffold(
    full_comparison: dict[str, Any],
    curve_points: list[dict[str, Any]],
    availability: dict[str, Any],
) -> dict[str, Any]:
    """Present evidence; do not silently lock an aligner choice."""
    efl_ok = availability.get("eflomal") and availability["eflomal"].available
    fa_ok = availability.get("fast_align") and availability["fast_align"].available
    efl_dev = full_comparison.get("dev", {}).get("eflomal", {})
    fa_dev = full_comparison.get("dev", {}).get("fast_align", {})

    def _agree(block: dict[str, Any]) -> float | None:
        if not block.get("available"):
            return None
        intrinsic = block.get("intrinsic") or {}
        value = intrinsic.get("mean_fwd_rev_agreement")
        return float(value) if value is not None else None

    efl_a = _agree(efl_dev)
    fa_a = _agree(fa_dev)

    # Learning-curve slope proxy: agreement at 25% vs 100% on eflomal.
    def _curve_agree(aligner: str, fraction: float) -> float | None:
        for point in curve_points:
            if abs(point["fraction"] - fraction) > 1e-9:
                continue
            block = point["aligners"].get(aligner, {})
            return _agree(block)
        return None

    efl_lo = _curve_agree("eflomal", 0.25)
    efl_hi = _curve_agree("eflomal", 1.0)
    fa_lo = _curve_agree("fast_align", 0.25)
    fa_hi = _curve_agree("fast_align", 1.0)

    evidence_thin = not (efl_ok and fa_ok and efl_a is not None and fa_a is not None)
    recommendation = "needs-jake-decision"
    rationale = [
        "eflomal evaluated first per PLAN.md / JAK-36; not preselected.",
        "Identical v3 train/dev/holdout + bound-group-v1 segmentation for both.",
        "Metrics are out-of-sample (train-only fit → apply to unseen dev/holdout).",
    ]
    if evidence_thin:
        rationale.append(
            "Evidence incomplete (aligner unavailable or missing intrinsic "
            "scores). Do not lock a default - Jake must decide after setup/run."
        )
    else:
        assert efl_a is not None and fa_a is not None
        delta = efl_a - fa_a
        rationale.append(
            f"Full-train dev fwd↔rev agreement: eflomal={efl_a:.4f}, "
            f"fast_align={fa_a:.4f} (Δ={delta:+.4f})."
        )
        if efl_lo is not None and efl_hi is not None:
            rationale.append(
                f"eflomal learning curve agreement 25%→100%: {efl_lo:.4f}→{efl_hi:.4f}."
            )
        if fa_lo is not None and fa_hi is not None:
            rationale.append(
                f"fast_align learning curve agreement 25%→100%: "
                f"{fa_lo:.4f}→{fa_hi:.4f}."
            )
        # Require a clear margin before suggesting adopt; else ask Jake.
        if abs(delta) < 0.02:
            recommendation = "needs-jake-decision"
            rationale.append(
                "Agreement delta < 0.02 on provisional intrinsic metric - "
                "thin margin; prefer Jake call (also weigh build burden)."
            )
        elif delta > 0:
            recommendation = "lean-eflomal-pending-jake"
            rationale.append(
                "eflomal leads on intrinsic agreement; still pending Jake adopt."
            )
        else:
            recommendation = "lean-fast_align-pending-jake"
            rationale.append(
                "fast_align leads on intrinsic agreement; still pending Jake adopt."
            )

    return {
        "status": "pending-jake",
        "recommendation": recommendation,
        "adopted_aligner": None,
        "question_for_jake": (
            "Adopt eflomal, adopt fast_align, or keep both available for "
            "Phase 2 lexical-table work (JAK-37+)? Evidence above; gold is "
            "provisional so do not treat F1 alone as decisive."
        ),
        "rationale": rationale,
    }


def write_eval_json(report: dict[str, Any], path: Path) -> None:
    """Write the statistical evaluation report JSON (no AlignmentLinks objects)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in report.items() if k != "retained_dev_links"}
    path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_alignment_artifact(
    links: AlignmentLinks,
    *,
    path: Path,
    provenance: dict[str, Any],
    compact: bool = True,
) -> None:
    """Write inspectable alignment JSON with provenance envelope.

    ``compact=True`` (default) stores Pharaoh strings + lengths (no token
    surfaces) so committed artifacts stay under the pre-commit size ceiling.
    Pass ``compact=False`` for a pretty-printed full dump.
    """
    from zetodos_thomas.integrations.aligners.pharaoh import format_pharaoh_line

    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        by_fwd = {a.unit_id: a.pairs for a in links.forward}
        by_rev = {a.unit_id: a.pairs for a in links.reverse}
        by_sym = {a.unit_id: a.pairs for a in links.symmetrized}
        rows = [
            {
                "unit_id": s.unit_id,
                "split_role": s.split_role,
                "src_len": len(s.source_tokens),
                "trg_len": len(s.target_tokens),
                "fwd": format_pharaoh_line(by_fwd.get(s.unit_id, ())),
                "rev": format_pharaoh_line(by_rev.get(s.unit_id, ())),
                "sym": format_pharaoh_line(by_sym.get(s.unit_id, ())),
            }
            for s in links.sentences
        ]
        payload: dict[str, Any] = {
            "id": path.stem,
            "provenance": provenance,
            "aligner": links.aligner.value,
            "symmetrization": links.symmetrization,
            "config": links.config,
            "notes": [
                *links.notes,
                "Links are Pharaoh i-j strings (fwd/rev/sym); tokens via corpus.",
            ],
            "sentence_count": len(rows),
            "rows": rows,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return
    payload = {
        "id": path.stem,
        "provenance": provenance,
        "alignment": links.as_dict(),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

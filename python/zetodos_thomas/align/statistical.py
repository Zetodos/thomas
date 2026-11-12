# SPDX-License-Identifier: Apache-2.0
"""Statistical aligner comparison + learning curves on v3 splits (JAK-36).

Trains/aligns only on expanded train partitions; scores fixed dev (and optional
holdout) via the joint unsupervised protocol. Never treats PA omission fixtures
as positive pairs. No LLM.
"""

from __future__ import annotations

import json
import tempfile
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
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    nonempty_positive_units,
    partition_corpus,
)
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
    """Map Pharaoh index pairs onto TokenLink rows via sentence token ids."""
    chosen = links.symmetrized if use_symmetrized else links.forward
    by_pairs = {a.unit_id: a.pairs for a in chosen}
    out: list[TokenLink] = []
    for sentence in links.sentences:
        if unit_ids is not None and sentence.unit_id not in unit_ids:
            continue
        for i, j in by_pairs.get(sentence.unit_id, ()):
            if i < 0 or j < 0:
                continue
            if i >= len(sentence.source_token_ids) or j >= len(
                sentence.target_token_ids
            ):
                continue
            out.append(
                TokenLink(
                    unit_id=sentence.unit_id,
                    left_token_id=sentence.source_token_ids[i],
                    right_token_id=sentence.target_token_ids[j],
                    left_norm=sentence.source_tokens[i],
                    right_norm=sentence.target_tokens[j],
                    score=1.0,
                )
            )
    return out


def _left_norms_related(pred_left: str, gold_left: str) -> bool:
    """True when segmented pred_left projects onto whitespace gold_left.

    Exact equality, or pred piece is a contiguous substring of the gold
    whitespace token (bound-group peel of a fused Horner surface).
    """
    if pred_left == gold_left:
        return True
    if not pred_left or not gold_left:
        return False
    return pred_left in gold_left


def score_norm_gold(
    predicted: list[TokenLink],
    gold: list[TokenLink],
    *,
    unit_ids: set[str] | None = None,
) -> dict[str, object]:
    """Soft gold score bridging whitespace v1 gold and bound-group v2 preds.

    A prediction matches a gold link when ``unit_id`` and ``right_norm`` agree
    and the predicted Sahidic piece projects onto the gold whitespace left
    token (equal or contiguous substring). Annotated-left filtering keeps any
    pred whose left piece projects onto at least one gold left on that unit.
    """
    gld = [g for g in gold if unit_ids is None or g.unit_id in unit_ids]
    pred = [p for p in predicted if unit_ids is None or p.unit_id in unit_ids]
    gold_lefts_by_unit: dict[str, list[str]] = {}
    for g in gld:
        gold_lefts_by_unit.setdefault(g.unit_id, []).append(g.left_norm)

    def _annotated(p: TokenLink) -> bool:
        return any(
            _left_norms_related(p.left_norm, gleft)
            for gleft in gold_lefts_by_unit.get(p.unit_id, [])
        )

    pred = [p for p in pred if _annotated(p)]
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gld):
            if gi in matched_gold:
                continue
            if p.unit_id != g.unit_id or p.right_norm != g.right_norm:
                continue
            if _left_norms_related(p.left_norm, g.left_norm):
                matched_gold.add(gi)
                matched_pred.add(pi)
                break
    tp = len(matched_gold)
    precision = (tp / len(pred)) if pred else None
    recall = (tp / len(gld)) if gld else None
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
        "predicted_denominator": len(pred),
        "gold_denominator": len(gld),
        "matched_predictions": len(matched_pred),
        "unit_ids_scored": sorted(
            {*(p.unit_id for p in pred), *(g.unit_id for g in gld)}
        ),
        "notes": (
            "Soft projection: segmented left matches whitespace gold left by "
            "equality or contiguous substring; right_norm must match exactly. "
            "Provisional gold only - not human-checked. Intrinsic agreement "
            "remains the primary bake-off metric."
        ),
    }


def _align_dispatch(
    name: AlignerName,
    sentences: list[BitextSentence],
    *,
    config: AlignerConfig,
    repo_root: Path | None,
) -> AlignmentLinks:
    if name is AlignerName.EFLOMAL:
        return align_eflomal(sentences, config=config, repo_root=repo_root)
    if name is AlignerName.FAST_ALIGN:
        return align_fast_align(sentences, config=config, repo_root=repo_root)
    msg = f"unsupported aligner {name}"
    raise ValueError(msg)


def _score_links_block(
    name: AlignerName,
    links: AlignmentLinks,
    *,
    eval_ids: set[str],
    train_size: int,
    eval_size: int,
    protocol: str,
    gold: list[TokenLink] | None,
) -> dict[str, Any]:
    intrinsic = score_intrinsic(links, eval_unit_ids=eval_ids)
    pred_links = links_to_token_links(links, unit_ids=eval_ids)
    result: dict[str, Any] = {
        "aligner": name.value,
        "available": True,
        "train_size": train_size,
        "eval_size": eval_size,
        "joint_size": len(links.sentences),
        "protocol": protocol,
        "intrinsic": intrinsic.as_dict(),
        "config": links.config,
        "notes": links.notes,
        "_links": links,  # retained for exact artifact export; stripped before JSON
    }
    if gold:
        result["norm_gold"] = score_norm_gold(pred_links, gold, unit_ids=eval_ids)
    return result


def _joint_align_and_score(
    name: AlignerName,
    train: list[BitextSentence],
    eval_sents: list[BitextSentence],
    *,
    config: AlignerConfig,
    repo_root: Path | None,
    gold: list[TokenLink] | None,
) -> dict[str, Any]:
    """Joint unsupervised align of train+dev; score eval only.

    Used for learning curves and full-train **dev** (not sealed holdout).
    """
    eval_ids = {s.unit_id for s in eval_sents}
    seen: set[str] = set()
    joint: list[BitextSentence] = []
    for sentence in [*train, *eval_sents]:
        if sentence.unit_id in seen:
            continue
        seen.add(sentence.unit_id)
        joint.append(sentence)
    try:
        links = _align_dispatch(name, joint, config=config, repo_root=repo_root)
    except (EflomalNotAvailableError, FastAlignNotAvailableError) as exc:
        return {
            "aligner": name.value,
            "available": False,
            "error": str(exc),
            "train_size": len(train),
            "eval_size": len(eval_sents),
        }
    return _score_links_block(
        name,
        links,
        eval_ids=eval_ids,
        train_size=len(train),
        eval_size=len(eval_sents),
        protocol=(
            "joint unsupervised alignment of train_subset + dev; "
            "metrics on dev unit_ids only (dev is not sealed holdout)"
        ),
        gold=gold,
    )


def _holdout_align_and_score(
    name: AlignerName,
    train: list[BitextSentence],
    holdout: list[BitextSentence],
    *,
    config: AlignerConfig,
    repo_root: Path | None,
    gold: list[TokenLink] | None,
    work_dir: Path | None,
) -> dict[str, Any]:
    """Score holdout without joint-fitting on holdout pairs.

    eflomal: align TRAIN only → Dirichlet priors → align HOLDOUT with frozen
    priors. fast_align: omitted (no frozen-prior apply path in this adapter).
    """
    if name is AlignerName.FAST_ALIGN:
        return {
            "aligner": name.value,
            "available": False,
            "skipped": True,
            "train_size": len(train),
            "eval_size": len(holdout),
            "error": (
                "holdout scores omitted for fast_align: adapter has no "
                "frozen-prior / apply path; refusing joint train+holdout fit"
            ),
            "protocol": "omitted-to-prevent-holdout-leakage",
        }
    if name is not AlignerName.EFLOMAL:
        msg = f"unsupported holdout aligner {name}"
        raise ValueError(msg)
    try:
        train_links = align_eflomal(train, config=config, repo_root=repo_root)
    except EflomalNotAvailableError as exc:
        return {
            "aligner": name.value,
            "available": False,
            "error": str(exc),
            "train_size": len(train),
            "eval_size": len(holdout),
        }
    priors_dir = work_dir or Path(tempfile.mkdtemp(prefix="thomas-eflomal-priors-"))
    priors_dir.mkdir(parents=True, exist_ok=True)
    priors_path = priors_dir / "train.priors"
    write_priors_from_links(train_links, priors_path)
    try:
        holdout_links = align_eflomal(
            holdout,
            config=config,
            repo_root=repo_root,
            priors_path=priors_path,
        )
    except EflomalNotAvailableError as exc:
        return {
            "aligner": name.value,
            "available": False,
            "error": str(exc),
            "train_size": len(train),
            "eval_size": len(holdout),
        }
    return _score_links_block(
        name,
        holdout_links,
        eval_ids={s.unit_id for s in holdout},
        train_size=len(train),
        eval_size=len(holdout),
        protocol=(
            "train-only eflomal align → Dirichlet priors → holdout-only align "
            "with frozen priors (holdout never in fitting bitext)"
        ),
        gold=gold,
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
    eval_ids = {u.unit_id for u in eval_units}
    return {
        "aligner": "lexical_dice_baseline",
        "available": True,
        "train_size": len(model.train_unit_ids),
        "eval_size": len(eval_units),
        "token_id_gold": score_links(predicted, gold, unit_ids=eval_ids).as_dict()
        if gold
        else None,
        "norm_gold": score_norm_gold(predicted, gold, unit_ids=eval_ids)
        if gold
        else None,
        "notes": [
            "Control baseline with bound-group-v1 Coptic segmentation.",
            "token_id_gold may be empty/weak because gold is whitespace v1 ids.",
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
        point: dict[str, Any] = {
            "fraction": fraction,
            "train_size_requested": n,
            "train_size_actual": len(train_subset),
            "train_unit_ids": [s.unit_id for s in train_subset],
            "aligners": {},
        }
        for name in aligners:
            point["aligners"][name.value] = _joint_align_and_score(
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
        dev_block = _joint_align_and_score(
            name, train, dev, config=cfg, repo_root=repo_root, gold=gold
        )
        links_obj = dev_block.pop("_links", None)
        if isinstance(links_obj, AlignmentLinks):
            retained_links[name.value] = links_obj
        full_comparison["dev"][name.value] = dev_block
        if include_holdout:
            hold_block = _holdout_align_and_score(
                name,
                train,
                holdout,
                config=cfg,
                repo_root=repo_root,
                gold=gold,
                work_dir=None,
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
        "availability": {k: v.as_dict() for k, v in availability.items()},
        "human_reference": {
            "documents": gold_docs,
            "gold_link_count": len(gold),
            "status": "provisional-draft",
            "scoring": (
                "soft projection: segmented left onto whitespace gold left; "
                "right_norm exact"
            ),
        },
        "learning_curve": {
            "fractions": list(curve_fractions),
            "eval_split": "dev",
            "points": curve_points,
        },
        "full_train_comparison": full_comparison,
        "jake_decision": decision,
        "retained_dev_links": retained_links,
        "limitations": [
            "Provisional gold is tiny / unreviewed; intrinsic agreement is primary.",
            "eflomal C sampler is stochastic; seed recorded but not fully pinning.",
            "Dev curves use joint train+dev (dev is not sealed holdout).",
            "Holdout: eflomal train-only priors then holdout-only align; "
            "fast_align holdout omitted (no apply path).",
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

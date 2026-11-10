# SPDX-License-Identifier: Apache-2.0
"""Deterministic lexical alignment baseline (no LLM; no eflomal dependency).

Builds a bidirectional co-occurrence table from nonempty train verse pairs and
scores proposed 1-best links against a human reference AlignmentDocument.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from zetodos_thomas.align.paratext import filter_alignable_tokens
from zetodos_thomas.align.tokenize import whitespace_tokens
from zetodos_thomas.schemas.alignment import (
    AlignmentDocument,
    ParallelRelation,
    Token,
)
from zetodos_thomas.schemas.source import ParallelUnit


@dataclass(frozen=True)
class TokenLink:
    """One proposed or gold Coptic↔Greek token link within a unit."""

    unit_id: str
    left_token_id: str
    right_token_id: str
    left_norm: str
    right_norm: str
    score: float = 1.0


@dataclass
class LexicalBaselineModel:
    """Co-occurrence counts learned from train pairs only."""

    pair_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    left_counts: Counter[str] = field(default_factory=Counter)
    right_counts: Counter[str] = field(default_factory=Counter)
    train_unit_ids: list[str] = field(default_factory=list)

    def dice(self, left: str, right: str) -> float:
        """Dice coefficient for a normalized token pair."""
        co = self.pair_counts[(left, right)]
        if co == 0:
            return 0.0
        return (2.0 * co) / (self.left_counts[left] + self.right_counts[right])


@dataclass
class BenchmarkScores:
    """Precision/recall with explicit denominators (no invented pass bar)."""

    precision: float | None
    recall: float | None
    f1: float | None
    true_positives: int
    predicted: int
    gold: int
    unit_ids_scored: list[str]
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "predicted_denominator": self.predicted,
            "gold_denominator": self.gold,
            "unit_ids_scored": self.unit_ids_scored,
            "notes": self.notes,
        }


def _norm_tokens(unit: ParallelUnit) -> tuple[list[Token], list[Token]]:
    left = filter_alignable_tokens(whitespace_tokens(unit.sahidic))
    right = filter_alignable_tokens(whitespace_tokens(unit.greek))
    return left, right


def fit_lexical_baseline(train_units: list[ParallelUnit]) -> LexicalBaselineModel:
    """Fit co-occurrence Dice tables on nonempty train pairs.

    Uses verse-level type co-occurrence: each normalized token type is counted
    once per verse in the joint and marginal tables (avoids Dice > 1 from
    repeats). Intentionally elementary; not IBM Model 1 / eflomal.
    """
    model = LexicalBaselineModel()
    for unit in train_units:
        if not (
            unit.greek.original_text.strip() and unit.sahidic.original_text.strip()
        ):
            continue
        left, right = _norm_tokens(unit)
        if not left or not right:
            continue
        model.train_unit_ids.append(unit.unit_id)
        # Type-once-per-verse so joint and marginals use the same events.
        left_types = {t.normalized for t in left}
        right_types = {t.normalized for t in right}
        for ln in left_types:
            model.left_counts[ln] += 1
        for rn in right_types:
            model.right_counts[rn] += 1
        for ln in left_types:
            for rn in right_types:
                model.pair_counts[(ln, rn)] += 1
    return model


def predict_links(
    model: LexicalBaselineModel,
    units: list[ParallelUnit],
    *,
    min_dice: float = 0.05,
) -> list[TokenLink]:
    """Propose 1-best Greek token for each alignable Sahidic token."""
    links: list[TokenLink] = []
    for unit in units:
        if not (
            unit.greek.original_text.strip() and unit.sahidic.original_text.strip()
        ):
            continue
        left, right = _norm_tokens(unit)
        if not left or not right:
            continue
        # Track right-token usage lightly to reduce many-to-one collapse.
        used_right: set[str] = set()
        for lt in left:
            scored: list[tuple[float, Token]] = []
            for rt in right:
                score = model.dice(lt.normalized, rt.normalized)
                # Mild positional prior: prefer relative index proximity.
                if left and right:
                    li = lt.sequence / max(len(left), 1)
                    ri = rt.sequence / max(len(right), 1)
                    score = score + 0.01 * (1.0 - abs(li - ri))
                scored.append((score, rt))
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_rt = scored[0]
            if best_score < min_dice:
                continue
            # Prefer unused right token among near-ties.
            for score, candidate in scored:
                if score < min_dice:
                    break
                if candidate.id not in used_right:
                    best_score, best_rt = score, candidate
                    break
            used_right.add(best_rt.id)
            links.append(
                TokenLink(
                    unit_id=unit.unit_id,
                    left_token_id=lt.id,
                    right_token_id=best_rt.id,
                    left_norm=lt.normalized,
                    right_norm=best_rt.normalized,
                    score=best_score,
                )
            )
    return links


def gold_links_from_reference(doc: AlignmentDocument) -> list[TokenLink]:
    """Extract token-level gold links from lexical/phrase parallels.

    For phrase parallels, expands to the Cartesian product of left/right span
    tokens (conservative gold for scoring phrase fidelity separately).
    Lexical parallels with single-token spans become 1:1 links.
    """
    tokens = {t.id: t for t in doc.tokens}
    spans = {s.id: s for s in doc.spans}
    links: list[TokenLink] = []
    for parallel in doc.parallels:
        if parallel.relation not in {
            ParallelRelation.LEXICAL,
            ParallelRelation.PHRASE,
            ParallelRelation.LOANWORD,
        }:
            continue
        left_token_ids: list[str] = []
        right_token_ids: list[str] = []
        for sid in parallel.left_span_ids:
            span = spans.get(sid)
            if span is not None:
                left_token_ids.extend(span.token_ids)
        for sid in parallel.right_span_ids:
            span = spans.get(sid)
            if span is not None:
                right_token_ids.extend(span.token_ids)
        if not left_token_ids or not right_token_ids:
            continue
        # Prefer 1:1 zip when lengths match; else Cartesian for phrase bags.
        if len(left_token_ids) == len(right_token_ids) == 1 or (
            parallel.relation is ParallelRelation.LEXICAL
            and len(left_token_ids) == len(right_token_ids)
        ):
            pairs = list(zip(left_token_ids, right_token_ids, strict=True))
        elif parallel.relation is ParallelRelation.LEXICAL:
            pairs = list(zip(left_token_ids, right_token_ids, strict=False))
        else:
            pairs = [
                (left_id, right_id)
                for left_id in left_token_ids
                for right_id in right_token_ids
            ]
        for lid, rid in pairs:
            lt = tokens.get(lid)
            rt = tokens.get(rid)
            if lt is None or rt is None:
                continue
            links.append(
                TokenLink(
                    unit_id=lt.unit_id,
                    left_token_id=lid,
                    right_token_id=rid,
                    left_norm=lt.normalized,
                    right_norm=rt.normalized,
                )
            )
    return links


def score_links(
    predicted: list[TokenLink],
    gold: list[TokenLink],
    *,
    unit_ids: set[str] | None = None,
    restrict_to_annotated_left: bool = True,
) -> BenchmarkScores:
    """Token-link precision/recall with explicit denominators.

    When ``restrict_to_annotated_left`` is true (default), predictions are kept
    only for Sahidic token ids that appear on the left of at least one gold
    link in the filtered set. Partial gold therefore does not inflate false
    positives on unannotated tokens; precision measures link accuracy on
    annotated coverage, not annotation completeness.
    """

    def _key(link: TokenLink) -> tuple[str, str, str]:
        return (link.unit_id, link.left_token_id, link.right_token_id)

    gld = [g for g in gold if unit_ids is None or g.unit_id in unit_ids]
    pred = [p for p in predicted if unit_ids is None or p.unit_id in unit_ids]
    if restrict_to_annotated_left:
        annotated_left = {(g.unit_id, g.left_token_id) for g in gld}
        pred = [p for p in pred if (p.unit_id, p.left_token_id) in annotated_left]
    pred_set = {_key(p) for p in pred}
    gold_set = {_key(g) for g in gld}
    tp = len(pred_set & gold_set)
    precision = (tp / len(pred_set)) if pred_set else None
    recall = (tp / len(gold_set)) if gold_set else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        # Both denominators nonempty but zero matches → defined zero F1.
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    scored_units = sorted({*([p.unit_id for p in pred]), *([g.unit_id for g in gld])})
    coverage_note = (
        "Predictions restricted to gold-annotated left (Sahidic) token ids "
        "so partial gold does not conflate coverage with precision. "
        if restrict_to_annotated_left
        else ""
    )
    return BenchmarkScores(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        predicted=len(pred_set),
        gold=len(gold_set),
        unit_ids_scored=scored_units,
        notes=(
            "No pass/fail threshold. Denominators are predicted link count and "
            "gold link count after optional unit_id filter. "
            f"{coverage_note}"
            "Omission fixtures must not appear in these denominators. "
            "F1 is 0.0 when both denominators are nonzero and TP=0."
        ),
    )


def null_alignment_rate(
    units: list[ParallelUnit], links: list[TokenLink]
) -> dict[str, object]:
    """Share of alignable Sahidic tokens left without a predicted link."""
    by_unit: dict[str, set[str]] = defaultdict(set)
    for link in links:
        by_unit[link.unit_id].add(link.left_token_id)
    total = 0
    nulls = 0
    for unit in units:
        if not (
            unit.greek.original_text.strip() and unit.sahidic.original_text.strip()
        ):
            continue
        left, _right = _norm_tokens(unit)
        total += len(left)
        linked = by_unit.get(unit.unit_id, set())
        nulls += sum(1 for t in left if t.id not in linked)
    return {
        "alignable_sahidic_tokens": total,
        "null_sahidic_tokens": nulls,
        "null_rate": (nulls / total) if total else None,
    }


def reference_parallel_inventory(doc: AlignmentDocument) -> dict[str, int]:
    """Count parallel relations in a human reference document."""
    counts: Counter[str] = Counter()
    for parallel in doc.parallels:
        counts[parallel.relation.value] += 1
    return dict(counts)


def assert_no_omission_in_links(
    links: list[TokenLink],
    omission_unit_ids: set[str],
) -> list[str]:
    """Return problems if omission fixtures leaked into positive link sets."""
    return [
        (
            f"{link.unit_id} in omission fixture but has link "
            f"{link.left_norm!r}<->{link.right_norm!r}"
        )
        for link in links
        if link.unit_id in omission_unit_ids
    ]

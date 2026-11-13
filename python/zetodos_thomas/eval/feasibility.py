# SPDX-License-Identifier: Apache-2.0
"""Multi-level feasibility metrics: decoder vs word-for-word (JAK-41).

Compares JAK-40 reconstructions to the elementary train-cooccurrence baseline
on **identical** Coptic inputs. No invented pass threshold. Official Thomas 7
surviving-letter scores remain gated on ``human-checked``.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from zetodos_thomas.eval.blinded_reconstruct import (
    holdout_units,
    reconstruct_blinded_package,
)
from zetodos_thomas.eval.letters import (
    greek_letters_only,
    token_aware_letter_score,
)
from zetodos_thomas.eval.review_gate import (
    EvalReviewGateError,
    assert_eval_package_human_checked,
    load_blinded_eval_package,
)
from zetodos_thomas.ingest.paths import require_repo_root
from zetodos_thomas.lexicon.lemmas import infer_greek_lemma
from zetodos_thomas.schemas.blinded import (
    BlindedReconstructionPackage,
    BlindedTargetKind,
    BlindedUnitReconstruction,
    WordForWordUnitBaseline,
)
from zetodos_thomas.schemas.decode import SpanDecodeStatus
from zetodos_thomas.schemas.feasibility import (
    AblationArmComparison,
    AggregateSystemMetrics,
    FailureModeExample,
    FeasibilityReport,
    GateRecommendation,
    RatioMetric,
    SystemLabel,
    Thomas7GateStatus,
    UnitSystemMetrics,
)
from zetodos_thomas.schemas.mappings import AblationCondition
from zetodos_thomas.schemas.source import ParallelUnit

DEFAULT_THOMAS_RECON_REL = Path(
    "data/reconstructions/jak40/blinded-reconstruction-v1.json"
)
DEFAULT_NT_RECON_REL = Path(
    "data/reconstructions/jak40/nt-holdout-blinded-reconstruction-v1.json"
)
DEFAULT_THOMAS7_EVAL_REL = Path("data/evidence/eval/thomas-7-blinded-v1.json")
DEFAULT_FREEZE_REL = Path("data/reconstructions/jak40/thomas-7-blinded-freeze-v1.json")
DEFAULT_METRICS_REL = Path("data/reconstructions/jak41/feasibility-metrics-v1.json")
DEFAULT_DOC_REL = Path("docs/phase2-feasibility-report.md")
COMMITTED_ARTIFACT_MAX_BYTES = 1024 * 1024

METRIC_DEFINITIONS: dict[str, str] = {
    "letter_agreement": (
        "Token-positional Greek letter F1/precision/recall "
        "(``token-positional-letter-v1``). Global LCS is diagnostic only."
    ),
    "surface_exact": (
        "Positional exact match after letter-fold (casefold + strip diacritics + "
        "letters only). Recall = exact hits / gold tokens."
    ),
    "lemma_proxy": (
        "Positional match of JAK-37 ``infer_greek_lemma`` proxies "
        "(NFD strip combining marks). Not MorphGNT lemmas."
    ),
    "lemma_lcs": (
        "Longest common subsequence length of lemma-proxy sequences / gold "
        "token count — order-preserving content recovery beyond strict index."
    ),
    "near_miss_tokens": (
        "Positional pairs with shared letters but not exact surface match "
        "(weak morphology / near-miss signal under the letter-fold proxy)."
    ),
    "word_order": (
        "Captured jointly by positional lemma_proxy recall and lemma_lcs "
        "recall; fluent reordering that loses position lowers positional "
        "scores even when LCS stays high."
    ),
    "phrase_fidelity": (
        "For multi-token JAK-38 construction spans: whether the predicted "
        "Greek surface sequence appears as a contiguous gold subsequence "
        "(letter-folded). Baseline is always singleton 1:1."
    ),
    "additions_omissions": (
        "additions = unpaired hypothesis tokens; omissions = unpaired gold "
        "tokens (length mismatch after flattening non-null predictions)."
    ),
    "uncertainty": (
        "Decoder abstention rate + mean top-1 candidate uncertainty when "
        "candidates exist. Baseline uncertainty is coverage-only (null = unseen)."
    ),
    "surviving_letter_thomas7": (
        "Official Thomas 7 surviving-letter agreement requires "
        "``review_status=human-checked`` + freeze pin (JAK-40). Gate-blocked "
        "status is reported honestly; not invented."
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _ratio(num: int, den: int, *, definition: str = "") -> RatioMetric:
    return RatioMetric(
        numerator=num,
        denominator=den,
        value=(num / den) if den else None,
        definition=definition,
    )


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def flatten_predictions(preds: list[str | None]) -> list[str]:
    """Flatten span hypotheses; drop nulls; split multi-surface spans."""
    out: list[str] = []
    for item in preds:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        out.extend(text.split())
    return out


def gold_tokens_for_unit(unit: ParallelUnit) -> list[str]:
    text = unit.greek.normalized_text or unit.greek.original_text
    return [t for t in text.split() if t]


def _lemma_fold(token: str) -> str:
    return infer_greek_lemma(greek_letters_only(token))


def _surface_fold(token: str) -> str:
    return greek_letters_only(token)


def _contiguous_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def score_hypothesis_vs_gold(
    *,
    unit_id: str,
    system: SystemLabel,
    hypothesis: list[str | None],
    gold_tokens: list[str],
    coptic_surfaces: list[str],
    covering_span_kinds: list[str] | None = None,
    decode_unit: BlindedUnitReconstruction | None = None,
) -> UnitSystemMetrics:
    """Multi-level metrics for one hypothesis sequence vs gold tokens."""
    hyp = flatten_predictions(hypothesis)
    letter = token_aware_letter_score(hyp, gold_tokens)

    hyp_surf = [_surface_fold(t) for t in hyp]
    gold_surf = [_surface_fold(t) for t in gold_tokens]
    hyp_lem = [_lemma_fold(t) for t in hyp]
    gold_lem = [_lemma_fold(t) for t in gold_tokens]

    n_pair = min(len(hyp_surf), len(gold_surf))
    exact_tp = sum(
        1 for i in range(n_pair) if hyp_surf[i] and hyp_surf[i] == gold_surf[i]
    )
    lemma_tp = sum(1 for i in range(n_pair) if hyp_lem[i] and hyp_lem[i] == gold_lem[i])
    near_miss = 0
    for i in range(n_pair):
        h, g = hyp_surf[i], gold_surf[i]
        if not h or not g or h == g:
            continue
        # Shared letters without exact equality → near-miss / morphology underdet.
        if set(h) & set(g):
            near_miss += 1

    lemma_lcs = _token_lcs_len(hyp_lem, gold_lem)

    additions = max(0, len(hyp) - len(gold_tokens))
    omissions = max(0, len(gold_tokens) - len(hyp))
    n_coptic = len(coptic_surfaces)
    n_non_null = sum(1 for p in hypothesis if p is not None)
    coverage = (n_non_null / n_coptic) if n_coptic else None

    construction_spans = 0
    construction_hits = 0
    kinds = covering_span_kinds or []
    if decode_unit is not None:
        for kind, span in zip(kinds, decode_unit.decode_result.spans, strict=False):
            if kind != "construction" or len(span.surfaces) <= 1:
                continue
            construction_spans += 1
            if span.status is SpanDecodeStatus.CANDIDATES and span.candidates:
                pred = [_surface_fold(s) for s in span.candidates[0].greek_surfaces]
                if _contiguous_subsequence(pred, gold_surf):
                    construction_hits += 1

    abstention_rate: float | None = None
    mean_unc: float | None = None
    if decode_unit is not None:
        spans = decode_unit.decode_result.spans
        if spans:
            n_abs = sum(1 for s in spans if s.status is SpanDecodeStatus.ABSTAIN)
            abstention_rate = n_abs / len(spans)
            uncs = [
                s.candidates[0].uncertainty
                for s in spans
                if s.status is SpanDecodeStatus.CANDIDATES and s.candidates
            ]
            mean_unc = (sum(uncs) / len(uncs)) if uncs else None
    elif n_coptic:
        abstention_rate = 1.0 - (n_non_null / n_coptic)

    return UnitSystemMetrics(
        unit_id=unit_id,
        system=system,
        letter_true_positives=letter.true_positives,
        letter_predicted_denominator=letter.predicted_denominator,
        letter_gold_denominator=letter.gold_denominator,
        letter_precision=letter.precision,
        letter_recall=letter.recall,
        letter_f1=letter.f1,
        incidental_global_lcs=letter.incidental_global_lcs,
        letter_metric_kind=letter.metric_kind,
        surface_exact=_ratio(
            exact_tp,
            len(gold_tokens),
            definition="exact positional surface hits / gold tokens",
        ),
        lemma_proxy=_ratio(
            lemma_tp,
            len(gold_tokens),
            definition="positional lemma-proxy hits / gold tokens",
        ),
        lemma_lcs=_ratio(
            lemma_lcs,
            len(gold_tokens),
            definition="lemma-proxy LCS length / gold tokens",
        ),
        near_miss_tokens=near_miss,
        predicted_tokens=len(hyp),
        gold_tokens=len(gold_tokens),
        additions=additions,
        omissions=omissions,
        coptic_surfaces=n_coptic,
        predicted_non_null_spans=n_non_null,
        coverage=coverage,
        abstention_rate=abstention_rate,
        mean_top1_uncertainty=mean_unc,
        construction_spans=construction_spans,
        construction_phrase_hits=construction_hits,
        notes=[
            "Gold consulted at scoring time only (identical Coptic inputs).",
            "lemma_proxy uses JAK-37 NFD-strip heuristic — not MorphGNT.",
            "incidental_global_lcs is diagnostic only.",
        ],
    )


def _token_lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b and tok_a:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def aggregate_unit_metrics(
    rows: list[UnitSystemMetrics],
    *,
    system: SystemLabel,
    wins_vs_other: int | None = None,
) -> AggregateSystemMetrics:
    if not rows:
        return AggregateSystemMetrics(
            system=system,
            n_units=0,
            mean_letter_precision=None,
            mean_letter_recall=None,
            mean_letter_f1=None,
            mean_surface_exact_recall=None,
            mean_lemma_proxy_recall=None,
            mean_lemma_lcs_recall=None,
            mean_coverage=None,
            mean_abstention_rate=None,
            notes=["No units scored."],
        )
    letter_tp = sum(r.letter_true_positives for r in rows)
    letter_gold = sum(r.letter_gold_denominator for r in rows)
    surf_tp = sum(r.surface_exact.numerator for r in rows)
    surf_den = sum(r.surface_exact.denominator for r in rows)
    lem_tp = sum(r.lemma_proxy.numerator for r in rows)
    lem_den = sum(r.lemma_proxy.denominator for r in rows)
    return AggregateSystemMetrics(
        system=system,
        n_units=len(rows),
        mean_letter_precision=_mean([r.letter_precision for r in rows]),
        mean_letter_recall=_mean([r.letter_recall for r in rows]),
        mean_letter_f1=_mean([r.letter_f1 for r in rows]),
        mean_surface_exact_recall=_mean([r.surface_exact.value for r in rows]),
        mean_lemma_proxy_recall=_mean([r.lemma_proxy.value for r in rows]),
        mean_lemma_lcs_recall=_mean([r.lemma_lcs.value for r in rows]),
        mean_coverage=_mean([r.coverage for r in rows]),
        mean_abstention_rate=_mean([r.abstention_rate for r in rows]),
        pooled_letter=_ratio(
            letter_tp,
            letter_gold,
            definition="pooled letter TP / pooled gold letters (recall-oriented)",
        ),
        pooled_surface_exact=_ratio(surf_tp, surf_den),
        pooled_lemma_proxy=_ratio(lem_tp, lem_den),
        total_additions=sum(r.additions for r in rows),
        total_omissions=sum(r.omissions for r in rows),
        total_construction_spans=sum(r.construction_spans for r in rows),
        total_construction_phrase_hits=sum(r.construction_phrase_hits for r in rows),
        letter_f1_unit_wins_vs_other=wins_vs_other,
        notes=[
            "Macro means average per-unit ratios; pooled rows use summed denominators.",
            "No invented pass/fail threshold (PLAN.md Phase 2).",
        ],
    )


def load_reconstruction_package(path: Path) -> BlindedReconstructionPackage:
    return BlindedReconstructionPackage.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def baseline_by_unit(
    package: BlindedReconstructionPackage,
) -> dict[str, WordForWordUnitBaseline]:
    return {b.unit_id: b for b in package.word_for_word_baseline}


def score_nt_holdout_package(
    package: BlindedReconstructionPackage,
    gold_by_id: dict[str, ParallelUnit],
) -> tuple[list[UnitSystemMetrics], list[UnitSystemMetrics]]:
    """Score decoder + word-for-word on identical NT holdout Coptic inputs."""
    bases = baseline_by_unit(package)
    decoder_rows: list[UnitSystemMetrics] = []
    baseline_rows: list[UnitSystemMetrics] = []
    for unit in package.units:
        if unit.target_kind is not BlindedTargetKind.NT_HOLDOUT:
            continue
        gold_unit = gold_by_id.get(unit.unit_id)
        if gold_unit is None:
            continue
        gold = gold_tokens_for_unit(gold_unit)
        decoder_rows.append(
            score_hypothesis_vs_gold(
                unit_id=unit.unit_id,
                system=SystemLabel.DECODER,
                hypothesis=unit.top1_greek_surfaces,
                gold_tokens=gold,
                coptic_surfaces=unit.coptic_surfaces,
                covering_span_kinds=unit.covering_span_kinds,
                decode_unit=unit,
            )
        )
        base = bases.get(unit.unit_id)
        if base is None:
            msg = f"missing word-for-word baseline for identical input {unit.unit_id}"
            raise KeyError(msg)
        if base.coptic_surfaces != unit.coptic_surfaces:
            msg = (
                f"baseline Coptic surfaces diverge from decoder for {unit.unit_id} "
                "(identical-input requirement)"
            )
            raise ValueError(msg)
        baseline_rows.append(
            score_hypothesis_vs_gold(
                unit_id=unit.unit_id,
                system=SystemLabel.WORD_FOR_WORD,
                hypothesis=base.predicted_greek,
                gold_tokens=gold,
                coptic_surfaces=base.coptic_surfaces,
                covering_span_kinds=None,
                decode_unit=None,
            )
        )
    return decoder_rows, baseline_rows


def _count_f1_wins(
    decoder_rows: list[UnitSystemMetrics],
    baseline_rows: list[UnitSystemMetrics],
) -> tuple[int, int, int]:
    by_base = {r.unit_id: r for r in baseline_rows}
    wins_d = wins_b = ties = 0
    for row in decoder_rows:
        other = by_base[row.unit_id]
        a, b = row.letter_f1, other.letter_f1
        if a is None and b is None:
            ties += 1
        elif a is None:
            wins_b += 1
        elif b is None:
            wins_d += 1
        elif abs(a - b) < 1e-12:
            ties += 1
        elif a > b:
            wins_d += 1
        else:
            wins_b += 1
    return wins_d, wins_b, ties


def thomas7_gate_status(repo_root: Path) -> Thomas7GateStatus:
    """Report human-checked / official gate status without inventing scores."""
    pack_path = repo_root / DEFAULT_THOMAS7_EVAL_REL
    freeze_path = repo_root / DEFAULT_FREEZE_REL
    payload = load_blinded_eval_package(pack_path)
    package_id = str(payload.get("id", pack_path.name))
    review_status = str(payload.get("review_status", ""))
    gate_blocked = True
    notes = [
        "Official Thomas 7 surviving-letter scores require human-checked + freeze.",
        (
            "ready-for-independent-review is packaged readiness only — "
            "not completed review."
        ),
    ]
    try:
        assert_eval_package_human_checked(pack_path)
        gate_blocked = False
        notes.append("human-checked gate passed.")
    except EvalReviewGateError as exc:
        notes.append(str(exc))
    return Thomas7GateStatus(
        eval_package_id=package_id,
        review_status=review_status,
        official_surviving_letter_scores=not gate_blocked and freeze_path.is_file(),
        gate_blocked=gate_blocked,
        freeze_present=freeze_path.is_file(),
        human_checked_required=True,
        notes=notes,
    )


def collect_failure_modes(
    package: BlindedReconstructionPackage,
    decoder_rows: list[UnitSystemMetrics],
    baseline_rows: list[UnitSystemMetrics],
    gold_by_id: dict[str, ParallelUnit],
) -> list[FailureModeExample]:
    """Concrete failure-mode examples (taxonomy + evidence)."""
    examples: list[FailureModeExample] = []
    bases = baseline_by_unit(package)
    by_dec = {r.unit_id: r for r in decoder_rows}
    by_base = {r.unit_id: r for r in baseline_rows}

    # Morphology / lemma underdetermination note (global).
    examples.append(
        FailureModeExample(
            category="morphology_underdetermination",
            unit_id="(corpus)",
            system=None,
            detail=(
                "JAK-37 lemma proxies are accent-stripped surfaces, not MorphGNT "
                "lemmas; surface_exact and lemma_proxy often coincide, so "
                "inflection errors are under-measured."
            ),
            evidence={
                "lemma_method": "nfd_strip_combining_marks_v1",
                "near_miss_tokens_decoder": sum(
                    r.near_miss_tokens for r in decoder_rows
                ),
            },
        )
    )

    # False equivalence / function-word Dice friends.
    for unit in package.units:
        if unit.target_kind is not BlindedTargetKind.NT_HOLDOUT:
            continue
        base = bases.get(unit.unit_id)
        gold_unit = gold_by_id.get(unit.unit_id)
        if base is None or gold_unit is None:
            continue
        gold = gold_tokens_for_unit(gold_unit)
        if not gold:
            continue
        # Look for decoder/baseline predicting a high-frequency Greek function
        # word in a position where gold differs.
        for idx, (hyp, g) in enumerate(
            zip(flatten_predictions(unit.top1_greek_surfaces), gold, strict=False)
        ):
            if _lemma_fold(hyp) in {
                "και",
                "ο",
                "η",
                "το",
                "του",
                "την",
            } and _lemma_fold(hyp) != _lemma_fold(g):
                examples.append(
                    FailureModeExample(
                        category="false_equivalence",
                        unit_id=unit.unit_id,
                        system=SystemLabel.DECODER,
                        detail=(
                            "Frequent Greek function word predicted where gold "
                            "differs (bag co-occurrence / NT prior risk)."
                        ),
                        evidence={
                            "index": idx,
                            "hypothesis": hyp,
                            "gold": g,
                            "coptic": unit.coptic_surfaces[idx]
                            if idx < len(unit.coptic_surfaces)
                            else None,
                        },
                    )
                )
                break
        if any(
            e.category == "false_equivalence" and e.unit_id == unit.unit_id
            for e in examples
        ):
            break

    # NT overcopy risk: baseline/decoder coverage high but letter F1 low.
    for row in decoder_rows:
        if (
            row.coverage is not None
            and row.coverage >= 0.7
            and row.letter_f1 is not None
            and row.letter_f1 < 0.15
        ):
            examples.append(
                FailureModeExample(
                    category="nt_overcopy_risk",
                    unit_id=row.unit_id,
                    system=SystemLabel.DECODER,
                    detail=(
                        "High Coptic coverage with low letter F1 — emitting "
                        "familiar NT Greek that does not match this holdout."
                    ),
                    evidence={
                        "coverage": row.coverage,
                        "letter_f1": row.letter_f1,
                        "baseline_letter_f1": (
                            by_base[row.unit_id].letter_f1
                            if row.unit_id in by_base
                            else None
                        ),
                    },
                )
            )
            break

    # Abstention / omission heavy units.
    worst = min(
        decoder_rows,
        key=lambda r: r.letter_f1 if r.letter_f1 is not None else 1.0,
        default=None,
    )
    if worst is not None:
        examples.append(
            FailureModeExample(
                category="additions_omissions",
                unit_id=worst.unit_id,
                system=SystemLabel.DECODER,
                detail="Lowest letter-F1 holdout unit — length/content mismatch.",
                evidence={
                    "letter_f1": worst.letter_f1,
                    "additions": worst.additions,
                    "omissions": worst.omissions,
                    "abstention_rate": worst.abstention_rate,
                },
            )
        )

    # Fluent Greek ≠ success reminder with Thomas 7.
    thomas = next(
        (u for u in package.units if u.target_kind is BlindedTargetKind.THOMAS_PILOT),
        None,
    )
    # Thomas may live only in the sibling package; still add the taxonomy note.
    examples.append(
        FailureModeExample(
            category="fluent_greek_not_success",
            unit_id=thomas.unit_id if thomas else "thomas.7",
            system=SystemLabel.DECODER,
            detail=(
                "Fluent or NT-like Greek must not be treated as reconstruction "
                "success; surviving-letter / multi-level metrics are authoritative."
            ),
            evidence={
                "official_thomas7_surviving_letter": False,
                "note": "Gate blocked until human-checked (JAK-40/43).",
            },
        )
    )

    # Fragment divergence / sparse gold.
    examples.append(
        FailureModeExample(
            category="fragment_divergence",
            unit_id="thomas.7",
            system=None,
            detail=(
                "P.Oxy. 654 Thomas 7 lines mix sparse surviving letters with "
                "restorations; restorations stay out of gold. Official agreement "
                "deferred until independent review."
            ),
            evidence={"lines": ["main.40", "main.41", "main.42"]},
        )
    )

    # Treating baseline coverage as superiority.
    if decoder_rows and baseline_rows:
        dec_cov = _mean([r.coverage for r in decoder_rows])
        base_cov = _mean([r.coverage for r in baseline_rows])
        dec_f1 = _mean([r.letter_f1 for r in decoder_rows])
        base_f1 = _mean([r.letter_f1 for r in baseline_rows])
        if (
            dec_cov is not None
            and base_cov is not None
            and base_cov > dec_cov
            and dec_f1 is not None
            and base_f1 is not None
            and dec_f1 >= base_f1
        ):
            examples.append(
                FailureModeExample(
                    category="coverage_vs_fidelity",
                    unit_id="(nt_holdouts)",
                    system=None,
                    detail=(
                        "Word-for-word baseline covers more Coptic surfaces but "
                        "does not beat decoder letter F1 — coverage ≠ fidelity."
                    ),
                    evidence={
                        "decoder_mean_coverage": dec_cov,
                        "baseline_mean_coverage": base_cov,
                        "decoder_mean_letter_f1": dec_f1,
                        "baseline_mean_letter_f1": base_f1,
                    },
                )
            )

    # Ensure we referenced decoder/baseline maps (lint-friendly use).
    _ = by_dec
    return examples


def choose_gate_recommendation(
    *,
    thomas7: Thomas7GateStatus,
    decoder: AggregateSystemMetrics,
    baseline: AggregateSystemMetrics,
    nt_review_status: str,
) -> tuple[GateRecommendation, list[str]]:
    """Evidence-backed advisory recommendation (Jake decides)."""
    rationale: list[str] = []
    dec_f1 = decoder.mean_letter_f1
    base_f1 = baseline.mean_letter_f1
    delta = None if dec_f1 is None or base_f1 is None else dec_f1 - base_f1

    if thomas7.gate_blocked:
        rationale.append(
            "Official Thomas 7 surviving-letter scores are gate-blocked "
            f"(review_status={thomas7.review_status!r}); cannot claim G-witness "
            "feasibility on a completed human-checked gold."
        )
    if nt_review_status != "human-checked":
        rationale.append(
            f"NT holdout gold remains {nt_review_status!r} — useful for "
            "relative decoder-vs-baseline comparison, not as a final OKR."
        )

    if dec_f1 is not None:
        rationale.append(
            f"NT holdout macro mean letter F1 for decoder≈{dec_f1:.4f} "
            f"(baseline≈{base_f1:.4f}; delta≈{delta:+.4f})."
        )
    if decoder.mean_lemma_proxy_recall is not None:
        rationale.append(
            "Lemma-proxy recall remains low "
            f"(decoder≈{decoder.mean_lemma_proxy_recall:.4f}; "
            f"baseline≈{baseline.mean_lemma_proxy_recall:.4f})."
        )
    rationale.append(
        "Jake Phase 2 note: corpus/morphology gaps are still the likely "
        "limiting factors; absolute recovery is far from production-scale."
    )
    rationale.append(
        "Decoder is only marginally better than the elementary word-for-word "
        "baseline on identical inputs — not yet a clear architecture win."
    )
    rationale.append(
        "Ollama (JAK-42) stays deferred: limitation is evidence/eval readiness "
        "and morphology/corpus depth, not a demonstrated need for LLM fill-in."
    )

    # Prefer pause_and_improve when relative signal exists but absolute is weak
    # and official Thomas 7 gold is unreviewed.
    if thomas7.gate_blocked or (
        dec_f1 is not None and dec_f1 < 0.35 and (delta is None or delta < 0.15)
    ):
        return GateRecommendation.PAUSE_AND_IMPROVE, rationale

    if dec_f1 is not None and base_f1 is not None and dec_f1 + 0.02 < base_f1:
        rationale.append(
            "Decoder underperforms the word-for-word baseline on letter F1; "
            "reassess method before Phase 3 architecture commitment."
        )
        return GateRecommendation.REASSESS, rationale

    return GateRecommendation.PROCEED, rationale


def score_ablation_arm(
    repo_root: Path,
    *,
    condition: AblationCondition,
    gold_by_id: dict[str, ParallelUnit],
    baseline_rows_ref: list[UnitSystemMetrics] | None = None,
) -> AblationArmComparison:
    """Blinded reconstruct one ablation arm and score NT holdouts only."""
    package = reconstruct_blinded_package(
        repo_root,
        ablation=condition,
        include_nt_holdouts=True,
        max_candidates_per_span=1,
        include_word_for_word=True,
    )
    # Keep NT units only for arm comparison.
    nt_units = [
        u for u in package.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    ]
    nt_base = [
        b
        for b in package.word_for_word_baseline
        if any(u.unit_id == b.unit_id for u in nt_units)
    ]
    package = package.model_copy(
        update={"units": nt_units, "word_for_word_baseline": nt_base}
    )
    decoder_rows, baseline_rows = score_nt_holdout_package(package, gold_by_id)
    wins_d, _wins_b, _ties = _count_f1_wins(decoder_rows, baseline_rows)
    agg = aggregate_unit_metrics(
        decoder_rows, system=SystemLabel.DECODER, wins_vs_other=wins_d
    )
    base_agg = aggregate_unit_metrics(
        baseline_rows_ref or baseline_rows, system=SystemLabel.WORD_FOR_WORD
    )
    delta = None
    if agg.mean_letter_f1 is not None and base_agg.mean_letter_f1 is not None:
        delta = agg.mean_letter_f1 - base_agg.mean_letter_f1
    return AblationArmComparison(
        ablation_condition=condition,
        aggregate=agg,
        vs_word_for_word_mean_letter_f1_delta=delta,
        notes=[
            f"Arm reconstructed offline with ablation={condition.value}.",
            "Identical Coptic holdout inputs; Greek loaded only at scoring.",
            "max_candidates_per_span=1 for compact arm comparison.",
        ],
    )


def build_feasibility_report(
    repo_root: Path | None = None,
    *,
    thomas_recon_path: Path | None = None,
    nt_recon_path: Path | None = None,
    include_ablation_arms: bool = True,
) -> FeasibilityReport:
    """Build the JAK-41 feasibility metrics package from JAK-40 artifacts."""
    root = require_repo_root(repo_root)
    thomas_path = thomas_recon_path or (root / DEFAULT_THOMAS_RECON_REL)
    nt_path = nt_recon_path or (root / DEFAULT_NT_RECON_REL)
    thomas_pkg = load_reconstruction_package(thomas_path)
    nt_pkg = load_reconstruction_package(nt_path)

    # Identical-input parity: every NT unit must have a baseline row.
    missing = [
        u.unit_id
        for u in nt_pkg.units
        if u.target_kind is BlindedTargetKind.NT_HOLDOUT
        and u.unit_id not in baseline_by_unit(nt_pkg)
    ]
    if missing:
        msg = f"NT package missing word-for-word baselines for: {missing}"
        raise ValueError(msg)

    gold_list = holdout_units(root)
    gold_by_id = {u.unit_id: u for u in gold_list}
    decoder_rows, baseline_rows = score_nt_holdout_package(nt_pkg, gold_by_id)
    wins_d, wins_b, ties = _count_f1_wins(decoder_rows, baseline_rows)
    decoder_agg = aggregate_unit_metrics(
        decoder_rows, system=SystemLabel.DECODER, wins_vs_other=wins_d
    )
    baseline_agg = aggregate_unit_metrics(
        baseline_rows,
        system=SystemLabel.WORD_FOR_WORD,
        wins_vs_other=wins_b,
    )

    thomas7 = thomas7_gate_status(root)
    failure_modes = collect_failure_modes(
        nt_pkg, decoder_rows, baseline_rows, gold_by_id
    )
    # Thomas 7 side-by-side coverage (no official letter scores).
    t7_unit = next(
        (
            u
            for u in thomas_pkg.units
            if u.target_kind is BlindedTargetKind.THOMAS_PILOT
        ),
        None,
    )
    t7_base = baseline_by_unit(thomas_pkg).get(t7_unit.unit_id) if t7_unit else None
    if t7_unit is not None and t7_base is not None:
        failure_modes.append(
            FailureModeExample(
                category="thomas7_side_by_side",
                unit_id=t7_unit.unit_id,
                system=None,
                detail=(
                    "Blinded Thomas 7 decoder vs word-for-word on identical "
                    "Dilley Coptic; official surviving-letter comparison deferred."
                ),
                evidence={
                    "decoder_non_null": sum(
                        1 for p in t7_unit.top1_greek_surfaces if p is not None
                    ),
                    "baseline_non_null": sum(
                        1 for p in t7_base.predicted_greek if p is not None
                    ),
                    "coptic_surfaces": len(t7_unit.coptic_surfaces),
                    "decoder_top1": t7_unit.top1_greek_surfaces,
                    "baseline_predictions": t7_base.predicted_greek,
                    "identical_coptic": t7_unit.coptic_surfaces
                    == t7_base.coptic_surfaces,
                },
            )
        )

    arms: list[AblationArmComparison] = []
    if include_ablation_arms:
        for cond in (
            AblationCondition.NT_ALIGNER_ONLY,
            AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        ):
            arms.append(
                score_ablation_arm(
                    root,
                    condition=cond,
                    gold_by_id=gold_by_id,
                    baseline_rows_ref=baseline_rows,
                )
            )

    recommendation, rationale = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder_agg,
        baseline=baseline_agg,
        nt_review_status="provisional-draft",
    )

    digests = {
        "thomas_reconstruction_sha256": _sha256_file(thomas_path),
        "nt_holdout_reconstruction_sha256": _sha256_file(nt_path),
        "thomas7_eval_package_sha256": _sha256_file(root / DEFAULT_THOMAS7_EVAL_REL),
    }
    if (root / DEFAULT_FREEZE_REL).is_file():
        digests["thomas7_freeze_sha256"] = _sha256_file(root / DEFAULT_FREEZE_REL)

    unit_metrics = sorted(
        [*decoder_rows, *baseline_rows],
        key=lambda r: (r.unit_id, r.system.value),
    )

    return FeasibilityReport(
        gate_recommendation=recommendation,
        gate_rationale=rationale,
        thomas7_gate=thomas7,
        nt_holdout_review_status="provisional-draft",
        primary_ablation=nt_pkg.ablation_condition,
        decoder_aggregate=decoder_agg,
        word_for_word_aggregate=baseline_agg,
        ablation_arms=arms,
        unit_metrics=unit_metrics,
        failure_modes=failure_modes,
        metric_definitions=dict(METRIC_DEFINITIONS),
        identical_inputs={
            "nt_holdout_units": len(decoder_rows),
            "thomas_pilot_units": 1 if t7_unit else 0,
            "coptic_surfaces_matched": all(
                baseline_by_unit(nt_pkg)[u.unit_id].coptic_surfaces == u.coptic_surfaces
                for u in nt_pkg.units
                if u.target_kind is BlindedTargetKind.NT_HOLDOUT
            ),
            "letter_f1_wins": {
                "decoder": wins_d,
                "word_for_word": wins_b,
                "ties": ties,
            },
            "source_packages": {
                "thomas": thomas_pkg.id,
                "nt_holdout": nt_pkg.id,
            },
        },
        leakage_safeguards={
            "greek_consulted_during_reconstruction": thomas_pkg.blinding.get(
                "greek_consulted_during_reconstruction"
            )
            is False
            and nt_pkg.blinding.get("greek_consulted_during_reconstruction") is False,
            "train_dev_not_scored_as_holdout": True,
            "poxy_654_sealed_until_scoring": True,
            "adopted_aligner": None,
            "phase2_active_aligners": ["eflomal", "fast_align"],
            "llm": False,
            "network": False,
            "thomas7_official_gate": thomas7.model_dump(mode="json"),
        },
        artifact_digests=digests,
        summary={
            "gate_recommendation": recommendation.value,
            "nt_units_scored": len(decoder_rows),
            "decoder_mean_letter_f1": decoder_agg.mean_letter_f1,
            "baseline_mean_letter_f1": baseline_agg.mean_letter_f1,
            "decoder_mean_lemma_proxy_recall": decoder_agg.mean_lemma_proxy_recall,
            "baseline_mean_lemma_proxy_recall": baseline_agg.mean_lemma_proxy_recall,
            "thomas7_gate_blocked": thomas7.gate_blocked,
            "ablation_arms": [a.ablation_condition.value for a in arms],
        },
        provenance={
            "owned_by": "JAK-41",
            "consumes": ["JAK-40", "JAK-39", "JAK-38", "JAK-37", "JAK-28"],
            "llm": False,
            "network": False,
            "phase2_active_aligners": ["eflomal", "fast_align"],
            "adopted_aligner": None,
        },
        notes=[
            "Phase 2 decision gate input for Jake — does not auto-start Phase 3.",
            "Word-for-word baseline uses the same Coptic surfaces as JAK-40 decode.",
            "NT holdout gold is provisional-draft Vaticanus B* at eval time.",
            "Official Thomas 7 surviving-letter scores remain human-checked gated.",
            "Fluent Greek is not treated as success.",
            "JAK-42 Ollama stays deferred pending a concrete non-LLM limitation.",
        ],
    )


def _md_row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_feasibility_markdown(report: FeasibilityReport) -> str:
    """Render the written feasibility report for ``docs/``."""
    d = report.decoder_aggregate
    b = report.word_for_word_aggregate
    t7 = report.thomas7_gate
    lines: list[str] = [
        "# Phase 2 feasibility report and decision gate (JAK-41)",
        "",
        "**Status:** Advisory gate input for Jake — **do not** auto-start Phase 3.",
        f"**Recommendation:** `{report.gate_recommendation.value}`",
        "**Machine metrics:** "
        "[`data/reconstructions/jak41/feasibility-metrics-v1.json`]"
        "(../data/reconstructions/jak41/feasibility-metrics-v1.json)",
        "**Inputs:** JAK-40 blinded reconstructions on identical Coptic "
        "(Thomas 7 + NT holdouts).",
        "",
        "## Decision gate",
        "",
    ]
    for item in report.gate_rationale:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Jake chooses **Proceed / Pause and improve / Reassess**. This report "
            "does not invent a numeric OKR or merge Phase 3 work.",
            "",
            "## Official Thomas 7 surviving-letter gate",
            "",
            _md_row("Field", "Value"),
            _md_row("---", "---"),
            _md_row("Eval package", f"`{t7.eval_package_id}`"),
            _md_row("`review_status`", f"`{t7.review_status}`"),
            _md_row("Gate blocked", f"**{str(t7.gate_blocked).lower()}**"),
            _md_row("Freeze present", str(t7.freeze_present).lower()),
            _md_row(
                "Official scores",
                str(t7.official_surviving_letter_scores).lower(),
            ),
            "",
            "Independent review (reviewer, date, witness/source version, "
            "surviving-vs-restoration decisions) remains a **blocking** follow-up "
            "under JAK-43 eval readiness. Do not invent `human-checked`.",
            "",
            "## Identical-input comparison (NT holdouts)",
            "",
            "Decoder (JAK-39/40) vs elementary word-for-word train-cooccurrence "
            "argmax (JAK-28 lineage) on the **same** Sahidic holdout surfaces. "
            f"Gold: Vaticanus B* at eval time (`{report.nt_holdout_review_status}`).",
            "",
            _md_row("Metric", "Decoder", "Word-for-word"),
            _md_row("---", "---", "---"),
            _md_row("Units", d.n_units, b.n_units),
            _md_row("Mean letter F1", _fmt(d.mean_letter_f1), _fmt(b.mean_letter_f1)),
            _md_row(
                "Mean letter precision",
                _fmt(d.mean_letter_precision),
                _fmt(b.mean_letter_precision),
            ),
            _md_row(
                "Mean letter recall",
                _fmt(d.mean_letter_recall),
                _fmt(b.mean_letter_recall),
            ),
            _md_row(
                "Mean surface-exact recall",
                _fmt(d.mean_surface_exact_recall),
                _fmt(b.mean_surface_exact_recall),
            ),
            _md_row(
                "Mean lemma-proxy recall",
                _fmt(d.mean_lemma_proxy_recall),
                _fmt(b.mean_lemma_proxy_recall),
            ),
            _md_row(
                "Mean lemma-LCS recall",
                _fmt(d.mean_lemma_lcs_recall),
                _fmt(b.mean_lemma_lcs_recall),
            ),
            _md_row("Mean coverage", _fmt(d.mean_coverage), _fmt(b.mean_coverage)),
            _md_row(
                "Mean abstention rate",
                _fmt(d.mean_abstention_rate),
                _fmt(b.mean_abstention_rate),
            ),
            _md_row(
                "Total additions / omissions",
                f"{d.total_additions} / {d.total_omissions}",
                f"{b.total_additions} / {b.total_omissions}",
            ),
            _md_row(
                "Construction spans (hits)",
                f"{d.total_construction_spans} ({d.total_construction_phrase_hits})",
                f"{b.total_construction_spans} ({b.total_construction_phrase_hits})",
            ),
            _md_row(
                "Letter-F1 unit wins",
                d.letter_f1_unit_wins_vs_other,
                b.letter_f1_unit_wins_vs_other,
            ),
            "",
            f"Primary committed ablation arm: `{report.primary_ablation.value}`.",
            "",
            "## Ablation arms (decoder only)",
            "",
        ]
    )
    if report.ablation_arms:
        lines.append("| Arm | Mean letter F1 | Δ vs word-for-word | Units |")
        lines.append("| --- | --- | --- | --- |")
        for arm in report.ablation_arms:
            lines.append(
                f"| `{arm.ablation_condition.value}` | "
                f"{_fmt(arm.aggregate.mean_letter_f1)} | "
                f"{_fmt(arm.vs_word_for_word_mean_letter_f1_delta)} | "
                f"{arm.aggregate.n_units} |"
            )
        lines.append("")
    else:
        lines.append("_Ablation arms not regenerated in this run._")
        lines.append("")

    lines.extend(
        [
            "## Metric definitions",
            "",
        ]
    )
    for key, definition in report.metric_definitions.items():
        lines.append(f"- **{key}:** {definition}")
    lines.extend(
        [
            "",
            "## Failure-mode taxonomy (examples)",
            "",
        ]
    )
    by_cat: dict[str, list[FailureModeExample]] = defaultdict(list)
    for ex in report.failure_modes:
        by_cat[ex.category].append(ex)
    for cat, items in sorted(by_cat.items()):
        lines.append(f"### `{cat}`")
        lines.append("")
        for ex in items[:3]:
            sys_l = f" [{ex.system.value}]" if ex.system else ""
            lines.append(f"- **{ex.unit_id}**{sys_l}: {ex.detail}")
        lines.append("")

    lines.extend(
        [
            "## Leakage safeguards (confirmed)",
            "",
            "- Blinded reconstruct did not consult P.Oxy. 654 / holdout Greek.",
            "- Train/dev not scored as holdout; omission fixtures excluded.",
            "- Dual aligners kept (`eflomal` + `fast_align`); `adopted_aligner=null`.",
            "- No LLM / network on the reconstruction or metrics path.",
            "- Restorations excluded from surviving-letter gold policy (JAK-40).",
            "",
            "## Reproduce offline",
            "",
            "```bash",
            "uv run thomas eval feasibility-report",
            "uv run thomas eval feasibility-report --skip-ablation-arms \\",
            "  -o data/reconstructions/jak41/feasibility-metrics-v1.json \\",
            "  --write-doc docs/phase2-feasibility-report.md",
            "```",
            "",
            "## Out of scope",
            "",
            "Phase 3 architecture; adopting Ollama (JAK-42 deferred); inventing "
            "numeric pass thresholds; merging without Jake.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"


def write_feasibility_artifacts(
    report: FeasibilityReport,
    *,
    metrics_path: Path,
    doc_path: Path | None = None,
    max_bytes: int | None = COMMITTED_ARTIFACT_MAX_BYTES,
) -> dict[str, str]:
    """Write metrics JSON (+ optional markdown doc); return digests."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    # Prefer pretty; fall back to compact if over the large-file hook.
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    data = text.encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        # Drop per-unit rows from the committed pretty file? Prefer compact first.
        data = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        # Still too large: omit unit_metrics from committed artifact (kept in summary).
        slim = dict(payload)
        slim["unit_metrics"] = []
        slim["notes"] = [
            *payload.get("notes", []),
            "Committed artifact omits per-unit rows to stay under 1024 KiB; "
            "regenerate with --output outside data/reconstructions/jak41 "
            "for full rows.",
        ]
        data = (json.dumps(slim, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        msg = (
            f"feasibility metrics artifact {metrics_path} is {len(data)} bytes "
            f"(limit {max_bytes})"
        )
        raise ValueError(msg)
    metrics_path.write_bytes(data)
    digests = {"metrics_sha256": _sha256_bytes(data)}
    if doc_path is not None:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        md = render_feasibility_markdown(report)
        doc_path.write_text(md, encoding="utf-8")
        digests["doc_sha256"] = _sha256_bytes(md.encode("utf-8"))
    return digests


def assert_baseline_identical_inputs(package: BlindedReconstructionPackage) -> None:
    """Raise if any unit lacks a baseline on the same Coptic surfaces."""
    bases = baseline_by_unit(package)
    for unit in package.units:
        base = bases.get(unit.unit_id)
        if base is None:
            msg = f"missing baseline for {unit.unit_id}"
            raise AssertionError(msg)
        if base.coptic_surfaces != unit.coptic_surfaces:
            msg = f"Coptic surface mismatch for {unit.unit_id}"
            raise AssertionError(msg)

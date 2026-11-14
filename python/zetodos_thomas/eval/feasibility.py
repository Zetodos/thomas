# SPDX-License-Identifier: Apache-2.0
"""Multi-level feasibility metrics: decoder vs word-for-word (JAK-41).

Compares JAK-40 reconstructions to the elementary train-cooccurrence baseline
on **identical** Coptic inputs. No invented pass threshold. Official Thomas 7
surviving-letter scores remain gated on ``human-checked`` + verified freeze.
Evaluation-ready is reported separately from the qualitative recommendation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zetodos_thomas.eval.blinded_reconstruct import (
    holdout_units,
    reconstruct_blinded_package,
)
from zetodos_thomas.eval.letters import (
    TokenAwareLetterScore,
    greek_letters_only,
    token_aware_letter_score,
)
from zetodos_thomas.eval.review_gate import (
    EvalReviewGateError,
    assert_eval_package_human_checked,
    load_blinded_eval_package,
)
from zetodos_thomas.eval.surviving_letters import (
    BlindedScoreIntegrityError,
    load_freeze_record,
    verify_official_freeze,
)
from zetodos_thomas.eval.validate_reconstruction import (
    ReconstructionValidationError,
    assert_valid_reconstruction,
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

ALIGNMENT_PROTOCOL = "token-edit-alignment-v1"
LEMMA_UNMEASURED_DEFINITION = (
    "Lemma/inflection accuracy unmeasured: JAK-37 infer_greek_lemma only "
    "strips combining marks and duplicates greek_letters_only / surface_exact; "
    "cannot detect παραβολη vs παραβολην. Not an independent gate signal."
)

METRIC_DEFINITIONS: dict[str, str] = {
    "letter_agreement": (
        "Letter F1/precision/recall on edit-aligned token pairs "
        f"({ALIGNMENT_PROTOCOL}); abstention gaps preserved before alignment. "
        "Global LCS is diagnostic only."
    ),
    "surface_exact": (
        "Exact surface matches among edit-aligned pairs after letter-fold "
        "(casefold + strip diacritics + letters only). Recall = exact hits / "
        "gold tokens. Strict index-positional diagnostics are separate."
    ),
    "lemma_proxy": LEMMA_UNMEASURED_DEFINITION,
    "lemma_lcs": (
        "Longest common subsequence length of letter-folded surfaces / gold "
        "token count — order-preserving content recovery (not lemma morphology)."
    ),
    "near_miss_tokens": (
        "Edit-aligned pairs with shared letters but not exact surface match "
        "(weak near-miss signal; not morphology accuracy)."
    ),
    "word_order": (
        "Order-sensitive: edit alignment charges substitutions/insertions/"
        "deletions; reordering that loses position lowers aligned surface "
        "scores even when surface-LCS stays high."
    ),
    "phrase_fidelity": (
        "Actual source-span-localized phrase fidelity is unavailable. "
        "construction_phrase_hits reports an unlocalized whole-verse phrase "
        "occurrence diagnostic only (not mapping fidelity)."
    ),
    "additions_omissions": (
        "From token-level edit alignment: insertions (unsupported hyp tokens), "
        "deletions (omitted gold tokens), and substitutions. Not length-only "
        "sequence-length differences."
    ),
    "uncertainty": (
        "Decoder span abstention rate + mean top-1 candidate uncertainty when "
        "candidates exist. Coverage is source-token coverage from "
        "input_span.token_ids; span_coverage is reported separately."
    ),
    "surviving_letter_thomas7": (
        "Official Thomas 7 surviving-letter agreement requires "
        "``review_status=human-checked`` + freeze pin whose digest matches the "
        "frozen reconstruction (JAK-40). Gate-blocked status is reported "
        "honestly; not invented."
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
    """Flatten span hypotheses; drop nulls; split multi-surface spans.

    Legacy helper retained for callers that need a null-dropping bag. Scoring
    uses :func:`expand_hypothesis_preserving_gaps` + edit alignment instead.
    """
    out: list[str] = []
    for item in preds:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        out.extend(text.split())
    return out


def span_source_token_counts(
    decode_unit: BlindedUnitReconstruction | None,
    hypothesis: list[str | None],
) -> list[int]:
    """Per-span source token counts from ``input_span.token_ids`` when present."""
    if decode_unit is not None:
        counts: list[int] = []
        for span in decode_unit.decode_result.spans:
            if span.input_span is not None and span.input_span.token_ids:
                counts.append(len(span.input_span.token_ids))
            else:
                counts.append(max(1, len(span.surfaces) or 1))
        if len(counts) == len(hypothesis):
            return counts
    return [1] * len(hypothesis)


def expand_hypothesis_preserving_gaps(
    hypothesis: list[str | None],
    *,
    source_token_counts: list[int] | None = None,
) -> list[str | None]:
    """Expand per-span predictions into source-token-aligned slots.

    Abstentions emit ``None`` gap markers (one per source token). Multi-token
    Greek surfaces fill consecutive slots; leftover surfaces become extra hyp
    tokens (insertions under edit alignment). Does **not** drop nulls.
    """
    counts = source_token_counts or [1] * len(hypothesis)
    if len(counts) != len(hypothesis):
        counts = [1] * len(hypothesis)
    out: list[str | None] = []
    for pred, n_src in zip(hypothesis, counts, strict=False):
        n = max(1, int(n_src))
        if pred is None:
            out.extend([None] * n)
            continue
        words = [w for w in str(pred).split() if w]
        if not words:
            out.extend([None] * n)
            continue
        if len(words) <= n:
            out.extend(words)
            out.extend([None] * (n - len(words)))
        else:
            out.extend(words[:n])
            out.extend(words[n:])
    return out


@dataclass(frozen=True, slots=True)
class TokenEditAlignment:
    """Order-aware token edit alignment (match / subst / ins / del)."""

    matches: tuple[tuple[str, str], ...]
    substitutions: tuple[tuple[str, str], ...]
    insertions: tuple[str, ...]
    deletions: tuple[str, ...]
    protocol: str = ALIGNMENT_PROTOCOL

    @property
    def additions(self) -> int:
        return len(self.insertions)

    @property
    def omissions(self) -> int:
        return len(self.deletions)


def align_tokens_edit(
    hypothesis: list[str | None],
    gold_tokens: list[str],
) -> TokenEditAlignment:
    """Needleman-Wunsch / Levenshtein alignment with abstention gaps.

    ``None`` hypothesis slots are empty: they align only via deletion of the
    corresponding gold token (or as a no-op against nothing). Non-null hyp
    tokens may match, substitute, or insert.
    """
    hyp = list(hypothesis)
    gold = list(gold_tokens)
    n, m = len(hyp), len(gold)
    # dp[i][j] = min edit cost aligning hyp[:i] to gold[:j]
    # match=0 when equal non-empty; subst=1; ins=1; del=1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "I"  # insert/skip hyp
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "D"  # delete gold
    for i in range(1, n + 1):
        h = hyp[i - 1]
        for j in range(1, m + 1):
            g = gold[j - 1]
            candidates: list[tuple[int, str]] = []
            # deletion (omit gold)
            candidates.append((dp[i][j - 1] + 1, "D"))
            # insertion (extra hyp) — None hyp is a free gap skip
            if h is None:
                candidates.append((dp[i - 1][j], "I"))
            else:
                candidates.append((dp[i - 1][j] + 1, "I"))
            # match / substitute
            if h is None:
                # empty hyp cannot match; treat as del+skip via other ops
                pass
            else:
                cost = 0 if _surface_fold(h) == _surface_fold(g) and h else 1
                candidates.append((dp[i - 1][j - 1] + cost, "M" if cost == 0 else "S"))
            best_cost, best_op = min(candidates, key=lambda c: (c[0], c[1]))
            dp[i][j] = best_cost
            bt[i][j] = best_op

    matches: list[tuple[str, str]] = []
    substitutions: list[tuple[str, str]] = []
    insertions: list[str] = []
    deletions: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j] if i >= 0 and j >= 0 else ""
        if i > 0 and j > 0 and op == "M":
            matches.append((hyp[i - 1] or "", gold[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and op == "S":
            substitutions.append((hyp[i - 1] or "", gold[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or op == "I"):
            if hyp[i - 1] is not None:
                insertions.append(hyp[i - 1] or "")
            i -= 1
        else:
            deletions.append(gold[j - 1])
            j -= 1
    matches.reverse()
    substitutions.reverse()
    insertions.reverse()
    deletions.reverse()
    return TokenEditAlignment(
        matches=tuple(matches),
        substitutions=tuple(substitutions),
        insertions=tuple(insertions),
        deletions=tuple(deletions),
    )


def gold_tokens_for_unit(unit: ParallelUnit) -> list[str]:
    text = unit.greek.normalized_text or unit.greek.original_text
    return [t for t in text.split() if t]


def _lemma_fold(token: str) -> str:
    """Deprecated accent-strip proxy — not independent of surface fold."""
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


def _paired_letter_score(
    pairs: list[tuple[str, str]],
) -> TokenAwareLetterScore:
    """Letter score on aligned pairs (incidental LCS diagnostic helper)."""
    hyp_side = [h for h, _g in pairs if h]
    gold_side = [g for _h, g in pairs if g]
    return token_aware_letter_score(hyp_side, gold_side)


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
    counts = span_source_token_counts(decode_unit, hypothesis)
    expanded = expand_hypothesis_preserving_gaps(hypothesis, source_token_counts=counts)
    alignment = align_tokens_edit(expanded, gold_tokens)

    paired_for_letters = [
        *[(h, g) for h, g in alignment.matches],
        *[(h, g) for h, g in alignment.substitutions],
    ]
    letter = _paired_letter_score(paired_for_letters)
    # Denominators must reflect full predicted/gold letter mass, not only pairs.
    hyp_letters = "".join(_surface_fold(t) for t in expanded if t is not None)
    gold_letters = "".join(_surface_fold(t) for t in gold_tokens)
    # Recompute precision/recall with full denominators + TP from aligned pairs.
    tp = 0
    for h, g in paired_for_letters:
        hs, gs = _surface_fold(h), _surface_fold(g)
        hi = gi = 0
        while hi < len(hs) and gi < len(gs):
            if hs[hi] == gs[gi]:
                tp += 1
                hi += 1
                gi += 1
            else:
                gi += 1
    letter_precision = (tp / len(hyp_letters)) if hyp_letters else None
    letter_recall = (tp / len(gold_letters)) if gold_letters else None
    if (
        letter_precision is None
        or letter_recall is None
        or (letter_precision + letter_recall) == 0
    ):
        letter_f1 = None
    else:
        letter_f1 = (
            2.0 * letter_precision * letter_recall / (letter_precision + letter_recall)
        )

    exact_tp = sum(
        1
        for h, g in alignment.matches
        if _surface_fold(h) and _surface_fold(h) == _surface_fold(g)
    )
    near_miss = 0
    for h, g in alignment.substitutions:
        hs, gs = _surface_fold(h), _surface_fold(g)
        if hs and gs and hs != gs and (set(hs) & set(gs)):
            near_miss += 1

    hyp_surf_non_null = [_surface_fold(t) for t in expanded if t is not None]
    gold_surf = [_surface_fold(t) for t in gold_tokens]
    surface_lcs = _token_lcs_len(hyp_surf_non_null, gold_surf)

    # Strict index-positional diagnostic (gap-preserving; not collapsed).
    n_strict = min(len(expanded), len(gold_tokens))
    strict_hits = sum(
        1
        for i in range(n_strict)
        if expanded[i] is not None
        and _surface_fold(expanded[i] or "") == gold_surf[i]
        and gold_surf[i]
    )

    n_coptic = len(coptic_surfaces)
    n_non_null_spans = sum(1 for p in hypothesis if p is not None)
    n_spans = len(hypothesis)

    # Token-level coverage from input_span.token_ids / covering windows.
    covered_tokens = 0
    if decode_unit is not None:
        for span in decode_unit.decode_result.spans:
            n_tok = (
                len(span.input_span.token_ids)
                if span.input_span is not None and span.input_span.token_ids
                else max(1, len(span.surfaces) or 1)
            )
            if span.status is not SpanDecodeStatus.ABSTAIN:
                covered_tokens += n_tok
    else:
        # Baseline: each non-null prediction covers one Coptic surface slot.
        covered_tokens = n_non_null_spans

    token_coverage = (covered_tokens / n_coptic) if n_coptic else None
    span_coverage = (n_non_null_spans / n_spans) if n_spans else None

    construction_spans = 0
    construction_hits = 0
    kinds = covering_span_kinds or []
    if decode_unit is not None:
        for kind, span in zip(kinds, decode_unit.decode_result.spans, strict=False):
            multi = (
                span.input_span is not None and len(span.input_span.token_ids) > 1
            ) or len(span.surfaces) > 1
            if kind != "construction" or not multi:
                continue
            construction_spans += 1
            if span.status is SpanDecodeStatus.CANDIDATES and span.candidates:
                pred = [_surface_fold(s) for s in span.candidates[0].greek_surfaces]
                # Unlocalized whole-verse occurrence only — not phrase fidelity.
                if _contiguous_subsequence(pred, gold_surf):
                    construction_hits += 1

    span_abstention_rate: float | None = None
    mean_unc: float | None = None
    if decode_unit is not None:
        spans = decode_unit.decode_result.spans
        if spans:
            n_abs = sum(1 for s in spans if s.status is SpanDecodeStatus.ABSTAIN)
            span_abstention_rate = n_abs / len(spans)
            uncs = [
                s.candidates[0].uncertainty
                for s in spans
                if s.status is SpanDecodeStatus.CANDIDATES and s.candidates
            ]
            mean_unc = (sum(uncs) / len(uncs)) if uncs else None
    elif n_coptic:
        span_abstention_rate = 1.0 - (n_non_null_spans / n_coptic)

    # Token abstention = 1 - token_coverage when available.
    abstention_rate = (
        (1.0 - token_coverage) if token_coverage is not None else span_abstention_rate
    )

    unmeasured_lemma = _ratio(
        0,
        0,
        definition=LEMMA_UNMEASURED_DEFINITION,
    )

    return UnitSystemMetrics(
        unit_id=unit_id,
        system=system,
        letter_true_positives=tp,
        letter_predicted_denominator=len(hyp_letters),
        letter_gold_denominator=len(gold_letters),
        letter_precision=letter_precision,
        letter_recall=letter_recall,
        letter_f1=letter_f1,
        incidental_global_lcs=letter.incidental_global_lcs,
        letter_metric_kind="token-edit-aligned-letter-v1",
        surface_exact=_ratio(
            exact_tp,
            len(gold_tokens),
            definition=(
                "edit-aligned exact surface hits / gold tokens "
                f"(strict-index hits diagnostic={strict_hits})"
            ),
        ),
        lemma_proxy=unmeasured_lemma,
        lemma_lcs=_ratio(
            surface_lcs,
            len(gold_tokens),
            definition="letter-folded surface LCS length / gold tokens (not lemmas)",
        ),
        near_miss_tokens=near_miss,
        predicted_tokens=sum(1 for t in expanded if t is not None),
        gold_tokens=len(gold_tokens),
        additions=alignment.additions,
        omissions=alignment.omissions,
        substitutions=len(alignment.substitutions),
        coptic_surfaces=n_coptic,
        predicted_non_null_spans=n_non_null_spans,
        predicted_covered_tokens=covered_tokens,
        coverage=token_coverage,
        span_coverage=span_coverage,
        token_coverage=token_coverage,
        abstention_rate=abstention_rate,
        span_abstention_rate=span_abstention_rate,
        mean_top1_uncertainty=mean_unc,
        construction_spans=construction_spans,
        construction_phrase_hits=construction_hits,
        phrase_fidelity_available=False,
        lemma_inflection_accuracy_available=False,
        alignment_protocol=ALIGNMENT_PROTOCOL,
        notes=[
            "Gold consulted at scoring time only (identical Coptic inputs).",
            LEMMA_UNMEASURED_DEFINITION,
            "construction_phrase_hits is unlocalized verse occurrence, not fidelity.",
            "incidental_global_lcs is diagnostic only.",
            f"alignment_protocol={ALIGNMENT_PROTOCOL}",
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
            notes=["No units scored."],
        )
    letter_tp = sum(r.letter_true_positives for r in rows)
    letter_gold = sum(r.letter_gold_denominator for r in rows)
    surf_tp = sum(r.surface_exact.numerator for r in rows)
    surf_den = sum(r.surface_exact.denominator for r in rows)
    return AggregateSystemMetrics(
        system=system,
        n_units=len(rows),
        mean_letter_precision=_mean([r.letter_precision for r in rows]),
        mean_letter_recall=_mean([r.letter_recall for r in rows]),
        mean_letter_f1=_mean([r.letter_f1 for r in rows]),
        mean_surface_exact_recall=_mean([r.surface_exact.value for r in rows]),
        mean_lemma_proxy_recall=None,  # unmeasured — not an independent signal
        mean_lemma_lcs_recall=_mean([r.lemma_lcs.value for r in rows]),
        mean_coverage=_mean([r.token_coverage for r in rows]),
        mean_abstention_rate=_mean([r.abstention_rate for r in rows]),
        mean_span_coverage=_mean([r.span_coverage for r in rows]),
        mean_token_coverage=_mean([r.token_coverage for r in rows]),
        pooled_letter=_ratio(
            letter_tp,
            letter_gold,
            definition="pooled letter TP / pooled gold letters (recall-oriented)",
        ),
        pooled_surface_exact=_ratio(surf_tp, surf_den),
        pooled_lemma_proxy=_ratio(
            0,
            0,
            definition=LEMMA_UNMEASURED_DEFINITION,
        ),
        total_additions=sum(r.additions for r in rows),
        total_omissions=sum(r.omissions for r in rows),
        total_substitutions=sum(r.substitutions for r in rows),
        total_construction_spans=sum(r.construction_spans for r in rows),
        total_construction_phrase_hits=sum(r.construction_phrase_hits for r in rows),
        letter_f1_unit_wins_vs_other=wins_vs_other,
        notes=[
            "Macro means average per-unit ratios; pooled rows use summed denominators.",
            "No invented pass/fail threshold (PLAN.md Phase 2).",
            "mean_coverage is source-token coverage; mean_span_coverage is separate.",
            "lemma_proxy / mean_lemma_proxy_recall unmeasured (not a gate signal).",
            "construction_phrase_hits are unlocalized verse occurrences only.",
        ],
    )


def load_reconstruction_package(path: Path) -> BlindedReconstructionPackage:
    return BlindedReconstructionPackage.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def load_and_validate_reconstruction_package(
    path: Path,
    *,
    repo_root: Path,
) -> BlindedReconstructionPackage:
    """Parse + JAK-40 integrity-validate a reconstruction package."""
    package = load_reconstruction_package(path)
    assert_valid_reconstruction(package, repo_root=repo_root)
    return package


def assert_exact_holdout_membership(
    package: BlindedReconstructionPackage,
    *,
    repo_root: Path,
    allow_partial_non_gating: bool = False,
) -> list[str]:
    """Require every canonical NT holdout exactly once (or label partial)."""
    expected = {u.unit_id for u in holdout_units(repo_root)}
    got_list = [
        u.unit_id
        for u in package.units
        if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    ]
    got = set(got_list)
    problems: list[str] = []
    if len(got_list) != len(got):
        problems.append("duplicated NT holdout unit_ids in reconstruction package")
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        problems.append(f"missing canonical holdout units: {missing}")
    if extra:
        problems.append(f"non-holdout units marked nt_holdout: {extra}")
    if problems and not allow_partial_non_gating:
        raise ReconstructionValidationError(
            "holdout membership check failed:\n- " + "\n- ".join(problems)
        )
    return problems


def baseline_by_unit(
    package: BlindedReconstructionPackage,
) -> dict[str, WordForWordUnitBaseline]:
    return {b.unit_id: b for b in package.word_for_word_baseline}


def score_nt_holdout_package(
    package: BlindedReconstructionPackage,
    gold_by_id: dict[str, ParallelUnit],
    *,
    require_all_holdouts: bool = True,
) -> tuple[list[UnitSystemMetrics], list[UnitSystemMetrics]]:
    """Score decoder + word-for-word on identical NT holdout Coptic inputs."""
    bases = baseline_by_unit(package)
    decoder_rows: list[UnitSystemMetrics] = []
    baseline_rows: list[UnitSystemMetrics] = []
    skipped: list[str] = []
    for unit in package.units:
        if unit.target_kind is not BlindedTargetKind.NT_HOLDOUT:
            continue
        gold_unit = gold_by_id.get(unit.unit_id)
        if gold_unit is None:
            skipped.append(unit.unit_id)
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
    if skipped and require_all_holdouts:
        msg = (
            "NT holdout package references units absent from gold_by_id "
            f"(non-gating if intentional): {skipped}"
        )
        raise ValueError(msg)
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


def thomas7_gate_status(
    repo_root: Path,
    *,
    thomas_package: BlindedReconstructionPackage | None = None,
) -> Thomas7GateStatus:
    """Report human-checked / official gate status without inventing scores."""
    pack_path = repo_root / DEFAULT_THOMAS7_EVAL_REL
    freeze_path = repo_root / DEFAULT_FREEZE_REL
    payload = load_blinded_eval_package(pack_path)
    package_id = str(payload.get("id", pack_path.name))
    review_status = str(payload.get("review_status", ""))
    gate_blocked = True
    freeze_present = freeze_path.is_file()
    freeze_verified = False
    notes = [
        "Official Thomas 7 surviving-letter scores require human-checked + "
        "verified freeze digest match against the reconstruction artifact.",
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

    if freeze_present and thomas_package is not None:
        try:
            verify_official_freeze(thomas_package, load_freeze_record(freeze_path))
            freeze_verified = True
            notes.append("freeze digest verified against Thomas reconstruction.")
        except BlindedScoreIntegrityError as exc:
            notes.append(f"freeze present but not verified: {exc}")
    elif freeze_present:
        notes.append(
            "freeze file exists but official scores require digest verification "
            "against the reconstruction package (not mere existence)."
        )
    else:
        notes.append("freeze file absent.")

    official = (not gate_blocked) and freeze_verified
    if not official:
        notes.append(
            "official_surviving_letter_scores=false until human-checked and "
            "freeze-verified scores are produced."
        )
    return Thomas7GateStatus(
        eval_package_id=package_id,
        review_status=review_status,
        official_surviving_letter_scores=official,
        gate_blocked=gate_blocked,
        freeze_present=freeze_present,
        freeze_verified=freeze_verified,
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

    examples.append(
        FailureModeExample(
            category="morphology_underdetermination",
            unit_id="(corpus)",
            system=None,
            detail=(
                "Lemma/inflection accuracy is unmeasured: JAK-37 lemma proxies "
                "duplicate surface letter-fold and cannot detect inflection "
                "errors (e.g. παραβολη vs παραβολην)."
            ),
            evidence={
                "lemma_method": "unmeasured",
                "near_miss_tokens_decoder": sum(
                    r.near_miss_tokens for r in decoder_rows
                ),
            },
        )
    )

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
        counts = span_source_token_counts(unit, unit.top1_greek_surfaces)
        expanded = expand_hypothesis_preserving_gaps(
            unit.top1_greek_surfaces, source_token_counts=counts
        )
        alignment = align_tokens_edit(expanded, gold)
        for idx, (hyp, g) in enumerate(alignment.substitutions):
            if _surface_fold(hyp) in {
                "και",
                "ο",
                "η",
                "το",
                "του",
                "την",
            } and _surface_fold(hyp) != _surface_fold(g):
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
                            "alignment_index": idx,
                            "hypothesis": hyp,
                            "gold": g,
                        },
                    )
                )
                break
        if any(
            e.category == "false_equivalence" and e.unit_id == unit.unit_id
            for e in examples
        ):
            break

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
                        "High Coptic token coverage with low letter F1 — emitting "
                        "familiar NT Greek that does not match this holdout."
                    ),
                    evidence={
                        "token_coverage": row.token_coverage,
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
                detail=(
                    "Lowest letter-F1 holdout unit — edit-alignment "
                    "insertions/deletions/substitutions."
                ),
                evidence={
                    "letter_f1": worst.letter_f1,
                    "additions": worst.additions,
                    "omissions": worst.omissions,
                    "substitutions": worst.substitutions,
                    "abstention_rate": worst.abstention_rate,
                },
            )
        )

    thomas = next(
        (u for u in package.units if u.target_kind is BlindedTargetKind.THOMAS_PILOT),
        None,
    )
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

    if decoder_rows and baseline_rows:
        dec_cov = _mean([r.token_coverage for r in decoder_rows])
        base_cov = _mean([r.token_coverage for r in baseline_rows])
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
                        "Word-for-word baseline covers more Coptic tokens but "
                        "does not beat decoder letter F1 — coverage ≠ fidelity."
                    ),
                    evidence={
                        "decoder_mean_token_coverage": dec_cov,
                        "baseline_mean_token_coverage": base_cov,
                        "decoder_mean_letter_f1": dec_f1,
                        "baseline_mean_letter_f1": base_f1,
                    },
                )
            )

    _ = by_dec
    return examples


def assess_evaluation_ready(
    *,
    thomas7: Thomas7GateStatus,
    decoder: AggregateSystemMetrics,
    baseline: AggregateSystemMetrics,
    nt_review_status: str,
    packages_validated: bool,
    holdout_membership_complete: bool,
) -> tuple[bool, list[str]]:
    """Separate evaluation-ready/not-ready from qualitative recommendation."""
    notes: list[str] = []
    ready = True
    if thomas7.gate_blocked:
        ready = False
        notes.append(
            f"Thomas 7 review_status={thomas7.review_status!r} is not human-checked."
        )
    if not thomas7.freeze_verified:
        ready = False
        notes.append("Thomas 7 freeze is absent or not digest-verified.")
    if not thomas7.official_surviving_letter_scores:
        ready = False
        notes.append("Official Thomas 7 surviving-letter scores are unavailable.")
    if nt_review_status != "human-checked":
        ready = False
        notes.append(f"NT holdout gold remains {nt_review_status!r}.")
    if decoder.mean_letter_f1 is None or baseline.mean_letter_f1 is None:
        ready = False
        notes.append("Decoder or baseline letter F1 metrics are missing.")
    if decoder.n_units == 0 or baseline.n_units == 0:
        ready = False
        notes.append("No units scored — data insufficient.")
    if not packages_validated:
        ready = False
        notes.append("Reconstruction packages failed integrity validation.")
    if not holdout_membership_complete:
        ready = False
        notes.append("Canonical NT holdout membership is incomplete or partial.")
    if ready:
        notes.append(
            "Evaluation inputs are present; numeric proceed criteria are still "
            "unapproved — recommendation stays qualitative."
        )
    return ready, notes


def choose_gate_recommendation(
    *,
    thomas7: Thomas7GateStatus,
    decoder: AggregateSystemMetrics,
    baseline: AggregateSystemMetrics,
    nt_review_status: str,
    evaluation_ready: bool | None = None,
    evaluation_ready_notes: list[str] | None = None,
    packages_validated: bool = True,
    holdout_membership_complete: bool = True,
) -> tuple[GateRecommendation, list[str], bool, list[str]]:
    """Evidence-backed advisory recommendation (Jake decides).

    No invented numeric cutoffs. ``proceed`` requires evaluation-ready inputs
    and approved decision criteria (none approved yet → never auto-proceed).
    """
    if evaluation_ready is None:
        evaluation_ready, ready_notes = assess_evaluation_ready(
            thomas7=thomas7,
            decoder=decoder,
            baseline=baseline,
            nt_review_status=nt_review_status,
            packages_validated=packages_validated,
            holdout_membership_complete=holdout_membership_complete,
        )
    else:
        ready_notes = list(evaluation_ready_notes or [])

    rationale: list[str] = []
    dec_f1 = decoder.mean_letter_f1
    base_f1 = baseline.mean_letter_f1
    delta = None if dec_f1 is None or base_f1 is None else dec_f1 - base_f1

    rationale.append(
        "Evaluation-ready="
        f"{str(evaluation_ready).lower()} — "
        + ("; ".join(ready_notes) if ready_notes else "inputs checked.")
    )
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
            f"(baseline≈{base_f1:.4f}; delta≈{delta:+.4f}) — descriptive only; "
            "no approved numeric pass threshold."
        )
    rationale.append(
        "Lemma/inflection accuracy is unmeasured and is not used as an "
        "independent gate signal."
    )
    rationale.append(
        "Jake Phase 2 note: corpus/morphology gaps are still the likely "
        "limiting factors; absolute recovery is far from production-scale."
    )
    rationale.append(
        "Ollama (JAK-42) stays deferred: limitation is evidence/eval readiness "
        "and morphology/corpus depth, not a demonstrated need for LLM fill-in."
    )
    rationale.append(
        "No invented numeric decision cutoffs (PLAN.md / JAK-41); "
        "Proceed requires evaluation-ready inputs plus approved criteria."
    )

    if not evaluation_ready:
        return (
            GateRecommendation.PAUSE_AND_IMPROVE,
            rationale,
            evaluation_ready,
            ready_notes,
        )

    # Evaluation-ready but no approved numeric proceed criteria.
    if dec_f1 is not None and base_f1 is not None and dec_f1 < base_f1:
        rationale.append(
            "Decoder descriptively underperforms the word-for-word baseline on "
            "mean letter F1 with evaluation-ready inputs — reassess method "
            "before Phase 3 architecture commitment."
        )
        return (
            GateRecommendation.REASSESS,
            rationale,
            evaluation_ready,
            ready_notes,
        )

    rationale.append(
        "Evaluation inputs are ready but no approved numeric proceed criteria "
        "exist; advisory recommendation remains pause_and_improve."
    )
    return (
        GateRecommendation.PAUSE_AND_IMPROVE,
        rationale,
        evaluation_ready,
        ready_notes,
    )


def _thomas7_blinded_coverage(unit: BlindedUnitReconstruction) -> dict[str, Any]:
    n_coptic = len(unit.coptic_surfaces)
    covered = 0
    for span in unit.decode_result.spans:
        n_tok = (
            len(span.input_span.token_ids)
            if span.input_span is not None and span.input_span.token_ids
            else max(1, len(span.surfaces) or 1)
        )
        if span.status is not SpanDecodeStatus.ABSTAIN:
            covered += n_tok
    return {
        "unit_id": unit.unit_id,
        "coptic_surfaces": n_coptic,
        "covered_tokens": covered,
        "token_coverage": (covered / n_coptic) if n_coptic else None,
        "top1_greek_surfaces": unit.top1_greek_surfaces,
        "non_null_spans": sum(1 for p in unit.top1_greek_surfaces if p is not None),
    }


def _count_hypothesis_changes(
    a: BlindedReconstructionPackage,
    b: BlindedReconstructionPackage,
) -> int:
    by_b = {u.unit_id: u for u in b.units}
    changes = 0
    for ua in a.units:
        ub = by_b.get(ua.unit_id)
        if ub is None:
            continue
        if ua.top1_greek_surfaces != ub.top1_greek_surfaces:
            changes += 1
    return changes


def score_ablation_arm(
    repo_root: Path,
    *,
    condition: AblationCondition,
    gold_by_id: dict[str, ParallelUnit],
    baseline_rows_ref: list[UnitSystemMetrics] | None = None,
    reference_package: BlindedReconstructionPackage | None = None,
) -> AblationArmComparison:
    """Regenerate an ablation arm for diagnostics only.

    Regenerated post-gold arms are **uninformative** for the L-evidence
    decision unless independently frozen provenance-matched outputs exist.
    """
    package = reconstruct_blinded_package(
        repo_root,
        ablation=condition,
        include_nt_holdouts=True,
        max_candidates_per_span=1,
        include_word_for_word=True,
    )
    nt_units = [
        u for u in package.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
    ]
    t7 = next(
        (u for u in package.units if u.target_kind is BlindedTargetKind.THOMAS_PILOT),
        None,
    )
    t7_cov = _thomas7_blinded_coverage(t7) if t7 is not None else {}

    loanword_changes = None
    if reference_package is not None:
        loanword_changes = _count_hypothesis_changes(reference_package, package)

    notes = [
        (
            "UNINFORMATIVE for L-evidence / loanword decision: arm regenerated "
            "at report time after Greek gold is available, not compared as an "
            "independently frozen blind output matched to JAK-40 digests."
        ),
        f"Arm reconstructed offline with ablation={condition.value}.",
        "Thomas 7 blinded coverage reported without unsealing Greek.",
        "Equal NT table metrics alone cannot verify loanword exercise.",
    ]
    if loanword_changes is not None:
        notes.append(
            f"Hypothesis unit-level changes vs reference package: {loanword_changes}."
        )

    # Still compute aggregates for transparency, but mark uninformative.
    nt_base = [
        b
        for b in package.word_for_word_baseline
        if any(u.unit_id == b.unit_id for u in nt_units)
    ]
    package_nt = package.model_copy(
        update={"units": nt_units, "word_for_word_baseline": nt_base}
    )
    decoder_rows, baseline_rows = score_nt_holdout_package(
        package_nt, gold_by_id, require_all_holdouts=False
    )
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
        informative_for_loanword_decision=False,
        evidence_status="uninformative_regenerated_post_gold",
        loanword_hypothesis_changes=loanword_changes,
        thomas7_blinded_coverage=t7_cov,
        notes=notes,
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

    packages_validated = True
    validation_notes: list[str] = []
    try:
        thomas_pkg = load_and_validate_reconstruction_package(
            thomas_path, repo_root=root
        )
        nt_pkg = load_and_validate_reconstruction_package(nt_path, repo_root=root)
    except ReconstructionValidationError as exc:
        packages_validated = False
        validation_notes.append(str(exc))
        thomas_pkg = load_reconstruction_package(thomas_path)
        nt_pkg = load_reconstruction_package(nt_path)

    # Cross-check ablation/split provenance consistency between packages.
    if thomas_pkg.ablation_condition != nt_pkg.ablation_condition:
        packages_validated = False
        validation_notes.append(
            "Thomas and NT packages have mismatched ablation_condition "
            f"({thomas_pkg.ablation_condition!r} vs {nt_pkg.ablation_condition!r})."
        )

    holdout_problems = assert_exact_holdout_membership(
        nt_pkg, repo_root=root, allow_partial_non_gating=True
    )
    holdout_complete = not holdout_problems
    if holdout_problems:
        validation_notes.extend(holdout_problems)

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
    decoder_rows, baseline_rows = score_nt_holdout_package(
        nt_pkg, gold_by_id, require_all_holdouts=holdout_complete
    )
    wins_d, wins_b, ties = _count_f1_wins(decoder_rows, baseline_rows)
    decoder_agg = aggregate_unit_metrics(
        decoder_rows, system=SystemLabel.DECODER, wins_vs_other=wins_d
    )
    baseline_agg = aggregate_unit_metrics(
        baseline_rows,
        system=SystemLabel.WORD_FOR_WORD,
        wins_vs_other=wins_b,
    )

    thomas7 = thomas7_gate_status(root, thomas_package=thomas_pkg)
    failure_modes = collect_failure_modes(
        nt_pkg, decoder_rows, baseline_rows, gold_by_id
    )
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
                    "token_coverage": _thomas7_blinded_coverage(t7_unit),
                },
            )
        )

    arms: list[AblationArmComparison] = []
    if include_ablation_arms:
        # Compare regenerated arms to each other for loanword hypothesis churn
        # without treating equal NT metrics as a measured L-evidence result.
        arm_pkgs: dict[AblationCondition, BlindedReconstructionPackage] = {}
        for cond in (
            AblationCondition.NT_ALIGNER_ONLY,
            AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        ):
            arm_pkgs[cond] = reconstruct_blinded_package(
                root,
                ablation=cond,
                include_nt_holdouts=True,
                max_candidates_per_span=1,
                include_word_for_word=True,
            )
        a_pkg = arm_pkgs[AblationCondition.NT_ALIGNER_ONLY]
        b_pkg = arm_pkgs[AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS]
        hyp_changes = _count_hypothesis_changes(a_pkg, b_pkg)
        t7_a = next(
            (u for u in a_pkg.units if u.target_kind is BlindedTargetKind.THOMAS_PILOT),
            None,
        )
        t7_b = next(
            (u for u in b_pkg.units if u.target_kind is BlindedTargetKind.THOMAS_PILOT),
            None,
        )
        for cond, pkg in arm_pkgs.items():
            nt_units = [
                u for u in pkg.units if u.target_kind is BlindedTargetKind.NT_HOLDOUT
            ]
            nt_base = [
                b
                for b in pkg.word_for_word_baseline
                if any(u.unit_id == b.unit_id for u in nt_units)
            ]
            package_nt = pkg.model_copy(
                update={"units": nt_units, "word_for_word_baseline": nt_base}
            )
            decoder_arm, _ = score_nt_holdout_package(
                package_nt, gold_by_id, require_all_holdouts=False
            )
            wins_arm, _, _ = _count_f1_wins(decoder_arm, baseline_rows)
            agg = aggregate_unit_metrics(
                decoder_arm, system=SystemLabel.DECODER, wins_vs_other=wins_arm
            )
            delta = None
            if (
                agg.mean_letter_f1 is not None
                and baseline_agg.mean_letter_f1 is not None
            ):
                delta = agg.mean_letter_f1 - baseline_agg.mean_letter_f1
            t7_u = t7_a if cond is AblationCondition.NT_ALIGNER_ONLY else t7_b
            arms.append(
                AblationArmComparison(
                    ablation_condition=cond,
                    aggregate=agg,
                    vs_word_for_word_mean_letter_f1_delta=delta,
                    informative_for_loanword_decision=False,
                    evidence_status="uninformative_regenerated_post_gold",
                    loanword_hypothesis_changes=hyp_changes,
                    thomas7_blinded_coverage=(
                        _thomas7_blinded_coverage(t7_u) if t7_u is not None else {}
                    ),
                    notes=[
                        (
                            "UNINFORMATIVE for L-evidence decision: regenerated "
                            "post-gold without independently frozen A/B "
                            "provenance matched to JAK-40."
                        ),
                        (
                            "A/B unit-level hypothesis changes "
                            f"(both arms): {hyp_changes}."
                        ),
                        "Thomas 7 blinded coverage without unsealing Greek.",
                        (
                            "Equal NT F1/coverage alone cannot distinguish "
                            "'loanwords made no difference' from 'no scored "
                            "holdout exercised a loanword'."
                        ),
                    ],
                )
            )

    recommendation, rationale, eval_ready, ready_notes = choose_gate_recommendation(
        thomas7=thomas7,
        decoder=decoder_agg,
        baseline=baseline_agg,
        nt_review_status="provisional-draft",
        packages_validated=packages_validated,
        holdout_membership_complete=holdout_complete,
    )
    ready_notes = [*ready_notes, *validation_notes]

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

    leakage_ok = (
        thomas_pkg.blinding.get("greek_consulted_during_reconstruction") is False
        and nt_pkg.blinding.get("greek_consulted_during_reconstruction") is False
        and packages_validated
        and holdout_complete
    )

    return FeasibilityReport(
        gate_recommendation=recommendation,
        gate_rationale=rationale,
        evaluation_ready=eval_ready,
        evaluation_ready_notes=ready_notes,
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
            "holdout_membership_complete": holdout_complete,
            "packages_validated": packages_validated,
            "letter_f1_wins": {
                "decoder": wins_d,
                "word_for_word": wins_b,
                "ties": ties,
            },
            "source_packages": {
                "thomas": thomas_pkg.id,
                "nt_holdout": nt_pkg.id,
            },
            "alignment_protocol": ALIGNMENT_PROTOCOL,
        },
        leakage_safeguards={
            "greek_consulted_during_reconstruction": (
                thomas_pkg.blinding.get("greek_consulted_during_reconstruction")
                is False
                and nt_pkg.blinding.get("greek_consulted_during_reconstruction")
                is False
            ),
            "train_dev_not_scored_as_holdout": holdout_complete,
            "poxy_654_sealed_until_scoring": True,
            "reconstruction_packages_validated": packages_validated,
            "holdout_membership_complete": holdout_complete,
            "safeguards_confirmed": leakage_ok,
            "adopted_aligner": None,
            "phase2_active_aligners": ["eflomal", "fast_align"],
            "llm": False,
            "network": False,
            "thomas7_official_gate": thomas7.model_dump(mode="json"),
        },
        artifact_digests=digests,
        summary={
            "gate_recommendation": recommendation.value,
            "evaluation_ready": eval_ready,
            "nt_units_scored": len(decoder_rows),
            "decoder_mean_letter_f1": decoder_agg.mean_letter_f1,
            "baseline_mean_letter_f1": baseline_agg.mean_letter_f1,
            "decoder_mean_token_coverage": decoder_agg.mean_token_coverage,
            "baseline_mean_token_coverage": baseline_agg.mean_token_coverage,
            "decoder_mean_lemma_proxy_recall": None,
            "baseline_mean_lemma_proxy_recall": None,
            "lemma_inflection_accuracy": "unmeasured",
            "phrase_fidelity": "unavailable_unlocalized_occurrence_only",
            "thomas7_gate_blocked": thomas7.gate_blocked,
            "ablation_arms_informative": False,
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
            (
                "No invented numeric pass thresholds; "
                "evaluation-ready separate from recommendation."
            ),
            *validation_notes,
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
        f"**Evaluation-ready:** `{str(report.evaluation_ready).lower()}`",
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
    if report.evaluation_ready_notes:
        lines.append("")
        lines.append("### Evaluation-ready checklist")
        lines.append("")
        for item in report.evaluation_ready_notes:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Jake chooses **Proceed / Pause and improve / Reassess**. This report "
            "does not invent a numeric OKR or merge Phase 3 work. "
            "`proceed` requires evaluation-ready inputs plus approved criteria.",
            "",
            "## Official Thomas 7 surviving-letter gate",
            "",
            _md_row("Field", "Value"),
            _md_row("---", "---"),
            _md_row("Eval package", f"`{t7.eval_package_id}`"),
            _md_row("`review_status`", f"`{t7.review_status}`"),
            _md_row("Gate blocked", f"**{str(t7.gate_blocked).lower()}**"),
            _md_row("Freeze present", str(t7.freeze_present).lower()),
            _md_row("Freeze verified", str(t7.freeze_verified).lower()),
            _md_row(
                "Official scores",
                str(t7.official_surviving_letter_scores).lower(),
            ),
            "",
            "Independent review (reviewer, date, witness/source version, "
            "surviving-vs-restoration decisions) remains a **blocking** follow-up "
            "under JAK-43 eval readiness. Do not invent `human-checked`. "
            "Freeze file existence alone does not make official scores true.",
            "",
            "## Identical-input comparison (NT holdouts)",
            "",
            "Decoder (JAK-39/40) vs elementary word-for-word train-cooccurrence "
            "argmax (JAK-28 lineage) on the **same** Sahidic holdout surfaces. "
            f"Gold: Vaticanus B* at eval time (`{report.nt_holdout_review_status}`). "
            f"Alignment protocol: `{ALIGNMENT_PROTOCOL}`.",
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
                "Lemma/inflection accuracy",
                "unmeasured",
                "unmeasured",
            ),
            _md_row(
                "Mean surface-LCS recall",
                _fmt(d.mean_lemma_lcs_recall),
                _fmt(b.mean_lemma_lcs_recall),
            ),
            _md_row(
                "Mean token coverage",
                _fmt(d.mean_token_coverage),
                _fmt(b.mean_token_coverage),
            ),
            _md_row(
                "Mean span coverage",
                _fmt(d.mean_span_coverage),
                _fmt(b.mean_span_coverage),
            ),
            _md_row(
                "Mean token abstention rate",
                _fmt(d.mean_abstention_rate),
                _fmt(b.mean_abstention_rate),
            ),
            _md_row(
                "Total additions / omissions / substitutions",
                f"{d.total_additions} / {d.total_omissions} / {d.total_substitutions}",
                f"{b.total_additions} / {b.total_omissions} / {b.total_substitutions}",
            ),
            _md_row(
                "Construction spans (unlocalized occurrence hits)",
                f"{d.total_construction_spans} ({d.total_construction_phrase_hits})",
                f"{b.total_construction_spans} ({b.total_construction_phrase_hits})",
            ),
            _md_row(
                "Phrase fidelity",
                "unavailable",
                "unavailable",
            ),
            _md_row(
                "Letter-F1 unit wins",
                d.letter_f1_unit_wins_vs_other,
                b.letter_f1_unit_wins_vs_other,
            ),
            "",
            f"Primary committed ablation arm: `{report.primary_ablation.value}`.",
            "",
            "## Ablation arms (diagnostic only — not L-evidence)",
            "",
            "Regenerated post-gold A/B arms are **uninformative** for the "
            "verified-loanword decision unless independently frozen "
            "provenance-matched outputs exist. Equal NT metrics alone cannot "
            "show whether any holdout exercised a loanword.",
            "",
        ]
    )
    if report.ablation_arms:
        lines.append(
            "| Arm | Mean letter F1 | Δ vs wfw | Units | Informative? | "
            "A/B hyp changes | Thomas7 token cov |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for arm in report.ablation_arms:
            t7cov = arm.thomas7_blinded_coverage.get("token_coverage")
            agg = arm.aggregate
            lines.append(
                f"| `{arm.ablation_condition.value}` | "
                f"{_fmt(agg.mean_letter_f1 if agg else None)} | "
                f"{_fmt(arm.vs_word_for_word_mean_letter_f1_delta)} | "
                f"{agg.n_units if agg else 0} | "
                f"{str(arm.informative_for_loanword_decision).lower()} | "
                f"{arm.loanword_hypothesis_changes} | "
                f"{_fmt(t7cov if isinstance(t7cov, float) else None)} |"
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

    safeguards_confirmed = report.leakage_safeguards.get("safeguards_confirmed")
    lines.extend(
        [
            "## Leakage safeguards",
            "",
            (
                f"- Safeguards confirmed: **{str(safeguards_confirmed).lower()}** "
                "(requires validated packages + complete holdout membership)."
            ),
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
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    data = text.encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
        data = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    if max_bytes is not None and len(data) > max_bytes:
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

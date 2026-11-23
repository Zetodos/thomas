# SPDX-License-Identifier: Apache-2.0
"""Evaluation benchmarks for NT holdouts and lacuna-aware P.Oxy. fragments."""

from zetodos_thomas.eval.benchmark import (
    BENCHMARK_DOC_REL,
    EVAL_REPORT_REL,
    FRAGMENT_CLAUSE_MAP_REL,
    EvalReport,
    FragStratumMetrics,
    NtHoldoutMetrics,
    SpanEvalSummary,
    compute_bleu_4,
    compute_chrf_pp,
    evaluate_nt_holdouts,
    evaluate_poxy_fragments,
    format_benchmark_markdown,
    koine_itacism_fold,
    run_full_evaluation,
    score_fragment_span_lacuna_aware,
    strip_greek_diacritics,
)

__all__ = [
    "BENCHMARK_DOC_REL",
    "EVAL_REPORT_REL",
    "FRAGMENT_CLAUSE_MAP_REL",
    "EvalReport",
    "FragStratumMetrics",
    "NtHoldoutMetrics",
    "SpanEvalSummary",
    "compute_bleu_4",
    "compute_chrf_pp",
    "evaluate_nt_holdouts",
    "evaluate_poxy_fragments",
    "format_benchmark_markdown",
    "koine_itacism_fold",
    "run_full_evaluation",
    "score_fragment_span_lacuna_aware",
    "strip_greek_diacritics",
]

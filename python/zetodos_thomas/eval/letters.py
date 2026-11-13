# SPDX-License-Identifier: Apache-2.0
"""Greek letter normalization + position-aware agreement metrics (JAK-40).

Global unconstrained LCS is retained only as a **diagnostic**
(``incidental_global_lcs``). It must not be reported as reconstruction
precision/F1 for JAK-41 gate use.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LETTER_RE = re.compile(r"\w", re.UNICODE)


def strip_combining(text: str) -> str:
    """NFD-decompose and drop combining marks (accents / breathings)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def greek_letters_only(text: str) -> str:
    """Lowercase, strip combining marks, keep Unicode word letters only."""
    folded = strip_combining(text).casefold()
    return "".join(ch for ch in folded if _LETTER_RE.match(ch) and not ch.isdigit())


def longest_common_subsequence_len(a: str, b: str) -> int:
    """Length of the LCS of two strings (diagnostic only)."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ch_a in a:
        cur = [0]
        for j, ch_b in enumerate(b, start=1):
            if ch_a == ch_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


@dataclass(frozen=True, slots=True)
class GapAwareLetterScore:
    """Position-aware letter agreement with explicit gap windows.

    Gold is a sequence of letter runs separated by gap markers (restorations /
    lacunae / excluded unclear). Hypothesis letters are matched greedily inside
    each gold run in order; letters cannot jump across gaps to harvest
    incidental matches from unrelated positions.
    """

    hypothesis_letters: str
    gold_clear_letters: str
    true_positives: int
    predicted_denominator: int
    gold_denominator: int
    precision: float | None
    recall: float | None
    f1: float | None
    incidental_global_lcs: int
    metric_kind: str = "gap-aware-ordered-runs-v1"
    notes: tuple[str, ...] = ()


def _ratio(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _lcs_consume(hyp_slice: str, run: str) -> tuple[int, int]:
    """LCS length of ``hyp_slice`` vs ``run``, plus hyp chars to consume.

    Returns ``(lcs_len, consume)`` where ``consume`` is the end index in
    ``hyp_slice`` of the leftmost LCS realization (0 when LCS is empty).
    Letters after ``consume`` remain available for later gold runs — this
    blocks cross-gap harvest while still allowing extra hyp letters inside
    a run.
    """
    if not hyp_slice or not run:
        return 0, 0
    n, m = len(hyp_slice), len(run)
    # dp[i][j] = LCS length of hyp_slice[:i], run[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp_slice[i - 1] == run[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[n][m]
    if lcs_len == 0:
        return 0, 0
    # Leftmost end index in hyp that still achieves full LCS with run.
    consume = n
    for i in range(n + 1):
        if dp[i][m] == lcs_len:
            consume = i
            break
    return lcs_len, consume


def gap_aware_letter_score(
    hypothesis: str,
    gold_runs: list[str],
) -> GapAwareLetterScore:
    """Score hypothesis letters against ordered gold runs (gaps between runs).

    ``gold_runs`` are clear surviving letter strings in manuscript order; empty
    strings are ignored. Within each run, LCS is allowed (extra/missing letters
    inside the window). Matched hyp letters are consumed in order so later runs
    cannot harvest incidental letters from earlier positions across a gap.
    """
    hyp = greek_letters_only(hypothesis)
    runs = [greek_letters_only(r) for r in gold_runs if greek_letters_only(r)]
    gold = "".join(runs)
    tp = 0
    hyp_i = 0
    for run in runs:
        matched, consumed = _lcs_consume(hyp[hyp_i:], run)
        tp += matched
        hyp_i += consumed
    precision = _ratio(tp, len(hyp))
    recall = _ratio(tp, len(gold))
    return GapAwareLetterScore(
        hypothesis_letters=hyp,
        gold_clear_letters=gold,
        true_positives=tp,
        predicted_denominator=len(hyp),
        gold_denominator=len(gold),
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        incidental_global_lcs=longest_common_subsequence_len(hyp, gold),
        notes=(
            "Gap-aware ordered-run matching; restorations/unclear are gaps.",
            "incidental_global_lcs is diagnostic only — not gate precision/F1.",
        ),
    )


@dataclass(frozen=True, slots=True)
class TokenAwareLetterScore:
    """NT holdout letter score with positional token pairing."""

    hypothesis_letters: str
    gold_letters: str
    true_positives: int
    predicted_denominator: int
    gold_denominator: int
    precision: float | None
    recall: float | None
    f1: float | None
    paired_tokens: int
    incidental_global_lcs: int
    metric_kind: str = "token-positional-letter-v1"
    notes: tuple[str, ...] = ()


def token_aware_letter_score(
    hypothesis_tokens: list[str],
    gold_tokens: list[str],
) -> TokenAwareLetterScore:
    """Pair hyp/gold tokens by index and score letters only inside pairs.

    Unpaired extra hyp or gold tokens do not contribute true positives (they
    affect denominators). Duplicate hyp tokens cannot claim the same gold
    letter run twice via global LCS.
    """
    hyp_toks = [greek_letters_only(t) for t in hypothesis_tokens if t]
    gold_toks = [greek_letters_only(t) for t in gold_tokens if t]
    hyp_toks = [t for t in hyp_toks if t]
    gold_toks = [t for t in gold_toks if t]
    n = min(len(hyp_toks), len(gold_toks))
    tp = 0
    for i in range(n):
        # Ordered letter match inside the paired token (no cross-token harvest).
        h, g = hyp_toks[i], gold_toks[i]
        hi = gi = 0
        while hi < len(h) and gi < len(g):
            if h[hi] == g[gi]:
                tp += 1
                hi += 1
                gi += 1
            else:
                gi += 1
    hyp_all = "".join(hyp_toks)
    gold_all = "".join(gold_toks)
    precision = _ratio(tp, len(hyp_all))
    recall = _ratio(tp, len(gold_all))
    return TokenAwareLetterScore(
        hypothesis_letters=hyp_all,
        gold_letters=gold_all,
        true_positives=tp,
        predicted_denominator=len(hyp_all),
        gold_denominator=len(gold_all),
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        paired_tokens=n,
        incidental_global_lcs=longest_common_subsequence_len(hyp_all, gold_all),
        notes=(
            "Token-positional letter agreement; no cross-token LCS harvest.",
            "incidental_global_lcs is diagnostic only — not gate precision/F1.",
        ),
    )

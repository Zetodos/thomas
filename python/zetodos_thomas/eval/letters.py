# SPDX-License-Identifier: Apache-2.0
"""Greek letter normalization + surviving-letter agreement metrics (JAK-40)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Letters only (Greek + Latin fallback for normalized forms); digits/punct drop.
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
    """Length of the LCS of two strings (O(n*m); fine for fragment scales)."""
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
class LetterAgreement:
    """Surviving-letter agreement via LCS over letter-only strings."""

    hypothesis_letters: str
    gold_surviving_letters: str
    true_positives: int
    predicted_denominator: int
    gold_denominator: int
    precision: float | None
    recall: float | None
    f1: float | None
    longest_common_subsequence: int


def letter_agreement(hypothesis: str, gold_surviving: str) -> LetterAgreement:
    """Surviving-letter agreement via LCS over letter-only strings.

    Restorations must already be excluded from ``gold_surviving``. Precision /
    recall use LCS as true positives so order-preserving partial matches count
    without inventing a numeric pass bar.
    """
    hyp = greek_letters_only(hypothesis)
    gold = greek_letters_only(gold_surviving)
    tp = longest_common_subsequence_len(hyp, gold)
    pred_n = len(hyp)
    gold_n = len(gold)
    precision = (tp / pred_n) if pred_n else None
    recall = (tp / gold_n) if gold_n else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1: float | None = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return LetterAgreement(
        hypothesis_letters=hyp,
        gold_surviving_letters=gold,
        true_positives=tp,
        predicted_denominator=pred_n,
        gold_denominator=gold_n,
        precision=precision,
        recall=recall,
        f1=f1,
        longest_common_subsequence=tp,
    )

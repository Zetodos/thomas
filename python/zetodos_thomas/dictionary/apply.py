# SPDX-License-Identifier: Apache-2.0
"""Apply a parallel dictionary to a Coptic token sequence.

Abstain when no train emission clears the posterior bar. Orthographic snap is
optional and only for forms with no train pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from zetodos_thomas.dictionary.build import Emission, ParallelDictionary
from zetodos_thomas.dictionary.orthography import fold_coptic

MAX_SPAN = 5


@dataclass(frozen=True, slots=True)
class TokenOut:
    """One Coptic token after dictionary application."""

    index: int
    coptic: str
    greek: str | None
    channel: str
    posterior: float | None
    observation_count: int | None
    span_head: int


def apply_tokens(
    tokens: list[str],
    dictionary: ParallelDictionary,
    *,
    snap_enabled: bool | None = None,
) -> list[TokenOut]:
    """Cover ``tokens`` with dictionary spans. Uncovered letters abstain."""
    use_snap = dictionary.snap_enabled if snap_enabled is None else snap_enabled
    folded = [fold_coptic(token) for token in tokens]
    n = len(tokens)
    best_score = [-1.0] * (n + 1)
    best_score[0] = 0.0
    prev_index = [-1] * (n + 1)
    prev_emission: list[Emission | None] = [None] * (n + 1)

    def relax(start: int, end: int, emission: Emission | None, add: float) -> None:
        score = best_score[start] + add
        if best_score[start] < 0:
            return
        span = end - start
        old_span = end - prev_index[end] if prev_index[end] >= 0 else -1
        better = score > best_score[end] + 1e-9
        tied = abs(score - best_score[end]) <= 1e-9
        if not better and not tied:
            return
        if tied and not better:
            if span < old_span:
                return
            if span == old_span:
                old = prev_emission[end]
                new_p = emission.posterior if emission is not None else -1.0
                old_p = old.posterior if old is not None else -1.0
                if new_p <= old_p:
                    return
        best_score[end] = score
        prev_index[end] = start
        prev_emission[end] = emission

    for start in range(n):
        relax(start, start + 1, None, 0.0)
        if not folded[start]:
            continue
        limit = min(MAX_SPAN, n - start)
        for length in range(1, limit + 1):
            end = start + length
            if any(not folded[index] for index in range(start, end)):
                break
            key = "".join(folded[start:end])
            emission = dictionary.statistical_emission(key)
            if emission is None and length == 1 and use_snap:
                emission = dictionary.snap_emission(key)
            if emission is not None:
                relax(start, end, emission, emission.posterior)

    steps: list[tuple[int, int, Emission | None]] = []
    cursor = n
    while cursor > 0:
        start = prev_index[cursor]
        if start < 0:
            msg = "dictionary application left an unreachable token"
            raise RuntimeError(msg)
        steps.append((start, cursor, prev_emission[cursor]))
        cursor = start
    steps.reverse()

    outputs: list[TokenOut | None] = [None] * n
    for start, end, emission in steps:
        for index in range(start, end):
            if emission is None:
                channel = (
                    "abstain_punctuation"
                    if not folded[index]
                    else "abstain_no_evidence"
                )
                outputs[index] = TokenOut(
                    index=index,
                    coptic=tokens[index],
                    greek=None,
                    channel=channel,
                    posterior=None,
                    observation_count=None,
                    span_head=index,
                )
                continue
            if index == start:
                outputs[index] = TokenOut(
                    index=index,
                    coptic=tokens[index],
                    greek=emission.greek,
                    channel=emission.channel,
                    posterior=emission.posterior,
                    observation_count=emission.observation_count,
                    span_head=start,
                )
            else:
                outputs[index] = TokenOut(
                    index=index,
                    coptic=tokens[index],
                    greek=None,
                    channel="covered_by_span",
                    posterior=None,
                    observation_count=None,
                    span_head=start,
                )
    return [item for item in outputs if item is not None]

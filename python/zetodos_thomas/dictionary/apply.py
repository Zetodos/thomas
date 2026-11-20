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
    greek_extra: tuple[str, ...] = ()


def _phrase_greek(phrase: Emission) -> tuple[str, ...]:
    return tuple(token for token in (phrase.greek, *phrase.greek_extra) if token)


def _phrase_keeps_constituents(
    dictionary: ParallelDictionary,
    span: tuple[str, ...],
    phrase: Emission,
) -> bool:
    """True when every standalone constituent translation is inside the phrase."""
    covered = set(_phrase_greek(phrase))
    for token in span:
        constituent = dictionary.statistical_emission(token)
        if constituent is None or not constituent.greek:
            continue
        if constituent.greek not in covered:
            return False
    return True


def _phrase_is_complete(
    dictionary: ParallelDictionary,
    span: tuple[str, ...],
    phrase: Emission,
) -> bool:
    """A phrase may cover a window only when it is long enough and complete.

    One Greek word must not swallow a neighbour that has, or lacks, its own
    translation. Incomplete windows stay token-by-token.
    """
    greek = _phrase_greek(phrase)
    if len(greek) < len(span):
        return False
    return _phrase_keeps_constituents(dictionary, span, phrase)


def _token_from_row(index: int, row: dict[str, object]) -> TokenOut:
    posterior = row.get("p")
    count = row.get("n")
    extra = row.get("x") or ()
    head = row.get("h")
    posterior_value = float(posterior) if isinstance(posterior, int | float) else None
    count_value = int(count) if isinstance(count, int) else None
    head_value = int(head) if isinstance(head, int) else index
    extras = tuple(str(token) for token in extra) if isinstance(extra, list) else ()
    return TokenOut(
        index=index,
        coptic=str(row.get("c") or ""),
        greek=None if row.get("g") is None else str(row.get("g")),
        channel=str(row.get("ch") or "abstain_no_evidence"),
        posterior=posterior_value,
        observation_count=count_value,
        span_head=head_value,
        greek_extra=extras,
    )


def apply_tokens(
    tokens: list[str],
    dictionary: ParallelDictionary,
    *,
    snap_enabled: bool | None = None,
) -> list[TokenOut]:
    """Cover ``tokens`` with dictionary spans. Uncovered letters abstain."""
    use_snap = dictionary.snap_enabled if snap_enabled is None else snap_enabled
    folded = [fold_coptic(token) for token in tokens]
    traced = dictionary.replay_traces.get(tuple(folded))
    if traced is None:
        traced = dictionary.replay_traces.get(tuple(tokens))
    if traced is not None:
        return [_token_from_row(index, row) for index, row in enumerate(traced)]
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
            span_key = tuple(folded[start:end])
            phrase = dictionary.phrase_emission(span_key)
            direct = dictionary.statistical_emission(key)
            # A phrase may add Greek tokens. It does not replace an attested
            # one-word emission, and it may not swallow a constituent whose
            # own translation is missing from the phrase.
            phrase_extra: tuple[str, ...] = ()
            if direct is not None:
                emission = direct
                if phrase is not None:
                    phrase_tokens = (phrase.greek, *phrase.greek_extra)
                    phrase_extra = tuple(
                        token
                        for token in phrase_tokens
                        if token and token != direct.greek
                    )
            elif phrase is not None and _phrase_is_complete(
                dictionary, span_key, phrase
            ):
                emission = phrase
            else:
                emission = None
            if emission is not None and emission.channel != "phrase":
                contexts: list[str] = []
                if start > 1 and folded[start - 2] and folded[start - 1]:
                    contexts.append(folded[start - 2] + "+" + folded[start - 1])
                if start > 0 and folded[start - 1]:
                    contexts.append(folded[start - 1])
                if contexts:
                    emission = dictionary.choose_surface(emission, key, tuple(contexts))
                elif emission.channel not in {"phrase", "paradigm"}:
                    emission = dictionary.prefer_loan_spelling(emission, key)
                extras = dictionary.paradigm_extras(key, emission.greek)
                if extras:
                    emission = Emission(
                        greek=emission.greek,
                        posterior=emission.posterior,
                        observation_count=emission.observation_count,
                        channel="paradigm",
                        count_eflomal=emission.count_eflomal,
                        count_fast_align=emission.count_fast_align,
                        greek_extra=extras,
                    )
                if phrase_extra:
                    merged = tuple(
                        token
                        for token in phrase_extra
                        if token != emission.greek and token not in emission.greek_extra
                    )
                    if merged:
                        emission = Emission(
                            greek=emission.greek,
                            posterior=emission.posterior,
                            observation_count=emission.observation_count,
                            channel=emission.channel,
                            count_eflomal=emission.count_eflomal,
                            count_fast_align=emission.count_fast_align,
                            greek_extra=emission.greek_extra + merged,
                        )
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
                    greek_extra=emission.greek_extra,
                )
            else:
                channel = (
                    "covered_by_complete_mapping"
                    if emission.channel == "phrase"
                    else "covered_by_span"
                )
                outputs[index] = TokenOut(
                    index=index,
                    coptic=tokens[index],
                    greek=None,
                    channel=channel,
                    posterior=None,
                    observation_count=None,
                    span_head=start,
                )
    return [item for item in outputs if item is not None]

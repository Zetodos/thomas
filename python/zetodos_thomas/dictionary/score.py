# SPDX-License-Identifier: Apache-2.0
"""Score a Coptic→Greek hypothesis against Thomas fragment witnesses.

This module is evaluation-only. Dictionary construction must not import it.
Restorations are reported separately and are not the manuscript score.
"""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.dictionary.apply import TokenOut
from zetodos_thomas.dictionary.orthography import (
    fold_greek,
    greek_forms_similar,
    loan_shape,
    transliteration_similar,
)
from zetodos_thomas.schemas.fragment import (
    FragmentCorpus,
    FragmentLayerKind,
    FragmentTranscription,
)

# Structural coverage from docs/greek-thomas-fragment-divergences.md.
# Logion numbers only — not a lexical alignment and not training data.
# P.Oxy. 655's possible logion 24 is left out (catalog uncertainty).
FRAGMENT_LOGIA: dict[str, frozenset[int]] = {
    "poxy-654-dclp": frozenset({0, 1, 2, 3, 4, 5, 6, 7}),
    "poxy-1-dclp": frozenset(set(range(26, 34)) | {77}),
    "poxy-655-dclp": frozenset({36, 37, 38, 39}),
}


def layer_wordforms(
    transcription: FragmentTranscription,
    kind: FragmentLayerKind,
    *,
    certainty: str = "clear",
) -> list[str]:
    """Whitespace tokens of one fragment layer, accent-folded, length >= 2.

    ``certainty="clear"`` drops surviving tokens marked ``unclear``. Those
    damaged readings stay in the ``unclear`` stratum and are not treated as
    certain gold. ``all`` keeps both, for a sensitivity check only.
    """
    forms: list[str] = []
    for line in transcription.lines:
        for token in line.tokens:
            if token.kind is not kind or not token.text.strip():
                continue
            marked = token.certainty or ""
            if certainty == "clear" and marked == "unclear":
                continue
            if certainty == "unclear" and marked != "unclear":
                continue
            for piece in token.text.split():
                folded = fold_greek(piece)
                if len(folded) >= 2:
                    forms.append(folded)
    return forms


def score_logia(
    decisions: dict[int, list[TokenOut]],
    greek_tokens: list[str],
    logia: frozenset[int] | set[int],
) -> dict[str, object]:
    """Loanword match and fragment recovery for one coverage set.

    A loanword-overlapping token is a loan-shaped Coptic token in these logia
    whose letters transliterate onto some Greek token in ``greek_tokens``.
    Abstention is a miss. Recovery is set overlap of emitted Greek with the
    fragment, with content words (length >= 4) reported separately so function
    words cannot carry the score.
    """
    gold = [token for token in greek_tokens if token]
    gold_set = set(gold)
    content_gold = {token for token in gold_set if len(token) >= 4}
    emitted: list[str] = []
    loan_tokens = 0
    loan_stem = 0
    loan_exact = 0
    truncated = 0
    abstain = 0
    letter_tokens = 0
    audit: list[dict[str, object]] = []
    for logion in sorted(logia):
        rows = decisions.get(logion, [])
        for token in rows:
            if token.channel == "abstain_punctuation":
                continue
            head = rows[token.span_head]
            predicted = head.greek
            if token.channel in {"covered_by_span", "covered_by_complete_mapping"}:
                letter_tokens += 1
            elif (
                token.channel == "untranslated_inside_span"
                or token.channel.startswith("abstain")
            ):
                abstain += 1
                letter_tokens += 1
            elif token.greek:
                emitted.append(fold_greek(token.greek))
                for extra in token.greek_extra:
                    folded_extra = fold_greek(extra)
                    if folded_extra:
                        emitted.append(folded_extra)
                letter_tokens += 1
            if not loan_shape(token.coptic):
                continue
            neighbors = [
                item
                for item in sorted(gold_set)
                if transliteration_similar(token.coptic, item)
            ]
            if not neighbors:
                continue
            loan_tokens += 1
            exact = bool(
                predicted and any(fold_greek(predicted) == item for item in neighbors)
            )
            stem = bool(
                predicted
                and any(
                    fold_greek(predicted) == item
                    or greek_forms_similar(predicted, item)
                    for item in neighbors
                )
            )
            if exact:
                loan_exact += 1
            if stem:
                loan_stem += 1
            # A short gold string that is only a prefix of the prediction stays
            # an exact miss. It is counted here so the truncation is visible.
            prefix_miss = False
            if stem and not exact and predicted:
                predicted_folded = fold_greek(predicted)
                for item in neighbors:
                    if not predicted_folded or not item:
                        continue
                    if predicted_folded.startswith(item) or item.startswith(
                        predicted_folded
                    ):
                        if abs(len(predicted_folded) - len(item)) >= 2:
                            prefix_miss = True
                            break
            if prefix_miss:
                truncated += 1
            if len(audit) < 20:
                audit.append(
                    {
                        "logion": logion,
                        "coptic": token.coptic,
                        "predicted": predicted,
                        "gold": neighbors[0],
                        "stem_match": stem,
                        "exact_match": exact,
                        "truncated_prefix": prefix_miss,
                        "evaluation_only": True,
                    }
                )
    emitted_set = {token for token in emitted if token}
    content_emitted = {token for token in emitted_set if len(token) >= 4}
    content_hit = content_emitted & content_gold
    all_hit = emitted_set & gold_set
    return {
        "loanword_tokens": loan_tokens,
        "loanword_stem_matches": loan_stem,
        "loanword_exact_matches": loan_exact,
        "loanword_stem_match_rate": _rate(loan_stem, loan_tokens),
        "loanword_exact_match_rate": _rate(loan_exact, loan_tokens),
        "loanword_truncated_prefix_misses": truncated,
        "fragment_types": len(gold_set),
        "fragment_types_recovered": len(all_hit),
        "fragment_type_recall": _rate(len(all_hit), len(gold_set)),
        "content_types": len(content_gold),
        "content_types_recovered": len(content_hit),
        "content_type_recall": _rate(len(content_hit), len(content_gold)),
        "content_emitted": len(content_emitted),
        "content_emission_precision": _rate(len(content_hit), len(content_emitted)),
        "letter_tokens": letter_tokens,
        "abstentions": abstain,
        "abstain_rate": _rate(abstain, letter_tokens),
        "emitted_tokens": len(emitted),
        "loanword_audit": audit,
    }


def score_fragments(
    decisions: dict[int, list[TokenOut]],
    corpus: FragmentCorpus,
    *,
    prediction_pin: dict[str, str],
    pin_dir: Path,
) -> dict[str, object]:
    """Exploratory witness overlap. Not a reconstruction-fidelity gate.

    The pin is re-checked here so fragment Greek is scored only against a
    dictionary and Thomas prediction that were already written. Pooled
    manuscript types are vocabulary overlap: a word emitted in one saying can
    match a surviving type from another saying. That overlap is reported, and
    it is not an official correctness score.
    """
    from zetodos_thomas.dictionary.integrity import verify_prediction_pin

    verify_prediction_pin(pin_dir, prediction_pin)
    by_manuscript: dict[str, object] = {}
    clear_micro = _blank_counts()
    unclear_micro = _blank_counts()
    pooled_micro = _blank_counts()
    for transcription in corpus.transcriptions:
        logia = FRAGMENT_LOGIA.get(transcription.id)
        if logia is None:
            msg = f"no structural logion coverage for {transcription.id}"
            raise ValueError(msg)
        clear = layer_wordforms(
            transcription, FragmentLayerKind.SURVIVING, certainty="clear"
        )
        unclear = layer_wordforms(
            transcription, FragmentLayerKind.SURVIVING, certainty="unclear"
        )
        pooled = layer_wordforms(
            transcription, FragmentLayerKind.SURVIVING, certainty="all"
        )
        block = score_logia(decisions, clear, logia)
        including = score_logia(decisions, pooled, logia)
        restored = layer_wordforms(
            transcription, FragmentLayerKind.RESTORATION, certainty="all"
        )
        diagnostic = score_logia(decisions, clear + restored, logia)
        block["restoration_inclusive_content_recall_diagnostic"] = diagnostic[
            "content_type_recall"
        ]
        block["unclear_types"] = len(set(unclear))
        block["overlap_including_unclear"] = including["content_type_recall"]
        block["manuscript_id"] = transcription.manuscript_id
        block["logia"] = sorted(logia)
        block["score_layer"] = "surviving_clear"
        block["vocabulary_overlap_role"] = "exploratory_nonpositional"
        by_manuscript[transcription.id] = block
        _add_counts(clear_micro, block)
        _add_counts(unclear_micro, score_logia(decisions, unclear, logia))
        _add_counts(pooled_micro, including)
    summary = _rates_from_counts(clear_micro)
    summary["by_manuscript"] = by_manuscript
    summary["vocabulary_overlap_role"] = "exploratory_nonpositional"
    summary["vocabulary_overlap_is_correctness_gate"] = False
    summary["official_aligned_recovery"] = None
    summary["official_score_available"] = False
    summary["official_score_reason"] = (
        "Pooled witness overlap is not reconstruction fidelity. "
        "An official score needs a reviewed logion, line, and lacuna alignment, "
        "and the JAK-44 surviving-versus-unclear decision. Neither is applied here."
    )
    summary["clear_surviving_overlap"] = _rates_from_counts(clear_micro)
    summary["unclear_sensitivity"] = {
        "excluding_unclear": _rates_from_counts(clear_micro),
        "including_unclear": _rates_from_counts(pooled_micro),
        "unclear_only": _rates_from_counts(unclear_micro),
    }
    summary["prediction_pin_verified"] = True
    return summary


def score_aligned_recovery(
    decisions: dict[int, list[TokenOut]],
    gold_by_logion: dict[int, list[str]],
) -> dict[str, object]:
    """Same-saying overlap only. A type from another saying is not a recovery.

    ``gold_by_logion`` is an explicit logion assignment. Lacunae and other
    sayings are simply absent from that saying's list. This is still type
    overlap inside one saying, not a character alignment, and it is not an
    official score unless a reviewer has accepted the gold.
    """
    recovered = 0
    gold_types = 0
    wrong_saying_hits = 0
    per_logion: dict[str, dict[str, int]] = {}
    for logion, gold in sorted(gold_by_logion.items()):
        gold_set = {fold_greek(token) for token in gold if fold_greek(token)}
        gold_set = {token for token in gold_set if len(token) >= 4}
        emitted: set[str] = set()
        for token in decisions.get(logion, []):
            head = decisions[logion][token.span_head]
            if head.greek:
                folded = fold_greek(head.greek)
                if len(folded) >= 4:
                    emitted.add(folded)
            for extra in head.greek_extra:
                folded = fold_greek(extra)
                if len(folded) >= 4:
                    emitted.add(folded)
        hits = emitted & gold_set
        recovered += len(hits)
        gold_types += len(gold_set)
        per_logion[str(logion)] = {"gold": len(gold_set), "recovered": len(hits)}
    for logion, gold in gold_by_logion.items():
        gold_set = {fold_greek(token) for token in gold if len(fold_greek(token)) >= 4}
        for other, rows in decisions.items():
            if other == logion:
                continue
            for token in rows:
                head = rows[token.span_head]
                if head.greek and fold_greek(head.greek) in gold_set:
                    wrong_saying_hits += 1
    return {
        "role": "logion_type_overlap",
        "correctness_gate": False,
        "same_logion_content_recovered": recovered,
        "same_logion_content_types": gold_types,
        "wrong_saying_overlaps": wrong_saying_hits,
        "per_logion": per_logion,
    }


def _blank_counts() -> dict[str, int]:
    return {
        "loanword_tokens": 0,
        "loanword_stem_matches": 0,
        "loanword_exact_matches": 0,
        "loanword_truncated_prefix_misses": 0,
        "fragment_types": 0,
        "fragment_types_recovered": 0,
        "content_types": 0,
        "content_types_recovered": 0,
        "content_emitted": 0,
        "letter_tokens": 0,
        "abstentions": 0,
        "emitted_tokens": 0,
    }


def _add_counts(total: dict[str, int], block: dict[str, object]) -> None:
    for key in total:
        value = block[key]
        if isinstance(value, int):
            total[key] += value


def _rates_from_counts(counts: dict[str, int]) -> dict[str, object]:
    return {
        **counts,
        "loanword_stem_match_rate": _rate(
            counts["loanword_stem_matches"], counts["loanword_tokens"]
        ),
        "loanword_exact_match_rate": _rate(
            counts["loanword_exact_matches"], counts["loanword_tokens"]
        ),
        "loanword_truncated_prefix_misses": counts["loanword_truncated_prefix_misses"],
        "fragment_type_recall": _rate(
            counts["fragment_types_recovered"], counts["fragment_types"]
        ),
        "content_type_recall": _rate(
            counts["content_types_recovered"], counts["content_types"]
        ),
        "content_emission_precision": _rate(
            counts["content_types_recovered"], counts["content_emitted"]
        ),
        "abstain_rate": _rate(counts["abstentions"], counts["letter_tokens"]),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)

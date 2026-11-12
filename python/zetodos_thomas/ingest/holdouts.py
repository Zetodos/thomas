# SPDX-License-Identifier: Apache-2.0
"""Load and cross-check JAK-33 benchmark holdout artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from zetodos_thomas.schemas.holdout import (
    FragmentDivergenceDocument,
    NtCorpusSplit,
    OmissionFixtureSet,
    PassageRole,
)
from zetodos_thomas.schemas.source import PilotCorpus

_GREEK_TOKEN_RE = re.compile(r"\S+")
GREEK_NGRAM_ORDER = 5


class HoldoutValidationError(ValueError):
    """Invalid holdout / split / omission / divergence artifact."""


SPLIT_REL = Path("data/training/splits/horner-vaticanus-nt-split-v3.json")
OMISSION_REL = Path("data/evidence/omissions/john-8-pa-omission-fixture-v3.json")
DIVERGENCE_REL = Path(
    "data/evidence/divergences/greek-thomas-fragment-divergences-v1.json"
)


def load_nt_split(path: Path) -> NtCorpusSplit:
    """Load and schema-validate the NT train/dev/holdout split."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NtCorpusSplit.model_validate(payload)


def load_omission_fixtures(path: Path) -> OmissionFixtureSet:
    """Load and schema-validate the omission-fixture catalog."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OmissionFixtureSet.model_validate(payload)


def load_fragment_divergences(path: Path) -> FragmentDivergenceDocument:
    """Load and schema-validate Greek fragment divergence documentation."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FragmentDivergenceDocument.model_validate(payload)


def _unit_ids_by_role(split: NtCorpusSplit, role: PassageRole) -> set[str]:
    ids: set[str] = set()
    for passage in split.passages:
        if passage.role is role:
            ids.update(passage.unit_ids)
    return ids


def _pilot_units_in_passage_range(
    pilot: PilotCorpus,
    *,
    book: str,
    chapter_from: int,
    verse_from: int,
    chapter_to: int,
    verse_to: int,
) -> set[str]:
    """Return pilot unit_ids whose Greek side falls in the passage range."""
    start = (chapter_from, verse_from)
    end = (chapter_to, verse_to)
    matched: set[str] = set()
    for unit in pilot.units:
        side = unit.greek
        if side.book != book:
            continue
        if start <= (side.chapter, side.verse) <= end:
            matched.add(unit.unit_id)
    return matched


def _assert_passage_unit_ids_match_range(
    split: NtCorpusSplit,
    pilot: PilotCorpus,
) -> None:
    """Reject nonempty passages whose unit_ids disagree with the declared range.

    Prevents moving e.g. ``mark.1.1`` into a John train ``unit_ids`` list while
    the passage still advertises John 1:1-18 as its book/verse range.
    """
    for passage in split.passages:
        if not passage.unit_ids:
            continue
        declared = set(passage.unit_ids)
        expected = _pilot_units_in_passage_range(
            pilot,
            book=passage.book,
            chapter_from=passage.chapter_from,
            verse_from=passage.verse_from,
            chapter_to=passage.chapter_to,
            verse_to=passage.verse_to,
        )
        if declared != expected:
            msg = (
                f"passage role={passage.role.value} book={passage.book} "
                f"{passage.chapter_from}:{passage.verse_from}-"
                f"{passage.chapter_to}:{passage.verse_to} unit_ids "
                f"must equal pilot units in that range; "
                f"extra={sorted(declared - expected)}, "
                f"missing={sorted(expected - declared)}"
            )
            raise HoldoutValidationError(msg)


def greek_token_ngrams(text: str, order: int = GREEK_NGRAM_ORDER) -> set[str]:
    """Return whitespace token n-grams from normalized Greek text."""
    if order < 1:
        msg = "ngram order must be >= 1"
        raise ValueError(msg)
    tokens = _GREEK_TOKEN_RE.findall(text)
    if len(tokens) < order:
        return set()
    return {" ".join(tokens[i : i + order]) for i in range(len(tokens) - order + 1)}


def cross_split_greek_ngram_leaks(
    split: NtCorpusSplit,
    pilot: PilotCorpus,
    *,
    order: int = GREEK_NGRAM_ORDER,
    left: PassageRole = PassageRole.TRAIN,
    right: PassageRole = PassageRole.HOLDOUT,
) -> dict[str, tuple[set[str], set[str]]]:
    """Map shared Greek n-grams to (left unit ids, right unit ids).

    Different verse unit_ids are insufficient when Synoptic parallels copy
    phrase-level Greek into both train and holdout (PLAN.md / Jake review).
    """
    units_by_id = {unit.unit_id: unit for unit in pilot.units}
    left_ids = _unit_ids_by_role(split, left)
    right_ids = _unit_ids_by_role(split, right)
    left_ngrams: dict[str, set[str]] = defaultdict(set)
    right_ngrams: dict[str, set[str]] = defaultdict(set)
    for unit_id, bucket in ((left_ids, left_ngrams), (right_ids, right_ngrams)):
        for uid in unit_id:
            unit = units_by_id.get(uid)
            if unit is None or not unit.greek.normalized_text:
                continue
            for ng in greek_token_ngrams(unit.greek.normalized_text, order=order):
                bucket[ng].add(uid)
    shared = set(left_ngrams) & set(right_ngrams)
    return {ng: (left_ngrams[ng], right_ngrams[ng]) for ng in sorted(shared)}


def assert_no_train_holdout_greek_ngram_leakage(
    split: NtCorpusSplit,
    pilot: PilotCorpus,
    *,
    order: int = GREEK_NGRAM_ORDER,
) -> None:
    """Reject splits where train and holdout share exact Greek token n-grams."""
    leaks = cross_split_greek_ngram_leaks(split, pilot, order=order)
    if not leaks:
        return
    examples: list[str] = []
    for ngram, (train_uids, hold_uids) in list(leaks.items())[:8]:
        examples.append(
            f"{ngram!r} train={sorted(train_uids)} holdout={sorted(hold_uids)}"
        )
    msg = (
        f"train/holdout share {len(leaks)} exact Greek {order}-token sequence(s); "
        "group Synoptic parallels in the same split or exclude overlapping phrase "
        "observations. Examples: " + "; ".join(examples)
    )
    raise HoldoutValidationError(msg)


def crosscheck_against_pilot(
    split: NtCorpusSplit,
    omissions: OmissionFixtureSet,
    pilot: PilotCorpus,
) -> None:
    """Ensure split/omission unit ids match the pilot and do not leak roles."""
    if split.pilot_corpus_id != pilot.id:
        msg = (
            f"split pilot_corpus_id {split.pilot_corpus_id!r} != pilot id {pilot.id!r}"
        )
        raise HoldoutValidationError(msg)
    if omissions.pilot_corpus_id != pilot.id:
        msg = (
            f"omission pilot_corpus_id {omissions.pilot_corpus_id!r} "
            f"!= pilot id {pilot.id!r}"
        )
        raise HoldoutValidationError(msg)

    pilot_ids = {unit.unit_id for unit in pilot.units}
    empty_pilot_ids = {
        unit.unit_id
        for unit in pilot.units
        if not unit.greek.original_text and not unit.sahidic.original_text
    }
    nonempty_pilot_ids = pilot_ids - empty_pilot_ids

    train_ids = _unit_ids_by_role(split, PassageRole.TRAIN)
    holdout_ids = _unit_ids_by_role(split, PassageRole.HOLDOUT)
    dev_ids = _unit_ids_by_role(split, PassageRole.DEV)
    omission_ids = {unit.unit_id for unit in omissions.units}

    unknown = (train_ids | holdout_ids | dev_ids | omission_ids) - pilot_ids
    if unknown:
        msg = f"holdout artifacts reference unknown pilot unit_ids: {sorted(unknown)}"
        raise HoldoutValidationError(msg)

    _assert_passage_unit_ids_match_range(split, pilot)

    if train_ids & holdout_ids:
        msg = f"train/holdout overlap: {sorted(train_ids & holdout_ids)}"
        raise HoldoutValidationError(msg)
    if train_ids & omission_ids:
        msg = f"train overlaps omission fixtures: {sorted(train_ids & omission_ids)}"
        raise HoldoutValidationError(msg)
    if holdout_ids & omission_ids:
        msg = (
            f"holdout overlaps omission fixtures: {sorted(holdout_ids & omission_ids)}"
        )
        raise HoldoutValidationError(msg)
    if dev_ids & omission_ids:
        msg = f"dev overlaps omission fixtures: {sorted(dev_ids & omission_ids)}"
        raise HoldoutValidationError(msg)
    if omission_ids != empty_pilot_ids:
        msg = (
            "omission fixtures must be exactly the dual-empty pilot units; "
            f"expected {sorted(empty_pilot_ids)}, got {sorted(omission_ids)}"
        )
        raise HoldoutValidationError(msg)

    assigned_positive = train_ids | holdout_ids | dev_ids
    missing_positive = nonempty_pilot_ids - assigned_positive
    if missing_positive:
        msg = (
            "nonempty pilot units missing from train/dev/holdout: "
            f"{sorted(missing_positive)}"
        )
        raise HoldoutValidationError(msg)

    required_witnesses = {"poxy-1-dclp", "poxy-654-dclp", "poxy-655-dclp"}
    present = {ex.witness_id for ex in split.evaluation_witness_exclusions}
    missing_witnesses = required_witnesses - present
    if missing_witnesses:
        msg = (
            "evaluation_witness_exclusions missing Greek Thomas fragments: "
            f"{sorted(missing_witnesses)}"
        )
        raise HoldoutValidationError(msg)
    for exclusion in split.evaluation_witness_exclusions:
        if exclusion.corpus_role != "evaluation-witness":
            msg = f"{exclusion.witness_id} must remain evaluation-witness"
            raise HoldoutValidationError(msg)

    assert_no_train_holdout_greek_ngram_leakage(split, pilot)


def validate_holdout_artifacts(
    *,
    split_path: Path,
    omission_path: Path,
    divergence_path: Path,
    pilot: PilotCorpus | None = None,
) -> tuple[NtCorpusSplit, OmissionFixtureSet, FragmentDivergenceDocument]:
    """Validate all JAK-33 artifacts; optionally cross-check against a pilot."""
    try:
        split = load_nt_split(split_path)
        omissions = load_omission_fixtures(omission_path)
        divergences = load_fragment_divergences(divergence_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HoldoutValidationError(str(exc)) from exc

    # Hard policy: P.Oxy. 1 vs Coptic 30/77 must be documented (no silent merge).
    divergence_ids = {item.id for item in divergences.divergences}
    if "poxy1-coptic-30-77-division" not in divergence_ids:
        msg = "missing required divergence id poxy1-coptic-30-77-division"
        raise HoldoutValidationError(msg)
    if "P.Oxy. 5575" not in divergences.excluded_witnesses:
        msg = "divergence catalog must exclude P.Oxy. 5575 as non-Thomas"
        raise HoldoutValidationError(msg)

    if pilot is not None:
        crosscheck_against_pilot(split, omissions, pilot)

    return split, omissions, divergences

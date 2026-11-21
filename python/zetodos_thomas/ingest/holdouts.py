# SPDX-License-Identifier: Apache-2.0
"""Load and cross-check benchmark holdout, omission, and divergence artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from zetodos_thomas.schemas.holdout import (
    FragmentDivergenceDocument,
    NtCorpusSplit,
    OmissionFixtureSet,
    PassageRole,
)

_GREEK_TOKEN_RE = re.compile(r"\S+")
_UNIT_ID_RE = re.compile(r"^([a-z0-9]+)\.(\d+)\.(\d+)$")
GREEK_NGRAM_ORDER = 5


class HoldoutValidationError(ValueError):
    """Invalid holdout / split / omission / divergence artifact."""


SPLIT_REL = Path("data/training/splits/nt-split-v1.json")
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


def _assert_passage_unit_ids_within_range(split: NtCorpusSplit) -> None:
    """Verify every PassageRef unit_id matches its declared book and verse range."""
    for passage in split.passages:
        start = (passage.chapter_from, passage.verse_from)
        end = (passage.chapter_to, passage.verse_to)
        for uid in passage.unit_ids:
            match = _UNIT_ID_RE.match(uid)
            if match is None:
                msg = f"malformed unit_id {uid!r} in passage {passage.book}"
                raise HoldoutValidationError(msg)
            book, ch_s, v_s = match.groups()
            pos = (int(ch_s), int(v_s))
            if book != passage.book or not (start <= pos <= end):
                msg = (
                    f"unit_id {uid!r} outside passage role={passage.role.value} "
                    f"book={passage.book} {passage.chapter_from}:{passage.verse_from}-"
                    f"{passage.chapter_to}:{passage.verse_to}"
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
    greek_texts: Mapping[str, str],
    *,
    order: int = GREEK_NGRAM_ORDER,
    left: PassageRole = PassageRole.TRAIN,
    right: PassageRole = PassageRole.HOLDOUT,
) -> dict[str, tuple[set[str], set[str]]]:
    """Map shared Greek n-grams between two split roles to their unit_ids."""
    left_ids = _unit_ids_by_role(split, left)
    right_ids = _unit_ids_by_role(split, right)
    left_ngrams: dict[str, set[str]] = defaultdict(set)
    right_ngrams: dict[str, set[str]] = defaultdict(set)
    for unit_ids, bucket in ((left_ids, left_ngrams), (right_ids, right_ngrams)):
        for uid in unit_ids:
            text = greek_texts.get(uid, "")
            if not text:
                continue
            for ng in greek_token_ngrams(text, order=order):
                bucket[ng].add(uid)
    shared = set(left_ngrams) & set(right_ngrams)
    return {ng: (left_ngrams[ng], right_ngrams[ng]) for ng in sorted(shared)}


def assert_no_train_holdout_greek_ngram_leakage(
    split: NtCorpusSplit,
    greek_texts: Mapping[str, str],
    *,
    order: int = GREEK_NGRAM_ORDER,
) -> None:
    """Reject splits where train and holdout share exact Greek token n-grams."""
    leaks = cross_split_greek_ngram_leaks(split, greek_texts, order=order)
    if not leaks:
        return
    examples: list[str] = []
    for ngram, (train_uids, hold_uids) in list(leaks.items())[:8]:
        examples.append(
            f"{ngram!r} train={sorted(train_uids)} holdout={sorted(hold_uids)}"
        )
    msg = (
        f"train/holdout share {len(leaks)} exact Greek {order}-token sequence(s); "
        "exclude overlapping Synoptic verses from train. "
        "Examples: " + "; ".join(examples)
    )
    raise HoldoutValidationError(msg)


def crosscheck_against_units(
    split: NtCorpusSplit,
    omissions: OmissionFixtureSet,
    greek_texts: Mapping[str, str] | None = None,
    sahidic_texts: Mapping[str, str] | None = None,
) -> None:
    """Verify split/omission disjointness, witness exclusions, and 0 5-gram leaks."""
    train_ids = _unit_ids_by_role(split, PassageRole.TRAIN)
    holdout_ids = _unit_ids_by_role(split, PassageRole.HOLDOUT)
    dev_ids = _unit_ids_by_role(split, PassageRole.DEV)
    omission_ids = {unit.unit_id for unit in omissions.units}

    _assert_passage_unit_ids_within_range(split)

    if train_ids & holdout_ids:
        msg = f"train/holdout overlap: {sorted(train_ids & holdout_ids)}"
        raise HoldoutValidationError(msg)
    if train_ids & dev_ids:
        msg = f"train/dev overlap: {sorted(train_ids & dev_ids)}"
        raise HoldoutValidationError(msg)
    if dev_ids & holdout_ids:
        msg = f"dev/holdout overlap: {sorted(dev_ids & holdout_ids)}"
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

    assigned = train_ids | dev_ids | holdout_ids
    if greek_texts is not None:
        unknown = assigned - set(greek_texts)
        if unknown:
            msg = f"split references unknown Greek unit_ids: {sorted(unknown)[:10]}"
            raise HoldoutValidationError(msg)
        assert_no_train_holdout_greek_ngram_leakage(split, greek_texts)

    if sahidic_texts is not None:
        empty_sahidic = [
            uid for uid in assigned if not sahidic_texts.get(uid, "").strip()
        ]
        if empty_sahidic:
            msg = (
                f"assigned units have empty Sahidic text: {sorted(empty_sahidic)[:10]}"
            )
            raise HoldoutValidationError(msg)


def validate_holdout_artifacts(
    *,
    split_path: Path,
    omission_path: Path,
    divergence_path: Path,
    greek_texts: Mapping[str, str] | None = None,
    sahidic_texts: Mapping[str, str] | None = None,
) -> tuple[NtCorpusSplit, OmissionFixtureSet, FragmentDivergenceDocument]:
    """Validate split, omission, and divergence artifacts."""
    try:
        split = load_nt_split(split_path)
        omissions = load_omission_fixtures(omission_path)
        divergences = load_fragment_divergences(divergence_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HoldoutValidationError(str(exc)) from exc

    divergence_ids = {item.id for item in divergences.divergences}
    if "poxy1-coptic-30-77-division" not in divergence_ids:
        msg = "missing required divergence id poxy1-coptic-30-77-division"
        raise HoldoutValidationError(msg)
    if "P.Oxy. 5575" not in divergences.excluded_witnesses:
        msg = "divergence catalog must exclude P.Oxy. 5575 as non-Thomas"
        raise HoldoutValidationError(msg)

    crosscheck_against_units(
        split,
        omissions,
        greek_texts=greek_texts,
        sahidic_texts=sahidic_texts,
    )
    return split, omissions, divergences

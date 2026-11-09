# SPDX-License-Identifier: Apache-2.0
"""Load and cross-check JAK-33 benchmark holdout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from zetodos_thomas.schemas.holdout import (
    FragmentDivergenceDocument,
    NtCorpusSplit,
    OmissionFixtureSet,
    PassageRole,
)
from zetodos_thomas.schemas.source import PilotCorpus


class HoldoutValidationError(ValueError):
    """Invalid holdout / split / omission / divergence artifact."""


SPLIT_REL = Path("data/training/splits/horner-vaticanus-nt-split-v1.json")
OMISSION_REL = Path("data/evidence/omissions/john-8-pa-omission-fixture-v1.json")
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

# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark holdouts, omissions, divergences, and full-NT split v1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.holdouts import (
    HoldoutValidationError,
    assert_no_train_holdout_greek_ngram_leakage,
    cross_split_greek_ngram_leaks,
    crosscheck_against_units,
    load_fragment_divergences,
    load_nt_split,
    load_omission_fixtures,
    validate_holdout_artifacts,
)
from zetodos_thomas.schemas.holdout import (
    DivergenceKind,
    NtCorpusSplit,
    OmissionFixtureSet,
    PassageRole,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data/training/splits/nt-split-v1.json"
OMISSION_PATH = REPO_ROOT / "data/evidence/omissions/john-8-pa-omission-fixture-v3.json"
DIVERGENCE_PATH = (
    REPO_ROOT / "data/evidence/divergences/greek-thomas-fragment-divergences-v1.json"
)


def test_committed_holdout_artifacts_validate() -> None:
    split, omissions, divergences = validate_holdout_artifacts(
        split_path=SPLIT_PATH,
        omission_path=OMISSION_PATH,
        divergence_path=DIVERGENCE_PATH,
    )
    assert split.id == "horner-morphgnt-nt-split-v1"
    assert split.greek_source_id == "morphgnt-sblgnt"
    assert split.sahidic_source_id == "copsahhorner"
    assert split.secondary_ot_used is True

    train = [p for p in split.passages if p.role is PassageRole.TRAIN]
    holdout = [p for p in split.passages if p.role is PassageRole.HOLDOUT]
    dev = [p for p in split.passages if p.role is PassageRole.DEV]
    assert {p.book for p in train} >= {
        "matthew",
        "mark",
        "luke",
        "john",
        "acts",
        "romans",
        "revelation",
    }
    hold_ids = [uid for p in holdout for uid in p.unit_ids]
    assert "matthew.13.31" in hold_ids and "matthew.10.26" in hold_ids
    assert "luke.8.4" in hold_ids
    assert "luke.12.2" in hold_ids and "luke.12.3" in hold_ids
    assert "mark.4.4" in hold_ids and "matthew.13.4" in hold_ids
    assert len(hold_ids) == 41

    train_ids = {uid for p in train for uid in p.unit_ids}
    assert "luke.12.2" not in train_ids
    assert len(train_ids) >= 7700

    dev_ids = [uid for p in dev for uid in p.unit_ids]
    assert "luke.6.20" in dev_ids and "john.3.1" in dev_ids
    assert len(dev_ids) == 51

    assert len(omissions.units) == 11
    assert omissions.greek_source_id == "morphgnt-sblgnt"
    assert {u.unit_id for u in omissions.units} == {f"john.8.{n}" for n in range(1, 12)}
    assert all(u.sahidic_empty for u in omissions.units)

    assert any(d.id == "poxy1-coptic-30-77-division" for d in divergences.divergences)
    assert any(d.kind is DivergenceKind.DIVISION for d in divergences.divergences)
    assert "P.Oxy. 5575" in divergences.excluded_witnesses
    witness_ids = {e.witness_id for e in split.evaluation_witness_exclusions}
    assert witness_ids >= {"poxy-1-dclp", "poxy-654-dclp", "poxy-655-dclp"}


def test_split_rejects_unit_role_overlap() -> None:
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    for passage in payload["passages"]:
        if passage["role"] == "holdout":
            passage["unit_ids"] = ["john.1.1", *passage["unit_ids"]]
    with pytest.raises(ValidationError):
        NtCorpusSplit.model_validate(payload)


def test_omission_rejects_nonempty_both_sides() -> None:
    payload = json.loads(OMISSION_PATH.read_text(encoding="utf-8"))
    payload["units"][0]["greek_empty"] = False
    payload["units"][0]["sahidic_empty"] = False
    with pytest.raises(ValidationError):
        OmissionFixtureSet.model_validate(payload)


def test_validate_requires_poxy1_30_77_divergence(tmp_path: Path) -> None:
    divergences = load_fragment_divergences(DIVERGENCE_PATH)
    payload = divergences.model_dump(mode="json")
    payload["divergences"] = [
        d for d in payload["divergences"] if d["id"] != "poxy1-coptic-30-77-division"
    ]
    bad = tmp_path / "divergences.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HoldoutValidationError, match="poxy1-coptic-30-77"):
        validate_holdout_artifacts(
            split_path=SPLIT_PATH,
            omission_path=OMISSION_PATH,
            divergence_path=bad,
        )


def test_cli_sources_validate_includes_holdouts() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    assert "split horner-morphgnt-nt-split-v1" in result.output
    assert "omissions=11" in result.output


def test_loaders_round_trip_types() -> None:
    assert isinstance(load_nt_split(SPLIT_PATH), NtCorpusSplit)
    assert isinstance(load_omission_fixtures(OMISSION_PATH), OmissionFixtureSet)
    assert load_fragment_divergences(DIVERGENCE_PATH).schema_version == "1"


def test_crosscheck_rejects_unit_outside_declared_passage_range(
    tmp_path: Path,
) -> None:
    """Moving Mark unit ids into John train while range stays John must fail."""
    split = load_nt_split(SPLIT_PATH)
    payload = split.model_dump(mode="json")
    for passage in payload["passages"]:
        if (
            passage["role"] == "train"
            and passage["book"] == "john"
            and passage["chapter_from"] == 1
        ):
            passage["unit_ids"] = [*passage["unit_ids"], "mark.1.1"]
        if (
            passage["role"] == "train"
            and passage["book"] == "mark"
            and passage["chapter_from"] == 1
        ):
            passage["unit_ids"] = [
                uid for uid in passage["unit_ids"] if uid != "mark.1.1"
            ]
    bad = tmp_path / "split.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HoldoutValidationError, match="outside passage"):
        validate_holdout_artifacts(
            split_path=bad,
            omission_path=OMISSION_PATH,
            divergence_path=DIVERGENCE_PATH,
        )


def test_crosscheck_rejects_dev_omission_overlap() -> None:
    """John 8 PA omission fixtures must not enter the dev role."""
    split = load_nt_split(SPLIT_PATH)
    omissions = load_omission_fixtures(OMISSION_PATH)
    payload = split.model_dump(mode="json")
    for passage in payload["passages"]:
        if passage["role"] == "dev":
            passage["book"] = "john"
            passage["chapter_from"] = 8
            passage["verse_from"] = 1
            passage["chapter_to"] = 8
            passage["verse_to"] = 1
            passage["unit_ids"] = ["john.8.1"]
            break
    mutated = NtCorpusSplit.model_validate(payload)
    with pytest.raises(HoldoutValidationError, match="dev overlaps omission"):
        crosscheck_against_units(mutated, omissions)


def test_cross_split_greek_ngram_leaks_detects_shared_5gram() -> None:
    split = load_nt_split(SPLIT_PATH)
    greek_texts = {uid: "" for p in split.passages for uid in p.unit_ids}
    greek_texts["john.1.1"] = "εν αρχη ην ο λογος και ο θεος"
    greek_texts["matthew.13.31"] = "ειπεν δε εν αρχη ην ο λογος υμιν"
    leaks = cross_split_greek_ngram_leaks(split, greek_texts)
    assert "εν αρχη ην ο λογος" in leaks
    with pytest.raises(HoldoutValidationError, match="5-token"):
        assert_no_train_holdout_greek_ngram_leakage(split, greek_texts)

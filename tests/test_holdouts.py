# SPDX-License-Identifier: Apache-2.0
"""Tests for JAK-33 benchmark holdouts, omissions, and divergences."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from zetodos_thomas.bundled import (
    bundled_fragment_divergences_path,
    bundled_nt_split_path,
    bundled_omission_fixtures_path,
    bundled_pilot_path,
)
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.holdouts import (
    HoldoutValidationError,
    crosscheck_against_pilot,
    load_fragment_divergences,
    load_nt_split,
    load_omission_fixtures,
    validate_holdout_artifacts,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.schemas.holdout import (
    DivergenceKind,
    NtCorpusSplit,
    OmissionFixtureSet,
    PassageRole,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data/training/splits/horner-vaticanus-nt-split-v3.json"
OMISSION_PATH = REPO_ROOT / "data/evidence/omissions/john-8-pa-omission-fixture-v3.json"
OMISSION_PATH_V2 = (
    REPO_ROOT / "data/evidence/omissions/john-8-pa-omission-fixture-v1.json"
)
SPLIT_PATH_V2 = REPO_ROOT / "data/training/splits/horner-vaticanus-nt-split-v2.json"
PILOT_PATH_V2 = REPO_ROOT / "data/sources/pilot/horner-vaticanus-nt-pilot-v2.json"
DIVERGENCE_PATH = (
    REPO_ROOT / "data/evidence/divergences/greek-thomas-fragment-divergences-v1.json"
)
PILOT_PATH = REPO_ROOT / "data/sources/pilot/horner-vaticanus-nt-pilot.json"


def test_committed_holdout_artifacts_validate() -> None:
    pilot = load_pilot_corpus(PILOT_PATH)
    split, omissions, divergences = validate_holdout_artifacts(
        split_path=SPLIT_PATH,
        omission_path=OMISSION_PATH,
        divergence_path=DIVERGENCE_PATH,
        pilot=pilot,
    )
    assert split.id == "horner-vaticanus-nt-split-v3"
    assert split.owned_by == "JAK-43"
    assert "JAK-36" in split.consumed_by
    assert split.greek_source_id == "ntvmr-vaticanus-b"
    assert split.pilot_corpus_id == "horner-vaticanus-nt-pilot-v3"
    assert pilot.id == "horner-vaticanus-nt-pilot-v3"
    assert split.secondary_ot_used is False

    train = [p for p in split.passages if p.role is PassageRole.TRAIN]
    holdout = [p for p in split.passages if p.role is PassageRole.HOLDOUT]
    dev = [p for p in split.passages if p.role is PassageRole.DEV]
    assert {p.book for p in train} >= {"john", "mark", "matthew", "luke"}
    assert any(p.book == "mark" for p in train)
    hold_ids = [uid for p in holdout for uid in p.unit_ids]
    assert "matthew.13.31" in hold_ids and "matthew.10.26" in hold_ids
    assert "luke.8.4" in hold_ids
    assert len(hold_ids) == 22
    dev_ids = [uid for p in dev for uid in p.unit_ids]
    assert "luke.6.20" in dev_ids and "john.3.1" in dev_ids
    assert len(dev_ids) == 51

    assert len(omissions.units) == 11
    assert omissions.pilot_corpus_id == "horner-vaticanus-nt-pilot-v3"
    assert {u.unit_id for u in omissions.units} == {f"john.8.{n}" for n in range(1, 12)}
    assert all(u.greek_empty and u.sahidic_empty for u in omissions.units)

    assert any(d.id == "poxy1-coptic-30-77-division" for d in divergences.divergences)
    assert any(d.kind is DivergenceKind.DIVISION for d in divergences.divergences)
    assert "P.Oxy. 5575" in divergences.excluded_witnesses
    witness_ids = {e.witness_id for e in split.evaluation_witness_exclusions}
    assert witness_ids >= {"poxy-1-dclp", "poxy-654-dclp", "poxy-655-dclp"}


def test_bundled_holdouts_match_checkout() -> None:
    from zetodos_thomas.bundled import bundled_omission_fixtures_for_pilot

    for checkout, bundled in (
        (SPLIT_PATH, bundled_nt_split_path()),
        (OMISSION_PATH, bundled_omission_fixtures_path()),
        (DIVERGENCE_PATH, bundled_fragment_divergences_path()),
        (
            OMISSION_PATH_V2,
            bundled_omission_fixtures_for_pilot("horner-vaticanus-nt-pilot-v2"),
        ),
    ):
        assert checkout.read_bytes() == bundled.read_bytes()
    assert PILOT_PATH.read_bytes() == bundled_pilot_path().read_bytes()


def test_archival_v2_split_loads_with_matching_omission() -> None:
    """load_splits(v2) must pair the v2-compatible omission catalog."""
    from zetodos_thomas.align.splits import load_splits

    splits = load_splits(SPLIT_PATH_V2)
    assert splits.pilot_corpus_id == "horner-vaticanus-nt-pilot-v2"
    assert splits.id == "horner-vaticanus-nt-split-v2"
    pilot = load_pilot_corpus(PILOT_PATH_V2)
    validate_holdout_artifacts(
        split_path=SPLIT_PATH_V2,
        omission_path=OMISSION_PATH_V2,
        divergence_path=DIVERGENCE_PATH,
        pilot=pilot,
    )


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
            pilot=load_pilot_corpus(PILOT_PATH),
        )


def test_cli_sources_validate_includes_holdouts() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate", "--repo-root", str(REPO_ROOT)])
    assert result.exit_code == 0, result.output
    assert "holdouts horner-vaticanus-nt-split-v3" in result.output
    assert "omissions=11" in result.output


def test_loaders_round_trip_types() -> None:
    assert isinstance(load_nt_split(SPLIT_PATH), NtCorpusSplit)
    assert isinstance(load_omission_fixtures(OMISSION_PATH), OmissionFixtureSet)
    assert load_fragment_divergences(DIVERGENCE_PATH).schema_version == "1"


def test_crosscheck_rejects_unit_outside_declared_passage_range(
    tmp_path: Path,
) -> None:
    """Moving Mark train ids into John train while ranges stay John fails."""
    split = load_nt_split(SPLIT_PATH)
    payload = split.model_dump(mode="json")
    for passage in payload["passages"]:
        if passage["role"] == "train" and passage["book"] == "john":
            passage["unit_ids"] = [*passage["unit_ids"], "mark.1.1"]
        if passage["role"] == "train" and passage["book"] == "mark":
            passage["unit_ids"] = [
                uid for uid in passage["unit_ids"] if uid != "mark.1.1"
            ]
    bad = tmp_path / "split.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HoldoutValidationError, match="must equal pilot units"):
        validate_holdout_artifacts(
            split_path=bad,
            omission_path=OMISSION_PATH,
            divergence_path=DIVERGENCE_PATH,
            pilot=load_pilot_corpus(PILOT_PATH),
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
    mutated = NtCorpusSplit.model_validate(payload)
    with pytest.raises(HoldoutValidationError, match="dev overlaps omission"):
        crosscheck_against_pilot(mutated, omissions, load_pilot_corpus(PILOT_PATH))


def test_integrated_sources_validate_and_align_benchmark() -> None:
    """Single canonical artifact: sources validate + align benchmark."""
    runner = CliRunner()
    sources = runner.invoke(app, ["sources", "validate", "--repo-root", str(REPO_ROOT)])
    assert sources.exit_code == 0, sources.output
    assert "horner-vaticanus-nt-split-v3" in sources.output

    out = REPO_ROOT / "data" / "sources" / "alignment" / "benchmark-report.json"
    bench = runner.invoke(
        app,
        [
            "align",
            "benchmark",
            "--repo-root",
            str(REPO_ROOT),
            "--splits",
            str(SPLIT_PATH),
            "-o",
            str(out),
        ],
    )
    assert bench.exit_code == 0, bench.output
    assert out.is_file()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pass_threshold"] is None
    assert report["pilot_id"] == "horner-vaticanus-nt-pilot-v3"
    assert report["human_reference"]["status"] == "provisional"
    assert "null_rate" not in report["lexical_baseline"]["null_alignments"]

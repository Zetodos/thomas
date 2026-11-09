# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase 1 Thomas 7/17/30/77 evidence pilot (JAK-34)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from zetodos_thomas.bundled import (
    bundled_evidence_pilot_path,
    bundled_manifests_dir,
)
from zetodos_thomas.cli.app import app
from zetodos_thomas.ingest.evidence_pilot import (
    EvidencePilotValidationError,
    load_evidence_pilot,
    validate_evidence_pilot_or_raise,
)
from zetodos_thomas.ingest.manifests import validate_manifests_dir
from zetodos_thomas.ingest.paths import resolve_evidence_pilot_path
from zetodos_thomas.schemas.evidence import PILOT_LOGIA, PilotEvidenceSet
from zetodos_thomas.schemas.fragment import EvidenceType

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
THOMAS_PATH = REPO_ROOT / "data" / "sources" / "thomas" / "dilley-thomas-v6.2.0.json"
FRAGMENTS_PATH = (
    REPO_ROOT / "data" / "sources" / "fragments" / "greek-thomas-fragments.json"
)
EVIDENCE_PATH = REPO_ROOT / "data" / "evidence" / "pilot" / "thomas-7-17-30-77.json"
EVIDENCE_JSONL = REPO_ROOT / "data" / "evidence" / "pilot" / "thomas-7-17-30-77.jsonl"


def test_pilot_package_schema_and_cross_refs() -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    assert package.id == "thomas-pilot-evidence-7-17-30-77-v1"
    assert package.greek_nt_spine_source_id == "ntvmr-vaticanus-b"
    assert tuple(s.logion for s in package.sayings) == PILOT_LOGIA
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    validate_evidence_pilot_or_raise(
        package,
        manifests=manifests,
        thomas_corpus_path=THOMAS_PATH,
        fragments_corpus_path=FRAGMENTS_PATH,
    )


def test_thomas_17_n_parallel_policy() -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    saying = next(s for s in package.sayings if s.logion == 17)
    n = next(e for e in saying.evidence if e.evidence_type is EvidenceType.N)
    assert n.status.value == "attested"
    assert "1 Cor" in n.citation or "1 Corinthians" in n.citation
    blob = f"{n.restrictions} {n.notes}".lower()
    assert "not proof" in blob or "not a thomas quotation" in blob
    assert n.agreement is not None
    assert n.agreement.value == "verbal"


def test_thomas_30_77_divergent_witness_not_merged() -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    div = [
        w for w in package.witness_alignments if w.kind.value == "divergent-sequence"
    ]
    assert div
    assert {"thomas.30", "thomas.77"} <= set(div[0].coptic_unit_ids)
    assert div[0].greek_transcription_id == "poxy-1-dclp"
    notes_l = div[0].notes.lower()
    assert "silent merge" in notes_l or "silently merge" in notes_l


def test_attested_and_missing_lists_present() -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    for saying in package.sayings:
        assert saying.attested
        assert saying.missing
        types = {e.evidence_type for e in saying.evidence}
        assert types >= {
            EvidenceType.G,
            EvidenceType.L,
            EvidenceType.N,
            EvidenceType.P,
        }


def test_jsonl_matches_json_sayings() -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    lines = [
        json.loads(line)
        for line in EVIDENCE_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == len(package.sayings)
    assert [row["unit_id"] for row in lines] == [s.unit_id for s in package.sayings]


def test_bundled_evidence_pilot_matches_checkout() -> None:
    assert EVIDENCE_PATH.read_text(encoding="utf-8") == (
        bundled_evidence_pilot_path().read_text(encoding="utf-8")
    )
    assert resolve_evidence_pilot_path(REPO_ROOT) == EVIDENCE_PATH
    # Packaged manifests still include Dilley + DCLP for wheel offline path.
    bundled_ids = {p.stem for p in bundled_manifests_dir().glob("*.json")}
    assert "dilley-thomas" in bundled_ids
    assert "dclp-poxy-1" in bundled_ids


def test_cli_sources_validate_includes_evidence_pilot() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "validate"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "evidence pilot" in result.stdout.lower()
    assert "thomas-pilot-evidence-7-17-30-77-v1" in result.stdout


def test_rejects_wrong_greek_nt_spine() -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    data = json.loads(package.model_dump_json())
    data["greek_nt_spine_source_id"] = "westcott-hort"
    try:
        PilotEvidenceSet.model_validate(data)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_cross_ref_rejects_unknown_transcription(tmp_path: Path) -> None:
    package = load_evidence_pilot(EVIDENCE_PATH)
    data = json.loads(package.model_dump_json())
    data["sayings"][0]["evidence"][0]["transcription_id"] = "missing-tx"
    bad = PilotEvidenceSet.model_validate(data)
    manifests = validate_manifests_dir(MANIFESTS_DIR)
    try:
        validate_evidence_pilot_or_raise(
            bad,
            manifests=manifests,
            thomas_corpus_path=THOMAS_PATH,
            fragments_corpus_path=FRAGMENTS_PATH,
        )
        ok = True
    except EvidencePilotValidationError:
        ok = False
    assert ok is False

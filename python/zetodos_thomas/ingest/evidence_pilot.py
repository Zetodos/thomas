# SPDX-License-Identifier: Apache-2.0
"""Load and cross-check the Phase 1 Thomas evidence pilot package (JAK-34)."""

from __future__ import annotations

import json
from pathlib import Path

from zetodos_thomas.ingest.dilley_thomas import CORPUS_ID as THOMAS_CORPUS_ID
from zetodos_thomas.ingest.dilley_thomas import SOURCE_ID as DILLEY_SOURCE_ID
from zetodos_thomas.ingest.dilley_thomas import load_thomas_corpus
from zetodos_thomas.ingest.fragments import (
    DEFAULT_FRAGMENT_CORPUS_FILENAME,
    load_fragment_corpus,
)
from zetodos_thomas.schemas.evidence import PILOT_LOGIA, PilotEvidenceSet
from zetodos_thomas.schemas.fragment import EvidenceType
from zetodos_thomas.schemas.source import SourceManifest

DEFAULT_EVIDENCE_PILOT_FILENAME = "thomas-7-17-30-77.json"


class EvidencePilotValidationError(ValueError):
    """Raised when the pilot evidence package fails offline checks."""


def evidence_pilot_path(repo_root: Path) -> Path:
    """Checkout path for the committed Phase 1 evidence pilot JSON."""
    return repo_root / "data" / "evidence" / "pilot" / DEFAULT_EVIDENCE_PILOT_FILENAME


def load_evidence_pilot(path: Path) -> PilotEvidenceSet:
    """Load and schema-validate a pilot evidence JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return PilotEvidenceSet.model_validate(data)


def write_evidence_pilot(package: PilotEvidenceSet, path: Path) -> None:
    """Write a pilot evidence package as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        package.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def validate_evidence_pilot_or_raise(
    package: PilotEvidenceSet,
    *,
    manifests: list[SourceManifest],
    thomas_corpus_path: Path,
    fragments_corpus_path: Path,
) -> PilotEvidenceSet:
    """Cross-check Dilley units, fragment transcriptions, and manifests."""
    manifest_ids = {m.id for m in manifests}
    if package.coptic_source_id not in manifest_ids:
        msg = f"coptic_source_id {package.coptic_source_id!r} missing from manifests"
        raise EvidencePilotValidationError(msg)
    if package.greek_nt_spine_source_id not in manifest_ids:
        msg = (
            "greek_nt_spine_source_id "
            f"{package.greek_nt_spine_source_id!r} missing from manifests"
        )
        raise EvidencePilotValidationError(msg)

    thomas = load_thomas_corpus(thomas_corpus_path)
    if thomas.source_id != DILLEY_SOURCE_ID:
        msg = f"Dilley corpus source_id must be {DILLEY_SOURCE_ID}"
        raise EvidencePilotValidationError(msg)
    if package.coptic_corpus_id not in {thomas.id, THOMAS_CORPUS_ID}:
        msg = (
            f"coptic_corpus_id {package.coptic_corpus_id!r} does not match "
            f"Dilley corpus id {thomas.id!r}"
        )
        raise EvidencePilotValidationError(msg)
    thomas_units = {u.unit_id for u in thomas.units}

    fragments = load_fragment_corpus(fragments_corpus_path)
    if fragments.id != package.fragment_corpus_id:
        msg = f"fragment_corpus_id {package.fragment_corpus_id!r} != {fragments.id!r}"
        raise EvidencePilotValidationError(msg)
    transcriptions = {t.id: t for t in fragments.transcriptions}

    for saying in package.sayings:
        if saying.unit_id not in thomas_units:
            msg = f"{saying.unit_id}: Dilley unit missing from {thomas_corpus_path}"
            raise EvidencePilotValidationError(msg)
        if saying.coptic_source_id != package.coptic_source_id:
            msg = (
                f"{saying.unit_id}: coptic_source_id drift "
                f"({saying.coptic_source_id} vs {package.coptic_source_id})"
            )
            raise EvidencePilotValidationError(msg)

        for record in saying.evidence:
            if record.source_id is not None and record.source_id not in manifest_ids:
                msg = (
                    f"{record.id}: source_id {record.source_id!r} "
                    "not present in manifests"
                )
                raise EvidencePilotValidationError(msg)
            if record.transcription_id is not None:
                if record.transcription_id not in transcriptions:
                    msg = (
                        f"{record.id}: transcription_id "
                        f"{record.transcription_id!r} missing from fragment corpus"
                    )
                    raise EvidencePilotValidationError(msg)
                tx = transcriptions[record.transcription_id]
                if record.source_id is not None and tx.source_id != record.source_id:
                    msg = (
                        f"{record.id}: transcription source_id {tx.source_id!r} "
                        f"!= record source_id {record.source_id!r}"
                    )
                    raise EvidencePilotValidationError(msg)

        # Thomas 17 N policy: verbal parallel must not claim Paul quoted Thomas.
        if saying.logion == 17:
            n_records = [
                e for e in saying.evidence if e.evidence_type is EvidenceType.N
            ]
            if not n_records or all(e.status.value == "missing" for e in n_records):
                msg = "thomas.17 must record the 1 Cor 2:9 N parallel"
                raise EvidencePilotValidationError(msg)
            joined = " ".join(
                f"{e.citation} {e.restrictions} {e.notes}" for e in n_records
            ).lower()
            if "1 cor" not in joined and "1cor" not in joined.replace(" ", ""):
                msg = "thomas.17 N slot must cite 1 Cor 2:9"
                raise EvidencePilotValidationError(msg)
            if "not proof" not in joined and "not a thomas quotation" not in joined:
                msg = (
                    "thomas.17 N slot must state that the parallel is not proof "
                    "Paul quoted Thomas"
                )
                raise EvidencePilotValidationError(msg)

    # 30/77 divergence must be an explicit witness alignment (not silent merge).
    div = [
        w
        for w in package.witness_alignments
        if w.kind.value == "divergent-sequence"
        and set(w.coptic_unit_ids) >= {"thomas.30", "thomas.77"}
    ]
    if not div:
        msg = (
            "pilot must include a divergent-sequence witness alignment "
            "for thomas.30 + thomas.77 vs P.Oxy. 1"
        )
        raise EvidencePilotValidationError(msg)
    for w in div:
        if w.greek_transcription_id not in transcriptions:
            msg = (
                f"{w.id}: greek_transcription_id "
                f"{w.greek_transcription_id!r} missing from fragment corpus"
            )
            raise EvidencePilotValidationError(msg)
        notes_l = w.notes.lower()
        if "silent merge" not in notes_l and "silently merge" not in notes_l:
            msg = f"{w.id}: notes must forbid silent merge of P.Oxy. 1 vs Coptic 30/77"
            raise EvidencePilotValidationError(msg)

    if tuple(s.logion for s in package.sayings) != PILOT_LOGIA:
        msg = f"sayings must remain {PILOT_LOGIA}"
        raise EvidencePilotValidationError(msg)

    # Fragments filename sanity for offline layout docs.
    if fragments_corpus_path.name != DEFAULT_FRAGMENT_CORPUS_FILENAME:
        # Allow alternate paths in tests; still validate content above.
        pass

    return package

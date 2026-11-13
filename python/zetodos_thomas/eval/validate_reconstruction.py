# SPDX-License-Identifier: Apache-2.0
"""Structured validation of blinded reconstruction packages (JAK-40).

Rejects self-asserted-flag-only artifacts: checks DecodeResult/top-1 coherence,
source token provenance, canonical split membership, and fragment exclusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zetodos_thomas.align.splits import (
    SplitRole,
    load_canonical_splits,
)
from zetodos_thomas.eval.spans import DILLEY_SEGMENTATION, HORNER_SEGMENTATION
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_pilot_path,
    resolve_thomas_path,
)
from zetodos_thomas.schemas.blinded import (
    BlindedReconstructionPackage,
    BlindedTargetKind,
    BlindedUnitReconstruction,
)
from zetodos_thomas.schemas.decode import SpanDecodeResult, SpanDecodeStatus

FORBIDDEN_SOURCE_IDS = frozenset(
    {
        "decode-input",
        "dclp-poxy-654",
        "poxy-654",
        "poxy",
    }
)
FORBIDDEN_SOURCE_MARKERS = ("poxy", "dclp-poxy")


class ReconstructionValidationError(ValueError):
    """Blinded reconstruction failed structured integrity checks."""


def _top1_from_span(span: SpanDecodeResult) -> str | None:
    if span.status is SpanDecodeStatus.CANDIDATES and span.candidates:
        return " ".join(span.candidates[0].greek_surfaces)
    return None


def validate_unit_decode_coherence(unit: BlindedUnitReconstruction) -> list[str]:
    """Return problems for one unit's decode/top-1/source provenance."""
    problems: list[str] = []
    result = unit.decode_result
    if len(unit.top1_greek_surfaces) != len(result.spans):
        problems.append(
            f"{unit.unit_id}: top1 length {len(unit.top1_greek_surfaces)} "
            f"!= spans {len(result.spans)}"
        )
    if unit.covering_span_kinds and len(unit.covering_span_kinds) != len(result.spans):
        problems.append(f"{unit.unit_id}: covering_span_kinds length mismatch vs spans")
    for i, span in enumerate(result.spans):
        expected = _top1_from_span(span)
        actual = (
            unit.top1_greek_surfaces[i] if i < len(unit.top1_greek_surfaces) else None
        )
        if expected != actual:
            problems.append(
                f"{unit.unit_id}: top1[{i}] {actual!r} != decode candidates "
                f"{expected!r}"
            )
        if not span.surfaces:
            problems.append(f"{unit.unit_id}: span {span.span_id} has empty surfaces")
        if span.input_span is None:
            problems.append(
                f"{unit.unit_id}: span {span.span_id} missing input_span provenance"
            )
        else:
            if not span.input_span.token_ids:
                problems.append(
                    f"{unit.unit_id}: span {span.span_id} input_span.token_ids empty"
                )
            if span.input_span.source_id in FORBIDDEN_SOURCE_IDS:
                problems.append(
                    f"{unit.unit_id}: forbidden input_span.source_id "
                    f"{span.input_span.source_id!r}"
                )
            if span.input_span.source_id != unit.source_id:
                problems.append(
                    f"{unit.unit_id}: input_span.source_id "
                    f"{span.input_span.source_id!r} != unit.source_id "
                    f"{unit.source_id!r}"
                )
            if len(span.input_span.token_ids) != len(span.surfaces):
                problems.append(
                    f"{unit.unit_id}: span {span.span_id} token_ids/surfaces "
                    "length mismatch"
                )
        if ":" in span.span_id:
            src = span.span_id.split(":", 1)[-1].split("/", 1)[0]
        else:
            src = ""
        forbidden_src = src in FORBIDDEN_SOURCE_IDS or any(
            m in src.lower() for m in FORBIDDEN_SOURCE_MARKERS
        )
        if forbidden_src:
            problems.append(
                f"{unit.unit_id}: span_id embeds forbidden source {span.span_id!r}"
            )
    if unit.source_id in FORBIDDEN_SOURCE_IDS:
        problems.append(f"{unit.unit_id}: forbidden unit.source_id {unit.source_id!r}")
    if unit.target_kind is BlindedTargetKind.THOMAS_PILOT:
        if unit.source_id != "dilley-thomas":
            problems.append(
                f"{unit.unit_id}: thomas_pilot requires dilley-thomas source"
            )
        if unit.segmentation and unit.segmentation != DILLEY_SEGMENTATION:
            problems.append(
                f"{unit.unit_id}: unexpected Dilley segmentation {unit.segmentation!r}"
            )
    if unit.target_kind is BlindedTargetKind.NT_HOLDOUT:
        if unit.source_id != "copsahhorner":
            problems.append(f"{unit.unit_id}: nt_holdout requires copsahhorner source")
        if unit.segmentation and unit.segmentation != HORNER_SEGMENTATION:
            problems.append(
                f"{unit.unit_id}: unexpected Horner segmentation {unit.segmentation!r}"
            )
    return problems


def validate_split_membership(
    package: BlindedReconstructionPackage,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """NT holdout units must be canonical holdout; train/dev must not appear."""
    problems: list[str] = []
    root = require_repo_root(repo_root)
    splits = load_canonical_splits(
        resolve_nt_split_path(root),
        resolve_omission_fixtures_path(root),
    )
    for unit in package.units:
        if unit.target_kind is BlindedTargetKind.THOMAS_PILOT:
            if unit.unit_id != "thomas.7" and not unit.unit_id.startswith("thomas."):
                problems.append(
                    f"{unit.unit_id}: thomas_pilot unit_id not a Thomas saying"
                )
            continue
        if unit.target_kind is BlindedTargetKind.NT_HOLDOUT:
            role = splits.role_for(unit.unit_id)
            if role is not SplitRole.HOLDOUT:
                problems.append(
                    f"{unit.unit_id}: marked nt_holdout but split role is {role!r}"
                )
        else:
            problems.append(f"{unit.unit_id}: unknown target_kind {unit.target_kind!r}")
    return problems


def validate_package_blinding_flags(package: BlindedReconstructionPackage) -> list[str]:
    problems: list[str] = []
    if package.owned_by != "JAK-40":
        problems.append(f"owned_by must be JAK-40 (got {package.owned_by!r})")
    if package.blinding.get("greek_consulted_during_reconstruction") is not False:
        problems.append("greek_consulted_during_reconstruction must be false")
    if package.blinding.get("fragment_corpus_opened") is not False:
        problems.append("fragment_corpus_opened must be false")
    if package.adopted_aligner is not None:
        problems.append("adopted_aligner must be null")
    if set(package.phase2_active_aligners) != {"eflomal", "fast_align"}:
        problems.append("phase2_active_aligners must be eflomal+fast_align")
    if package.provenance.get("llm") is not False:
        problems.append("provenance.llm must be false")
    if package.provenance.get("network") is not False:
        problems.append("provenance.network must be false")
    return problems


def validate_artifact_digests(
    package: BlindedReconstructionPackage,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    """Require recorded source digests to match on-disk corpora when present."""
    problems: list[str] = []
    root = require_repo_root(repo_root)
    import hashlib

    checks = {
        "dilley_thomas_sha256": resolve_thomas_path(root),
        "pilot_corpus_sha256": resolve_pilot_path(root),
        "nt_split_sha256": resolve_nt_split_path(root),
    }
    for key, path in checks.items():
        recorded = package.artifact_digests.get(key)
        if not recorded:
            problems.append(f"missing artifact_digests[{key}]")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != recorded:
            problems.append(
                f"artifact_digests[{key}] mismatch "
                f"(got {actual[:12]}… expected {recorded[:12]}…)"
            )
    return problems


def validate_reconstruction_package(
    package: BlindedReconstructionPackage,
    *,
    repo_root: Path | None = None,
    check_digests: bool = True,
    check_splits: bool = True,
) -> list[str]:
    """Return all structured validation problems (empty => OK)."""
    problems = validate_package_blinding_flags(package)
    for unit in package.units:
        problems.extend(validate_unit_decode_coherence(unit))
    if check_splits:
        problems.extend(validate_split_membership(package, repo_root=repo_root))
    if check_digests:
        problems.extend(validate_artifact_digests(package, repo_root=repo_root))
    # Fragment Greek must not appear as a reconstruction unit source.
    for unit in package.units:
        for span in unit.decode_result.spans:
            for cand in span.candidates:
                # Evidence refs must not point at fragment corpus ids.
                for ev in cand.evidence:
                    if any(m in ev.ref_id.lower() for m in FORBIDDEN_SOURCE_MARKERS):
                        problems.append(
                            f"{unit.unit_id}: candidate evidence ref embeds "
                            f"fragment handle {ev.ref_id!r}"
                        )
    return problems


def validate_reconstruction_path(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load + validate a package JSON; return a report dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    package = BlindedReconstructionPackage.model_validate(payload)
    problems = validate_reconstruction_package(package, repo_root=repo_root)
    return {
        "ok": not problems,
        "problems": problems,
        "id": package.id,
        "path": str(path),
        "unit_count": len(package.units),
    }


def assert_valid_reconstruction(
    package: BlindedReconstructionPackage,
    *,
    repo_root: Path | None = None,
) -> None:
    problems = validate_reconstruction_package(package, repo_root=repo_root)
    if problems:
        raise ReconstructionValidationError(
            "reconstruction validation failed:\n- " + "\n- ".join(problems)
        )

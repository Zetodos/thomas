# SPDX-License-Identifier: Apache-2.0
"""Load and validate source manifests from ``data/manifests/``."""

from __future__ import annotations

import json
from pathlib import Path

from zetodos_thomas.schemas.source import (
    GATED_DEFAULT_SOURCE_IDS,
    AvailabilityMode,
    SourceManifest,
)


class ManifestValidationError(ValueError):
    """One or more source manifests failed validation policy checks."""


def load_manifest(path: Path) -> SourceManifest:
    """Parse a JSON source manifest into a :class:`SourceManifest`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return SourceManifest.model_validate(data)


def iter_manifest_paths(manifests_dir: Path) -> list[Path]:
    """Return sorted ``*.json`` manifest paths (skip README / keep files)."""
    if not manifests_dir.is_dir():
        return []
    return sorted(
        path
        for path in manifests_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )


def load_all_manifests(manifests_dir: Path) -> list[SourceManifest]:
    """Load every JSON manifest under ``manifests_dir``."""
    return [load_manifest(path) for path in iter_manifest_paths(manifests_dir)]


def validate_default_manifests(manifests: list[SourceManifest]) -> list[str]:
    """Return human-readable problems for the default offline allowlist.

    Rejects empty checksums, duplicate ids, and gated Wells corpora appearing
    in the default manifest set (Sahidica / NTAEG / Bohairica).
    """
    problems: list[str] = []
    seen: set[str] = set()
    for manifest in manifests:
        if manifest.id in seen:
            problems.append(f"duplicate source id: {manifest.id}")
        seen.add(manifest.id)

        if manifest.id in GATED_DEFAULT_SOURCE_IDS:
            problems.append(
                f"gated Wells/related source id in default manifests: {manifest.id}"
            )
        if (
            manifest.gate is not None
            and manifest.mode is not AvailabilityMode.MANUAL_IMPORT
        ):
            problems.append(
                f"{manifest.id}: gated sources must use mode=manual-import "
                f"(found {manifest.mode})"
            )
        if not manifest.checksum:
            problems.append(f"{manifest.id}: checksum is empty (pin at ingest)")
    return problems


def validate_manifests_dir(manifests_dir: Path) -> list[SourceManifest]:
    """Load manifests and raise :class:`ManifestValidationError` on policy failures."""
    manifests = load_all_manifests(manifests_dir)
    if not manifests:
        msg = f"no JSON manifests found under {manifests_dir}"
        raise ManifestValidationError(msg)
    problems = validate_default_manifests(manifests)
    if problems:
        raise ManifestValidationError("; ".join(problems))
    return manifests

# SPDX-License-Identifier: Apache-2.0
"""Packaged redistributable manifests and corpora for wheel installs."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def bundled_manifests_dir() -> Path:
    """Return the on-disk path to packaged source manifests."""
    return Path(str(files("zetodos_thomas.bundled") / "manifests"))


def bundled_pilot_path() -> Path:
    """Return the on-disk path to the packaged Horner x Vaticanus B* pilot JSON."""
    return Path(
        str(
            files("zetodos_thomas.bundled") / "pilot" / "horner-vaticanus-nt-pilot.json"
        )
    )


def bundled_thomas_path() -> Path:
    """Return the on-disk path to the packaged Dilley Thomas corpus JSON."""
    return Path(
        str(files("zetodos_thomas.bundled") / "thomas" / "dilley-thomas-v6.2.0.json")
    )


def bundled_fragments_corpus_path() -> Path:
    """Return the on-disk path to the packaged Greek Thomas fragment corpus."""
    return Path(
        str(
            files("zetodos_thomas.bundled")
            / "fragments"
            / "greek-thomas-fragments.json"
        )
    )


def bundled_fragments_tei_dir() -> Path:
    """Return the on-disk path to packaged DCLP TEI fragment sources."""
    return Path(str(files("zetodos_thomas.bundled") / "fragments" / "dclp"))


def bundled_evidence_pilot_path() -> Path:
    """Return the on-disk path to the packaged Phase 1 evidence pilot JSON."""
    return Path(
        str(
            files("zetodos_thomas.bundled")
            / "evidence"
            / "pilot"
            / "thomas-7-17-30-77.json"
        )
    )

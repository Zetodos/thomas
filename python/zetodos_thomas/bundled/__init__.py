# SPDX-License-Identifier: Apache-2.0
"""Packaged redistributable manifests and pilot excerpt for wheel installs."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def bundled_manifests_dir() -> Path:
    """Return the on-disk path to packaged source manifests."""
    return Path(str(files("zetodos_thomas.bundled") / "manifests"))


def bundled_pilot_path() -> Path:
    """Return the on-disk path to the packaged Horner x WH pilot JSON."""
    return Path(
        str(files("zetodos_thomas.bundled") / "pilot" / "horner-wh-nt-pilot.json")
    )

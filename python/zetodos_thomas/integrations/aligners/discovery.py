# SPDX-License-Identifier: Apache-2.0
"""Locate optional eflomal / fast_align binaries after explicit setup."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from zetodos_thomas.integrations.aligners.types import AlignerName


@dataclass(frozen=True)
class AlignerAvailability:
    """Whether an aligner can run offline in this environment."""

    name: AlignerName
    available: bool
    detail: str
    binary_path: str | None = None
    python_module: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "available": self.available,
            "detail": self.detail,
            "binary_path": self.binary_path,
            "python_module": self.python_module,
        }


def _candidate_dirs(repo_root: Path | None) -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("THOMAS_ALIGNER_BIN")
    if env:
        dirs.append(Path(env).expanduser())
    if repo_root is not None:
        dirs.append(repo_root / "data" / "local" / "aligners")
    home_local = Path.home() / ".local" / "bin"
    dirs.append(home_local)
    return dirs


def _find_binary(name: str, repo_root: Path | None) -> Path | None:
    for directory in _candidate_dirs(repo_root):
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    which = shutil.which(name)
    return Path(which) if which else None


def probe_eflomal(repo_root: Path | None = None) -> AlignerAvailability:
    """Probe eflomal Python module and/or native binary."""
    binary = _find_binary("eflomal", repo_root)
    has_module = importlib.util.find_spec("eflomal") is not None
    if has_module:
        return AlignerAvailability(
            name=AlignerName.EFLOMAL,
            available=True,
            detail=(
                "eflomal Python package importable"
                + (f"; binary at {binary}" if binary else "")
            ),
            binary_path=str(binary) if binary else None,
            python_module=True,
        )
    if binary is not None:
        return AlignerAvailability(
            name=AlignerName.EFLOMAL,
            available=True,
            detail=f"eflomal binary available at {binary} (no Python module)",
            binary_path=str(binary),
            python_module=False,
        )
    return AlignerAvailability(
        name=AlignerName.EFLOMAL,
        available=False,
        detail=(
            "eflomal not found. Run `uv run thomas align setup-aligners` "
            "(needs gcc, make, libomp, python3-dev) or install from "
            "https://github.com/robertostling/eflomal — then re-run offline."
        ),
    )


def probe_fast_align(repo_root: Path | None = None) -> AlignerAvailability:
    """Probe fast_align (+ optional atools) binary."""
    binary = _find_binary("fast_align", repo_root)
    atools = _find_binary("atools", repo_root)
    if binary is None:
        return AlignerAvailability(
            name=AlignerName.FAST_ALIGN,
            available=False,
            detail=(
                "fast_align not found. Run `uv run thomas align setup-aligners` "
                "(needs cmake + g++) or build https://github.com/clab/fast_align "
                "into data/local/aligners/."
            ),
        )
    detail = f"fast_align at {binary}"
    if atools is not None:
        detail += f"; atools at {atools} (optional; pure-Python symmetrize used)"
    return AlignerAvailability(
        name=AlignerName.FAST_ALIGN,
        available=True,
        detail=detail,
        binary_path=str(binary),
        python_module=False,
    )


def discover_aligners(repo_root: Path | None = None) -> dict[str, AlignerAvailability]:
    """Return availability for every supported aligner."""
    return {
        AlignerName.EFLOMAL.value: probe_eflomal(repo_root),
        AlignerName.FAST_ALIGN.value: probe_fast_align(repo_root),
    }

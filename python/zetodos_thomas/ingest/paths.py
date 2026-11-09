# SPDX-License-Identifier: Apache-2.0
"""Repository path helpers for CLI and setup commands."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the thomas checkout by walking parents for ``pyproject.toml``.

    Resolves from ``start`` (default: cwd), not from the installed package
    location, so commands write into the user's checkout ``data/`` tree.
    """
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.is_file():
            continue
        text = pyproject.read_text(encoding="utf-8")
        if 'name = "zetodos-thomas"' in text:
            return candidate
    msg = (
        "could not find zetodos-thomas repo root from "
        f"{cur}; pass --repo-root or run from the checkout"
    )
    raise FileNotFoundError(msg)

# SPDX-License-Identifier: Apache-2.0
"""Repository and bundled resource path helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.bundled import (
    bundled_manifests_dir,
    bundled_pilot_path,
    bundled_thomas_path,
)
from zetodos_thomas.ingest.dilley_thomas import CORPUS_ID as THOMAS_CORPUS_ID

PILOT_FILENAME = "horner-vaticanus-nt-pilot.json"
THOMAS_FILENAME = f"{THOMAS_CORPUS_ID}.json"


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the thomas checkout by walking parents for ``pyproject.toml``.

    Resolves from ``start`` (default: cwd), not from the installed package
    location, so writable downloads/outputs go into the user's checkout.
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


def resolve_manifests_dir(repo_root: Path | None) -> Path:
    """Prefer checkout ``data/manifests``; fall back to packaged resources."""
    if repo_root is not None:
        checkout = repo_root / "data" / "manifests"
        if any(checkout.glob("*.json")):
            return checkout
    try:
        root = find_repo_root()
        checkout = root / "data" / "manifests"
        if any(checkout.glob("*.json")):
            return checkout
    except FileNotFoundError:
        pass
    return bundled_manifests_dir()


def resolve_pilot_path(repo_root: Path | None) -> Path:
    """Prefer checkout pilot JSON; fall back to the packaged pilot."""
    if repo_root is not None:
        checkout = repo_root / "data" / "sources" / "pilot" / PILOT_FILENAME
        if checkout.is_file():
            return checkout
    try:
        root = find_repo_root()
        checkout = root / "data" / "sources" / "pilot" / PILOT_FILENAME
        if checkout.is_file():
            return checkout
    except FileNotFoundError:
        pass
    return bundled_pilot_path()


def resolve_thomas_path(repo_root: Path | None) -> Path:
    """Prefer checkout Dilley Thomas JSON; fall back to the packaged corpus."""
    if repo_root is not None:
        checkout = repo_root / "data" / "sources" / "thomas" / THOMAS_FILENAME
        if checkout.is_file():
            return checkout
    try:
        root = find_repo_root()
        checkout = root / "data" / "sources" / "thomas" / THOMAS_FILENAME
        if checkout.is_file():
            return checkout
    except FileNotFoundError:
        pass
    return bundled_thomas_path()


def require_repo_root(explicit: Path | None = None) -> Path:
    """Return a writable checkout root (required for download / rebuild)."""
    if explicit is not None:
        return explicit.resolve()
    return find_repo_root()

# SPDX-License-Identifier: Apache-2.0
"""Repository and bundled resource path helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path

from zetodos_thomas.bundled import (
    bundled_evidence_pilot_path,
    bundled_fragment_divergences_path,
    bundled_fragments_corpus_path,
    bundled_manifests_dir,
    bundled_nt_split_path,
    bundled_omission_fixtures_path,
    bundled_pilot_path,
    bundled_thomas_path,
)
from zetodos_thomas.ingest.dilley_thomas import CORPUS_ID as THOMAS_CORPUS_ID
from zetodos_thomas.ingest.evidence_pilot import DEFAULT_EVIDENCE_PILOT_FILENAME
from zetodos_thomas.ingest.fragments import DEFAULT_FRAGMENT_CORPUS_FILENAME

PILOT_FILENAME = "horner-vaticanus-nt-pilot.json"
THOMAS_FILENAME = f"{THOMAS_CORPUS_ID}.json"
NT_SPLIT_REL = Path("data/training/splits/horner-vaticanus-nt-split-v3.json")
OMISSION_REL = Path("data/evidence/omissions/john-8-pa-omission-fixture-v1.json")
DIVERGENCE_REL = Path(
    "data/evidence/divergences/greek-thomas-fragment-divergences-v1.json"
)


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


def resolve_fragments_corpus_path(repo_root: Path | None) -> Path:
    """Prefer checkout fragment corpus JSON; fall back to packaged resource."""
    relative = Path("data") / "sources" / "fragments" / DEFAULT_FRAGMENT_CORPUS_FILENAME
    if repo_root is not None:
        checkout = repo_root / relative
        if checkout.is_file():
            return checkout
    try:
        root = find_repo_root()
        checkout = root / relative
        if checkout.is_file():
            return checkout
    except FileNotFoundError:
        pass
    return bundled_fragments_corpus_path()


def resolve_evidence_pilot_path(repo_root: Path | None) -> Path:
    """Prefer checkout evidence pilot JSON; fall back to packaged resource."""
    relative = Path("data") / "evidence" / "pilot" / DEFAULT_EVIDENCE_PILOT_FILENAME
    if repo_root is not None:
        checkout = repo_root / relative
        if checkout.is_file():
            return checkout
    try:
        root = find_repo_root()
        checkout = root / relative
        if checkout.is_file():
            return checkout
    except FileNotFoundError:
        pass
    return bundled_evidence_pilot_path()


def require_repo_root(explicit: Path | None = None) -> Path:
    """Return a writable checkout root (required for download / rebuild)."""
    if explicit is not None:
        return explicit.resolve()
    return find_repo_root()


def _resolve_data_file(repo_root: Path | None, relative: Path, bundled: Path) -> Path:
    """Prefer checkout path when the file exists; else packaged resource."""
    if repo_root is not None:
        checkout = repo_root / relative
        if checkout.is_file():
            return checkout
    try:
        root = find_repo_root()
        checkout = root / relative
        if checkout.is_file():
            return checkout
    except FileNotFoundError:
        pass
    return bundled


def resolve_nt_split_path(repo_root: Path | None) -> Path:
    """Prefer checkout NT split JSON; fall back to packaged artifact."""
    return _resolve_data_file(repo_root, NT_SPLIT_REL, bundled_nt_split_path())


def resolve_omission_fixtures_path(repo_root: Path | None) -> Path:
    """Prefer checkout omission fixtures; fall back to packaged artifact."""
    return _resolve_data_file(repo_root, OMISSION_REL, bundled_omission_fixtures_path())


def resolve_fragment_divergences_path(repo_root: Path | None) -> Path:
    """Prefer checkout divergence catalog; fall back to packaged artifact."""
    return _resolve_data_file(
        repo_root, DIVERGENCE_REL, bundled_fragment_divergences_path()
    )

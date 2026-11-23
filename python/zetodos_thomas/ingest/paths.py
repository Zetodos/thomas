# SPDX-License-Identifier: Apache-2.0
"""Repository and packaged data path helpers for CLI commands."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from zetodos_thomas.ingest.dilley_thomas import CORPUS_ID as THOMAS_CORPUS_ID
from zetodos_thomas.ingest.evidence_pilot import DEFAULT_EVIDENCE_PILOT_FILENAME
from zetodos_thomas.ingest.fragments import DEFAULT_FRAGMENT_CORPUS_FILENAME

THOMAS_FILENAME = f"{THOMAS_CORPUS_ID}.json"
NT_SPLIT_REL = Path("data/training/splits/nt-split-v1.json")
LEGACY_NT_SPLIT_REL = Path("data/training/splits/horner-vaticanus-nt-split-v3.json")
OMISSION_REL = Path("data/evidence/omissions/john-8-pa-omission-fixture-v3.json")
DIVERGENCE_REL = Path(
    "data/evidence/divergences/greek-thomas-fragment-divergences-v1.json"
)


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the thomas checkout by walking parents for ``pyproject.toml``."""
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


def _packaged_data_path(relative: Path) -> Path:
    """Fallback path inside an installed wheel (``zetodos_thomas/data/...``)."""
    traversable = resources.files("zetodos_thomas").joinpath(*relative.parts)
    return Path(str(traversable))


def _resolve_data_file(repo_root: Path | None, relative: Path) -> Path:
    if repo_root is not None:
        checkout = repo_root / relative
        if checkout.is_file() or checkout.is_dir():
            return checkout
    try:
        root = find_repo_root()
        checkout = root / relative
        if checkout.is_file() or checkout.is_dir():
            return checkout
    except FileNotFoundError:
        pass
    return _packaged_data_path(relative)


def require_repo_root(explicit: Path | None = None) -> Path:
    """Return a writable checkout root (required for download / rebuild)."""
    if explicit is not None:
        return explicit.resolve()
    return find_repo_root()


def resolve_manifests_dir(repo_root: Path | None) -> Path:
    """Prefer checkout ``data/manifests``; fall back to packaged resources."""
    return _resolve_data_file(repo_root, Path("data/manifests"))


def resolve_thomas_path(repo_root: Path | None) -> Path:
    """Prefer checkout Dilley Thomas JSON; fall back to packaged data."""
    return _resolve_data_file(repo_root, Path("data/sources/thomas") / THOMAS_FILENAME)


def resolve_fragments_corpus_path(repo_root: Path | None) -> Path:
    """Prefer checkout fragment corpus JSON; fall back to packaged data."""
    return _resolve_data_file(
        repo_root, Path("data/sources/fragments") / DEFAULT_FRAGMENT_CORPUS_FILENAME
    )


def resolve_evidence_pilot_path(repo_root: Path | None) -> Path:
    """Prefer checkout evidence pilot JSON; fall back to packaged data."""
    return _resolve_data_file(
        repo_root, Path("data/evidence/pilot") / DEFAULT_EVIDENCE_PILOT_FILENAME
    )


def resolve_nt_split_path(repo_root: Path | None) -> Path:
    """Prefer checkout NT split JSON (falls back to legacy v3 if v1 not built)."""
    primary = _resolve_data_file(repo_root, NT_SPLIT_REL)
    if primary.is_file():
        return primary
    return _resolve_data_file(repo_root, LEGACY_NT_SPLIT_REL)


def resolve_omission_fixtures_path(repo_root: Path | None) -> Path:
    """Prefer checkout omission fixtures; fall back to packaged data."""
    return _resolve_data_file(repo_root, OMISSION_REL)


def resolve_fragment_divergences_path(repo_root: Path | None) -> Path:
    """Prefer checkout divergence catalog; fall back to packaged data."""
    return _resolve_data_file(repo_root, DIVERGENCE_REL)

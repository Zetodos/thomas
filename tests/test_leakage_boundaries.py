# SPDX-License-Identifier: Apache-2.0
"""Strict AST and runtime zero-leakage boundary checks."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "python" / "zetodos_thomas"

# Training & reconstruction packages that must NEVER touch fragment witnesses
SEALED_MODEL_PACKAGES = ("morphology", "lexicon", "align", "reconstruct", "neural")
FORBIDDEN_FRAG_IMPORTS = (
    "dclp_fragments",
    "fragments",
    "fragment_clause_map",
    "eval",
)
FORBIDDEN_PATH_LITERALS = (
    "greek-thomas-fragments",
    "fragment-clause-map",
    "dclp/",
)
FORBIDDEN_GATED_LITERALS = (
    "sahidica",
    "ntaeg",
    "bohairica",
    "ud_coptic",
    "coptic_scriptorium_mark",
)


def test_model_packages_never_import_fragments_or_eval() -> None:
    """AST scan ensuring morphology/lexicon/align/reconstruct never import fragments."""
    for pkg in SEALED_MODEL_PACKAGES:
        pkg_dir = PKG_ROOT / pkg
        for py_file in pkg_dir.rglob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in FORBIDDEN_FRAG_IMPORTS:
                            assert bad not in alias.name, (
                                f"{py_file} imports forbidden module {alias.name!r}"
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for bad in FORBIDDEN_FRAG_IMPORTS:
                        assert bad not in node.module, (
                            f"{py_file} imports from forbidden {node.module!r}"
                        )
            low = src.lower()
            for lit in FORBIDDEN_PATH_LITERALS:
                assert lit not in low, (
                    f"{py_file} references sealed path literal {lit!r}"
                )
            for gated in FORBIDDEN_GATED_LITERALS:
                assert gated not in low, f"{py_file} references gated corpus {gated!r}"


def test_thomas_not_used_by_morphology_trainer() -> None:
    src = (
        (PKG_ROOT / "morphology" / "coptic_analyzer.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "dilley-thomas" not in src
    assert "thomas_gospel" not in src

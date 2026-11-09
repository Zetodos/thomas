# SPDX-License-Identifier: Apache-2.0
"""Typer application entry for the ``thomas`` console script."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="thomas",
    help="Offline Gospel of Thomas reconstruction tooling.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Zetodos Thomas CLI (Phase 0 scaffold; subcommands arrive in later phases)."""


if __name__ == "__main__":
    app()

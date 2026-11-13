# SPDX-License-Identifier: Apache-2.0
"""Typer application entry for the ``thomas`` console script."""

from __future__ import annotations

import typer

from zetodos_thomas.cli.align import align_app
from zetodos_thomas.cli.decode import decode_app
from zetodos_thomas.cli.lexicon import lexicon_app
from zetodos_thomas.cli.mappings import mappings_app
from zetodos_thomas.cli.sources import sources_app

app = typer.Typer(
    name="thomas",
    help="Offline Gospel of Thomas reconstruction tooling.",
    no_args_is_help=True,
)
app.add_typer(sources_app, name="sources")
app.add_typer(align_app, name="align")
app.add_typer(lexicon_app, name="lexicon")
app.add_typer(mappings_app, name="mappings")
app.add_typer(decode_app, name="decode")


@app.callback()
def main() -> None:
    """Zetodos Thomas CLI."""


if __name__ == "__main__":
    app()

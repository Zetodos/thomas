# SPDX-License-Identifier: Apache-2.0
"""Typer application entry for the ``thomas`` console script."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.cli.eval_cmd import eval_app
from zetodos_thomas.cli.model_cmd import model_app
from zetodos_thomas.cli.reconstruct_cmd import reconstruct_app
from zetodos_thomas.cli.sources import sources_app
from zetodos_thomas.ingest.download import DownloadError, download_all_sources
from zetodos_thomas.ingest.paths import require_repo_root

app = typer.Typer(
    name="thomas",
    help="Offline Gospel of Thomas reconstruction tooling.",
    no_args_is_help=True,
)
app.add_typer(sources_app, name="sources")
app.add_typer(model_app, name="model")
app.add_typer(reconstruct_app, name="reconstruct")
app.add_typer(eval_app, name="eval")


@app.command("setup")
def setup_command(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Writable checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download and re-verify sources even when already present",
    ),
    skip_aligners: bool = typer.Option(
        False,
        "--skip-aligners",
        help="Skip building local eflomal binary",
    ),
) -> None:
    """One-time network setup: download pinned corpora/lexicons and build eflomal."""
    try:
        root = require_repo_root(repo_root)
        paths = download_all_sources(root, force=force)
        for name, path in sorted(paths.items()):
            typer.echo(f"OK: {name} -> {path}")
        if not skip_aligners:
            import subprocess
            import sys

            script = root / "scripts" / "setup_aligners.py"
            subprocess.run(
                [sys.executable, str(script), "--repo-root", str(root)],
                check=True,
            )
    except (DownloadError, FileNotFoundError, OSError, Exception) as exc:
        typer.secho(f"Setup failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.callback()
def main() -> None:
    """Zetodos Thomas CLI."""


if __name__ == "__main__":
    app()

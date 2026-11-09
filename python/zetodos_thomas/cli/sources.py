# SPDX-License-Identifier: Apache-2.0
"""``thomas sources`` Typer subcommands."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.ingest.download import DownloadError, download_pilot_sources
from zetodos_thomas.ingest.manifests import (
    ManifestValidationError,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.paths import find_repo_root
from zetodos_thomas.ingest.pilot import (
    build_pilot_corpus,
    load_pilot_corpus,
    write_pilot_corpus,
)
from zetodos_thomas.schemas.source import GATED_DEFAULT_SOURCE_IDS

sources_app = typer.Typer(
    name="sources",
    help="Validate manifests and manage redistributable source imports.",
    no_args_is_help=True,
)


def _default_repo_root() -> Path:
    try:
        return find_repo_root()
    except FileNotFoundError:
        # Fallback for tests / unusual layouts; prefer find_repo_root in practice.
        return Path.cwd()


def _paths(repo_root: Path) -> dict[str, Path]:
    return {
        "manifests": repo_root / "data" / "manifests",
        "pilot": repo_root / "data" / "sources" / "pilot" / "horner-wh-nt-pilot.json",
        "sword": repo_root / "data" / "downloads" / "sword",
        "wh": repo_root / "data" / "downloads" / "westcott-hort" / "textonly",
    }


@sources_app.command("validate")
def validate_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root containing data/manifests (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    manifests_dir: Path | None = typer.Option(
        None,
        "--manifests-dir",
        help="Directory of JSON source manifests",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    pilot_path: Path | None = typer.Option(
        None,
        "--pilot",
        help="Optional pilot corpus JSON to validate (omit file to skip)",
        exists=False,
        dir_okay=False,
    ),
) -> None:
    """Validate default manifests (and optional pilot) without network access."""
    root = repo_root or _default_repo_root()
    paths = _paths(root)
    manifests_path = manifests_dir or paths["manifests"]
    pilot = pilot_path or paths["pilot"]

    try:
        manifests = validate_manifests_dir(manifests_path)
    except ManifestValidationError as exc:
        typer.secho(f"Manifest validation failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"OK: {len(manifests)} manifest(s) under {manifests_path}")
    for manifest in manifests:
        typer.echo(f"  - {manifest.id} ({manifest.mode}, {manifest.spdx_or_custom})")

    blocked = sorted(GATED_DEFAULT_SOURCE_IDS & {m.id for m in manifests})
    if blocked:  # defensive; validate_manifests_dir already rejects these
        typer.secho(
            f"Gated ids present (forbidden on default path): {blocked}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    typer.echo(
        "OK: default path excludes Sahidica / NTAEG / Bohairica "
        f"({len(GATED_DEFAULT_SOURCE_IDS)} gated ids)"
    )

    if pilot.is_file():
        corpus = load_pilot_corpus(pilot)
        empty_sahidic = sum(
            1 for unit in corpus.units if not unit.sahidic.original_text
        )
        typer.echo(
            f"OK: pilot {corpus.id} - {len(corpus.units)} units "
            f"({empty_sahidic} empty Sahidic slots) from {pilot}"
        )
        if corpus.greek_source_id != "westcott-hort":
            typer.secho(
                "Pilot greek_source_id must be westcott-hort (JAK-15)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        if corpus.sahidic_source_id != "copsahhorner":
            typer.secho(
                "Pilot sahidic_source_id must be copsahhorner (JAK-14)",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    else:
        typer.echo(f"Note: pilot file not found at {pilot} (skipped)")


@sources_app.command("download")
def download_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (default: auto-detect from cwd)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even when checksum-verified files already exist",
    ),
) -> None:
    """Download CopSahHorner + Westcott-Hort textonly into data/downloads/.

    Reads pins from data/manifests/, verifies sha256, and unpacks the SWORD
    module. Requires network. After this, optionally run ``build-pilot``.
    """
    root = repo_root or _default_repo_root()
    try:
        sword_root, wh_dir = download_pilot_sources(root, force=force)
    except (DownloadError, FileNotFoundError, OSError) as exc:
        typer.secho(f"Download failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK: CopSahHorner SWORD tree at {sword_root}")
    typer.echo(f"OK: Westcott-Hort textonly at {wh_dir}")
    typer.echo("Next (optional): uv run thomas sources build-pilot")


@sources_app.command("build-pilot")
def build_pilot(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (default: auto-detect from cwd)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    sword_root: Path | None = typer.Option(
        None,
        "--sword-root",
        help="Local SWORD tree containing CopSahHorner (under data/downloads/)",
        exists=True,
        file_okay=False,
    ),
    wh_textonly_dir: Path | None = typer.Option(
        None,
        "--wh-textonly-dir",
        help="Directory of Robinson WH *.WH textonly files",
        exists=True,
        file_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination JSON for the redistributable pilot excerpt",
    ),
) -> None:
    """Build the committed pilot excerpt from local downloads (explicit setup)."""
    root = repo_root or _default_repo_root()
    paths = _paths(root)
    corpus = build_pilot_corpus(
        sword_root=sword_root or paths["sword"],
        wh_textonly_dir=wh_textonly_dir or paths["wh"],
    )
    dest = output or paths["pilot"]
    write_pilot_corpus(corpus, dest)
    typer.echo(f"Wrote {len(corpus.units)} parallel units to {dest}")

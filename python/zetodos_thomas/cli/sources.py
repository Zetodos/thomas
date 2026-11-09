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
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_manifests_dir,
    resolve_pilot_path,
)
from zetodos_thomas.ingest.pilot import (
    build_pilot_corpus,
    load_pilot_corpus,
    write_pilot_corpus,
)
from zetodos_thomas.ingest.verify_inputs import verify_pilot_rebuild_inputs
from zetodos_thomas.schemas.source import GATED_DEFAULT_SOURCE_IDS

sources_app = typer.Typer(
    name="sources",
    help="Validate manifests and manage redistributable source imports.",
    no_args_is_help=True,
)


@sources_app.command("validate")
def validate_sources(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (optional; packaged manifests used if absent)",
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
    """Validate default manifests (and optional pilot) without network access.

    Manifests/pilot resolve from the checkout when present; otherwise from
    packaged wheel resources under ``zetodos_thomas.bundled``.
    """
    manifests_path = manifests_dir or resolve_manifests_dir(repo_root)
    pilot = pilot_path or resolve_pilot_path(repo_root)

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
        help="Writable checkout root (default: auto-detect from cwd)",
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

    Writes only under the checkout (never into the package install). Requires
    network. After this, optionally run ``build-pilot``.
    """
    try:
        root = require_repo_root(repo_root)
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
        help="Writable checkout root (default: auto-detect from cwd)",
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
    """Build the pilot excerpt from local downloads after checksum verification."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(f"build-pilot requires a checkout: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    manifests_dir = root / "data" / "manifests"
    downloads_dir = root / "data" / "downloads"
    sword = sword_root or downloads_dir / "sword"
    wh_dir = wh_textonly_dir or downloads_dir / "westcott-hort" / "textonly"
    dest = output or root / "data" / "sources" / "pilot" / "horner-wh-nt-pilot.json"

    try:
        verify_pilot_rebuild_inputs(
            manifests_dir=manifests_dir,
            sword_root=sword,
            wh_textonly_dir=wh_dir,
            downloads_dir=downloads_dir,
        )
        corpus = build_pilot_corpus(sword_root=sword, wh_textonly_dir=wh_dir)
    except (DownloadError, FileNotFoundError, OSError, ValueError) as exc:
        typer.secho(f"build-pilot failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    write_pilot_corpus(corpus, dest)
    typer.echo(f"Wrote {len(corpus.units)} parallel units to {dest}")

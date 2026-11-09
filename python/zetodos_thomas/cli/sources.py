# SPDX-License-Identifier: Apache-2.0
"""``thomas sources`` Typer subcommands."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.ingest.manifests import (
    ManifestValidationError,
    validate_manifests_dir,
)
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

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFESTS = REPO_ROOT / "data" / "manifests"
DEFAULT_PILOT = REPO_ROOT / "data" / "sources" / "pilot" / "horner-wh-nt-pilot.json"
DEFAULT_SWORD = REPO_ROOT / "data" / "downloads" / "sword"
DEFAULT_WH = REPO_ROOT / "data" / "downloads" / "westcott-hort" / "textonly"


@sources_app.command("validate")
def validate_sources(
    manifests_dir: Path = typer.Option(
        DEFAULT_MANIFESTS,
        "--manifests-dir",
        help="Directory of JSON source manifests",
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    pilot_path: Path = typer.Option(
        DEFAULT_PILOT,
        "--pilot",
        help="Optional pilot corpus JSON to validate (omit file to skip)",
        exists=False,
        dir_okay=False,
    ),
) -> None:
    """Validate default manifests (and optional pilot) without network access."""
    try:
        manifests = validate_manifests_dir(manifests_dir)
    except ManifestValidationError as exc:
        typer.secho(f"Manifest validation failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"OK: {len(manifests)} manifest(s) under {manifests_dir}")
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

    if pilot_path.is_file():
        corpus = load_pilot_corpus(pilot_path)
        empty_sahidic = sum(
            1 for unit in corpus.units if not unit.sahidic.original_text
        )
        typer.echo(
            f"OK: pilot {corpus.id} - {len(corpus.units)} units "
            f"({empty_sahidic} empty Sahidic slots) from {pilot_path}"
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
        typer.echo(f"Note: pilot file not found at {pilot_path} (skipped)")


@sources_app.command("build-pilot")
def build_pilot(
    sword_root: Path = typer.Option(
        DEFAULT_SWORD,
        "--sword-root",
        help="Local SWORD tree containing CopSahHorner (under data/downloads/)",
        exists=True,
        file_okay=False,
    ),
    wh_textonly_dir: Path = typer.Option(
        DEFAULT_WH,
        "--wh-textonly-dir",
        help="Directory of Robinson WH *.WH textonly files",
        exists=True,
        file_okay=False,
    ),
    output: Path = typer.Option(
        DEFAULT_PILOT,
        "--output",
        "-o",
        help="Destination JSON for the redistributable pilot excerpt",
    ),
) -> None:
    """Build the committed pilot excerpt from local downloads (explicit setup)."""
    corpus = build_pilot_corpus(
        sword_root=sword_root,
        wh_textonly_dir=wh_textonly_dir,
    )
    write_pilot_corpus(corpus, output)
    typer.echo(f"Wrote {len(corpus.units)} parallel units to {output}")

# SPDX-License-Identifier: Apache-2.0
"""``thomas mappings`` Typer subcommands for JAK-38 phrase/loanword artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    load_canonical_splits,
    load_splits,
)
from zetodos_thomas.align.validate import validate_pilot_corpus
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_pilot_path,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.mappings.ablation import (
    build_ablation_packages,
    default_ablation_paths,
    write_ablation_packages,
)
from zetodos_thomas.mappings.loanwords import (
    default_verified_loanwords_path,
    verified_loanword_table,
    write_verified_loanword_table,
)
from zetodos_thomas.mappings.phrases import (
    default_statistical_constructions_path,
    generate_statistical_constructions,
    write_statistical_construction_table,
)
from zetodos_thomas.mappings.validate import validate_mappings_dir

mappings_app = typer.Typer(
    name="mappings",
    help="Phrase/construction mappings + verified loanwords (JAK-38).",
    no_args_is_help=True,
)


def _resolve_runtime_splits(root: Path, splits_path: Path | None) -> PilotSplits:
    omission = resolve_omission_fixtures_path(root)
    if splits_path is not None:
        return load_splits(splits_path, omission_path=omission)
    return load_canonical_splits(resolve_nt_split_path(root), omission)


@mappings_app.command("generate-phrases")
def generate_phrases_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    pilot_path: Path | None = typer.Option(None, "--pilot", dir_okay=False),
    splits_path: Path | None = typer.Option(
        None,
        "--splits",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
    eflomal_links: Path | None = typer.Option(
        None,
        "--eflomal-links",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    fast_align_links: Path | None = typer.Option(
        None,
        "--fast-align-links",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    compact: bool = typer.Option(False, "--compact/--pretty"),
) -> None:
    """Extract minimal statistical phrase/construction mappings (train only)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    pilot = pilot_path or resolve_pilot_path(root)
    if not pilot.is_file():
        typer.secho(f"pilot not found: {pilot}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    splits = _resolve_runtime_splits(root, splits_path)
    corpus = load_pilot_corpus(pilot)
    pilot_report = validate_pilot_corpus(corpus, splits=splits)
    if not pilot_report.ok:
        typer.secho("pilot source validation failed", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    link_paths: dict[str, Path] | None = None
    if eflomal_links is not None or fast_align_links is not None:
        link_paths = {}
        if eflomal_links is not None:
            link_paths["eflomal"] = eflomal_links
        if fast_align_links is not None:
            link_paths["fast_align"] = fast_align_links

    try:
        table = generate_statistical_constructions(
            corpus=corpus,
            splits=splits,
            repo_root=root,
            link_paths=link_paths,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    out = output or default_statistical_constructions_path(root)
    write_statistical_construction_table(table, out, compact=compact)
    typer.echo(f"Wrote {out}")
    typer.echo(
        f"constructions={table.summary.get('construction_count')} "
        f"families={table.summary.get('per_family_counts')} "
        f"aligners={table.summary.get('phase2_active_aligners')}"
    )


@mappings_app.command("write-loanwords")
def write_loanwords_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
    compact: bool = typer.Option(False, "--compact/--pretty"),
) -> None:
    """Write the curated CCL/DDGLC independently verified loanword subset."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        table = verified_loanword_table(repo_root=root)
    except (OSError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    out = output or default_verified_loanwords_path(root)
    write_verified_loanword_table(table, out, compact=compact)
    typer.echo(f"Wrote {out}")
    typer.echo(f"entries={table.summary.get('entry_count')} license={table.license}")


@mappings_app.command("build-ablation")
def build_ablation_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Write NT-aligner-only and NT-aligner+loanwords ablation packages."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    packages = build_ablation_packages()
    written = write_ablation_packages(packages, default_ablation_paths(root))
    for condition, path in written.items():
        typer.echo(f"Wrote {condition}: {path}")


@mappings_app.command("validate")
def validate_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Validate statistical / verified separation, spans, and ablations."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    splits = _resolve_runtime_splits(root, None)
    train_unit_ids = set(splits.unit_ids_for(SplitRole.TRAIN))
    report = validate_mappings_dir(root, train_unit_ids=train_unit_ids)
    typer.echo(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if not report.ok:
        raise typer.Exit(code=1)
    typer.echo("OK: JAK-38 mapping artifacts validation passed")

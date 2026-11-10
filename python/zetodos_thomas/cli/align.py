# SPDX-License-Identifier: Apache-2.0
"""``thomas align`` Typer subcommands for pilot validation and benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.align.benchmark import run_benchmark, write_benchmark_json
from zetodos_thomas.align.eflomal_probe import probe_eflomal
from zetodos_thomas.align.splits import (
    PilotSplits,
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

align_app = typer.Typer(
    name="align",
    help="Validate and benchmark the Greek-Sahidic alignment pilot (JAK-28).",
    no_args_is_help=True,
)


def _resolve_runtime_splits(root: Path, splits_path: Path | None) -> PilotSplits:
    """Load JAK-33 canonical split (+ omission catalog). Explicit path must exist."""
    omission = resolve_omission_fixtures_path(root)
    if splits_path is not None:
        return load_splits(splits_path, omission_path=omission)
    return load_canonical_splits(resolve_nt_split_path(root), omission)


def _default_reference_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "sources" / "alignment" / "reference"


def _default_benchmark_json(repo_root: Path) -> Path:
    return repo_root / "data" / "sources" / "alignment" / "benchmark-report.json"


@align_app.command("validate")
def validate_align(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    pilot_path: Path | None = typer.Option(
        None,
        "--pilot",
        help="Pilot corpus JSON",
        exists=False,
        dir_okay=False,
    ),
    splits_path: Path | None = typer.Option(
        None,
        "--splits",
        help=(
            "JAK-33 NtCorpusSplit JSON (must exist when provided; "
            "default: data/training/splits/horner-vaticanus-nt-split-v2.json)"
        ),
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Structural + normalization + omission/paratext validation (offline)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    pilot = pilot_path or resolve_pilot_path(root)
    if not pilot.is_file():
        typer.secho(f"pilot not found: {pilot}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    corpus = load_pilot_corpus(pilot)
    splits = _resolve_runtime_splits(root, splits_path)
    report = validate_pilot_corpus(corpus, splits=splits)
    typer.echo(
        f"pilot {corpus.id}: {report.unit_count} units "
        f"({report.nonempty_positive} nonempty positive, "
        f"{report.omission_units} omission)"
    )
    for issue in report.issues:
        color = typer.colors.RED if issue.severity == "error" else typer.colors.YELLOW
        loc = f" [{issue.unit_id}]" if issue.unit_id else ""
        typer.secho(f"{issue.severity}:{issue.code}{loc}: {issue.message}", fg=color)
    if not report.ok:
        raise typer.Exit(code=1)
    typer.echo("OK: pilot structural validation passed")


@align_app.command("benchmark")
def benchmark_align(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    pilot_path: Path | None = typer.Option(
        None,
        "--pilot",
        exists=False,
        dir_okay=False,
    ),
    splits_path: Path | None = typer.Option(
        None,
        "--splits",
        help=(
            "JAK-33 NtCorpusSplit JSON (must exist when provided; "
            "default: data/training/splits/horner-vaticanus-nt-split-v2.json)"
        ),
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    reference_dir: Path | None = typer.Option(
        None,
        "--reference-dir",
        help="Directory of provisional gold AlignmentDocument JSON files",
        exists=False,
        file_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write machine-readable benchmark JSON",
    ),
) -> None:
    """Run lexical baseline vs provisional gold; print summary (no pass threshold)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    pilot = pilot_path or resolve_pilot_path(root)
    ref_dir = reference_dir or _default_reference_dir(root)
    out = output or _default_benchmark_json(root)

    if not pilot.is_file():
        typer.secho(f"pilot not found: {pilot}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    splits = _resolve_runtime_splits(root, splits_path)
    refs = sorted(ref_dir.glob("*.json")) if ref_dir.is_dir() else []
    if not refs:
        typer.secho(f"no reference JSON under {ref_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    report = run_benchmark(pilot_path=pilot, splits=splits, reference_paths=refs)
    write_benchmark_json(report, out)

    overall = report["lexical_baseline"]["overall"]
    typer.echo(f"Wrote {out}")
    typer.echo(
        "lexical Dice baseline — "
        f"P={overall['precision']!r} R={overall['recall']!r} F1={overall['f1']!r} "
        f"(TP={overall['true_positives']}, "
        f"pred_n={overall['predicted_denominator']}, "
        f"gold_n={overall['gold_denominator']})"
    )
    typer.echo(f"validation_ok={report['validation_ok']}")
    efl = report["eflomal"]
    typer.echo(
        f"eflomal available={efl['available']} "
        f"default_path={efl['selected_for_default_path']}"
    )
    if report["omission_leak_problems"]:
        typer.secho(
            "omission fixture leakage detected",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if not report["validation_ok"]:
        raise typer.Exit(code=1)


@align_app.command("eflomal-probe")
def eflomal_probe_cmd() -> None:
    """Report whether eflomal can be imported (informational)."""
    result = probe_eflomal()
    typer.echo(json.dumps(result.__dict__, indent=2))
    # Always exit 0 — unavailability is expected without a native build.

# SPDX-License-Identifier: Apache-2.0
"""``thomas align`` Typer subcommands for pilot validation and benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from zetodos_thomas.align.benchmark import run_benchmark, write_benchmark_json
from zetodos_thomas.align.bitext import build_bitext
from zetodos_thomas.align.eflomal_probe import probe_eflomal
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    load_canonical_splits,
    load_splits,
)
from zetodos_thomas.align.statistical import (
    export_alignment_artifact,
    run_statistical_evaluation,
    write_eval_json,
)
from zetodos_thomas.align.validate import validate_pilot_corpus
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_pilot_path,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.integrations.aligners.discovery import discover_aligners
from zetodos_thomas.integrations.aligners.eflomal_adapter import (
    EflomalNotAvailableError,
    align_eflomal,
)
from zetodos_thomas.integrations.aligners.fast_align_adapter import (
    FastAlignNotAvailableError,
    align_fast_align,
)
from zetodos_thomas.integrations.aligners.types import AlignerConfig, AlignerName

align_app = typer.Typer(
    name="align",
    help="Validate and benchmark the Greek-Sahidic alignment pilot (JAK-28/36).",
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
            "NtCorpusSplit JSON (must exist when provided; "
            "default: data/training/splits/horner-vaticanus-nt-split-v3.json)"
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
            "NtCorpusSplit JSON (must exist when provided; "
            "default: data/training/splits/horner-vaticanus-nt-split-v3.json)"
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


def _default_reference_paths(root: Path) -> list[Path]:
    ref_dir = _default_reference_dir(root)
    return sorted(ref_dir.glob("*.json")) if ref_dir.is_dir() else []


def _default_statistical_dir(root: Path) -> Path:
    return root / "data" / "sources" / "alignment" / "statistical"


@align_app.command("eflomal-probe")
def eflomal_probe_cmd() -> None:
    """Report whether eflomal can be imported (informational; JAK-28 legacy)."""
    result = probe_eflomal()
    typer.echo(json.dumps(result.__dict__, indent=2))
    # Always exit 0 — unavailability is expected without a native build.


@align_app.command("aligner-probe")
def aligner_probe_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Report eflomal + fast_align availability after setup (JAK-36)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    report = {k: v.as_dict() for k, v in discover_aligners(root).items()}
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@align_app.command("setup-aligners")
def setup_aligners_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    skip_eflomal: bool = typer.Option(False, "--skip-eflomal"),
    skip_fast_align: bool = typer.Option(False, "--skip-fast-align"),
) -> None:
    """Clone/build eflomal + fast_align into data/local/aligners (needs network)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    script = root / "scripts" / "setup_aligners.py"
    cmd = [sys.executable, str(script), "--repo-root", str(root)]
    if skip_eflomal:
        cmd.append("--skip-eflomal")
    if skip_fast_align:
        cmd.append("--skip-fast-align")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@align_app.command("statistical-eval")
def statistical_eval_cmd(
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
    reference_dir: Path | None = typer.Option(
        None,
        "--reference-dir",
        file_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write statistical evaluation JSON",
    ),
    export_artifacts: bool = typer.Option(
        True,
        "--export-artifacts/--no-export-artifacts",
        help="Also write full-train×dev alignment JSON per available aligner",
    ),
    seed: int = typer.Option(36, "--seed", help="Reproducibility seed (config)"),
) -> None:
    """Run eflomal vs fast_align learning curves on v3 splits (JAK-36)."""
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
    refs = (
        sorted(reference_dir.glob("*.json"))
        if reference_dir is not None and reference_dir.is_dir()
        else _default_reference_paths(root)
    )
    out_dir = _default_statistical_dir(root)
    out = output or (out_dir / "statistical-aligner-eval-v1.json")
    corpus = load_pilot_corpus(pilot)
    config = AlignerConfig(seed=seed)
    report = run_statistical_evaluation(
        corpus=corpus,
        splits=splits,
        config=config,
        repo_root=root,
        reference_paths=refs,
    )
    write_eval_json(report, out)
    typer.echo(f"Wrote {out}")
    decision = report["jake_decision"]
    typer.echo(
        f"jake_decision={decision['recommendation']} "
        f"(adopted={decision['adopted_aligner']!r})"
    )
    for name, block in report["full_train_comparison"]["dev"].items():
        if not block.get("available"):
            typer.echo(f"  {name}: unavailable - {block.get('error', '')}")
            continue
        intrinsic = block.get("intrinsic") or {}
        typer.echo(
            f"  {name}: dev agreement="
            f"{intrinsic.get('mean_fwd_rev_agreement')!r} "
            f"density={intrinsic.get('mean_alignment_density')!r}"
        )

    if export_artifacts:
        bitext = build_bitext(corpus, splits)
        train = bitext.sentences_for(SplitRole.TRAIN)
        dev = bitext.sentences_for(SplitRole.DEV)
        seen: set[str] = set()
        joint = []
        for sentence in [*train, *dev]:
            if sentence.unit_id in seen:
                continue
            seen.add(sentence.unit_id)
            joint.append(sentence)
        provenance = {
            "owned_by": "JAK-36",
            "pilot_corpus_id": bitext.pilot_corpus_id,
            "split_id": bitext.split_id,
            "bitext_checksum_sha256": bitext.checksum_sha256,
            "coptic_segmentation": bitext.coptic_segmentation,
            "greek_tokenization": bitext.greek_tokenization,
            "protocol": "joint unsupervised train+dev; artifact includes all links",
            "seed": seed,
            "config": config.as_dict(),
        }
        for name, align_fn, err_type in (
            (AlignerName.EFLOMAL, align_eflomal, EflomalNotAvailableError),
            (AlignerName.FAST_ALIGN, align_fast_align, FastAlignNotAvailableError),
        ):
            art_path = out_dir / f"{name.value}-train-dev-links-v1.json"
            try:
                links = align_fn(joint, config=config, repo_root=root)
            except err_type as exc:
                typer.secho(
                    f"skip artifact {name.value}: {exc}", fg=typer.colors.YELLOW
                )
                continue
            export_alignment_artifact(
                links,
                path=art_path,
                provenance={**provenance, "aligner": name.value},
            )
            typer.echo(f"Wrote {art_path}")

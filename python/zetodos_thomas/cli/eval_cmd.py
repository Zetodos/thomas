# SPDX-License-Identifier: Apache-2.0
"""``thomas eval`` Typer subcommands for JAK-40 blinded reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.eval.blinded_reconstruct import (
    COMMITTED_ARTIFACT_MAX_BYTES,
    DEFAULT_FREEZE_REL,
    DEFAULT_NT_RECON_REL,
    DEFAULT_THOMAS_RECON_REL,
    BlindedReconstructError,
    reconstruct_blinded_package,
    write_committed_reconstruction_artifacts,
    write_reconstruction_package,
)
from zetodos_thomas.eval.surviving_letters import (
    score_nt_unit_letters,
    score_thomas7_surviving,
    try_score_thomas7_or_gate_report,
)
from zetodos_thomas.eval.validate_reconstruction import (
    validate_reconstruction_path,
)
from zetodos_thomas.ingest.paths import require_repo_root
from zetodos_thomas.schemas.blinded import (
    BlindedReconstructionPackage,
    BlindedTargetKind,
    SurvivingLetterEvalReport,
)
from zetodos_thomas.schemas.mappings import AblationCondition

eval_app = typer.Typer(
    name="eval",
    help="Blinded reconstruction + surviving-letter evaluation (JAK-40).",
    no_args_is_help=True,
)

DEFAULT_RECON_REL = DEFAULT_THOMAS_RECON_REL
DEFAULT_NT_RECON_PATH = DEFAULT_NT_RECON_REL
DEFAULT_SCORE_REL = Path(
    "data/reconstructions/jak40/thomas-7-surviving-letter-eval-v1.json"
)
DEFAULT_NT_SCORE_REL = Path("data/reconstructions/jak40/nt-holdout-letter-eval-v1.json")


def _parse_condition(value: str) -> AblationCondition:
    try:
        return AblationCondition(value)
    except ValueError as exc:
        allowed = ", ".join(c.value for c in AblationCondition)
        msg = f"unknown ablation condition {value!r}; expected one of: {allowed}"
        raise typer.BadParameter(msg) from exc


def _write_json(path: Path, payload: SurvivingLetterEvalReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    path.write_text(text + "\n", encoding="utf-8")


@eval_app.command("reconstruct")
def reconstruct_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    ablation: str = typer.Option(
        AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS.value,
        "--ablation",
        help="Ablation arm (JAK-38).",
    ),
    thomas_only: bool = typer.Option(
        False,
        "--thomas-only/--with-nt-holdouts",
        help="Reconstruct Thomas 7 only (skip NT holdout Sahidic).",
    ),
    commit_split: bool = typer.Option(
        True,
        "--commit-split/--single-file",
        help=(
            "Default: write Thomas 7 + NT holdout sibling files under the "
            "1024 KiB large-file hook. --single-file writes one JSON via -o."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Single-file BlindedReconstructionPackage path (--single-file).",
    ),
    compact: bool = typer.Option(False, "--compact/--pretty"),
) -> None:
    """Blinded decode of Thomas 7 (+ NT holdouts) without consulting Greek."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    condition = _parse_condition(ablation)
    try:
        if commit_split and not thomas_only and output is None:
            digests = write_committed_reconstruction_artifacts(root, ablation=condition)
            typer.echo(f"Wrote {root / DEFAULT_RECON_REL}")
            typer.echo(f"Wrote {root / DEFAULT_NT_RECON_PATH}")
            typer.echo(f"Wrote {root / DEFAULT_FREEZE_REL}")
            typer.echo(
                f"thomas_sha256={digests['thomas'][:12]}… "
                f"nt_sha256={digests['nt_holdout'][:12]}… "
                f"freeze={digests['freeze'][:12]}… "
                f"ablation={condition.value} greek_consulted=False"
            )
            return

        package = reconstruct_blinded_package(
            root,
            ablation=condition,
            include_nt_holdouts=not thomas_only,
        )
    except (OSError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    out = output or (root / DEFAULT_RECON_REL)
    # Enforce the 1 MiB commit hook for paths under data/reconstructions/jak40/.
    under_commit_dir = "data/reconstructions/jak40" in str(out).replace("\\", "/")
    try:
        digest = write_reconstruction_package(
            package,
            out,
            max_bytes=COMMITTED_ARTIFACT_MAX_BYTES if under_commit_dir else None,
        )
    except BlindedReconstructError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {out}")
    typer.echo(
        f"units={package.summary.get('unit_count')} "
        f"candidates_spans={package.summary.get('spans_with_candidates')} "
        f"abstained_spans={package.summary.get('spans_abstained')} "
        f"ablation={package.ablation_condition.value} "
        f"sha256={digest[:12]}… "
        f"greek_consulted={package.blinding.get('greek_consulted_during_reconstruction')}"
    )
    if compact:
        pass


@eval_app.command("score-thomas7")
def score_thomas7_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    reconstruction: Path | None = typer.Option(
        None,
        "--reconstruction",
        "-r",
        exists=True,
        dir_okay=False,
        readable=True,
        help="BlindedReconstructionPackage JSON.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write SurvivingLetterEvalReport JSON.",
    ),
    allow_unreviewed: bool = typer.Option(
        False,
        "--allow-unreviewed/--require-human-checked",
        help=(
            "Protocol smoke only: score without human-checked gate "
            "(marks official=false). Default requires human-checked."
        ),
    ),
    gate_report: bool = typer.Option(
        False,
        "--gate-report/--fail-closed",
        help=(
            "When review is pending, write a gate-blocked report instead of "
            "exiting non-zero (default fail-closed)."
        ),
    ),
) -> None:
    """Score Thomas 7 vs P.Oxy. 654 surviving letters (fragments at eval only)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    recon_path = reconstruction or (root / DEFAULT_RECON_REL)
    if not recon_path.is_file():
        typer.secho(f"missing reconstruction: {recon_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    package = BlindedReconstructionPackage.model_validate(
        json.loads(recon_path.read_text(encoding="utf-8"))
    )
    try:
        if gate_report and not allow_unreviewed:
            report = try_score_thomas7_or_gate_report(root, package)
        else:
            report = score_thomas7_surviving(
                root,
                package,
                require_human_checked=not allow_unreviewed,
            )
    except (OSError, ValueError, KeyError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    out = output or (root / DEFAULT_SCORE_REL)
    _write_json(out, report)
    typer.echo(f"Wrote {out}")
    typer.echo(
        f"official={report.official} review_status={report.review_status} "
        f"scores={len(report.scores)} "
        f"gate_blocked={report.blinding.get('gate_blocked', False)}"
    )
    if not report.official and not allow_unreviewed and not gate_report:
        raise typer.Exit(code=1)
    if report.blinding.get("gate_blocked") and not gate_report:
        raise typer.Exit(code=1)


@eval_app.command("score-nt-holdouts")
def score_nt_holdouts_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    reconstruction: Path | None = typer.Option(
        None,
        "--reconstruction",
        "-r",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Letter-agree NT holdout reconstructions vs Vaticanus B* (eval-time Greek)."""
    from zetodos_thomas.eval.blinded_reconstruct import holdout_units

    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    recon_path = reconstruction or (root / DEFAULT_NT_RECON_PATH)
    if not recon_path.is_file():
        # Fall back to combined/single-file Thomas+NT package if present.
        fallback = root / DEFAULT_RECON_REL
        if fallback.is_file():
            recon_path = fallback
    if not recon_path.is_file():
        typer.secho(f"missing reconstruction: {recon_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    package = BlindedReconstructionPackage.model_validate(
        json.loads(recon_path.read_text(encoding="utf-8"))
    )
    by_id = {u.unit_id: u for u in holdout_units(root)}
    scores = []
    for unit in package.units:
        if unit.target_kind is not BlindedTargetKind.NT_HOLDOUT:
            continue
        gold_unit = by_id.get(unit.unit_id)
        if gold_unit is None:
            continue
        scores.append(
            score_nt_unit_letters(
                unit_id=unit.unit_id,
                top1_greek=unit.top1_greek_surfaces,
                greek_unit=gold_unit,
                review_status="provisional-draft",
                official=False,
            )
        )

    report = SurvivingLetterEvalReport(
        id="jak40-nt-holdout-letter-eval-v1",
        title="NT holdout letter evaluation (JAK-40; provisional gold)",
        reconstruction_package_id=package.id,
        eval_package_id="horner-vaticanus-nt-holdout",
        review_status="provisional-draft",
        official=False,
        nt_holdout_letter_scores=scores,
        blinding={
            "greek_loaded_during_reconstruction": False,
            "greek_loaded_at_scoring": True,
            "nt_holdout_only": True,
        },
        summary={
            "n_scores": len(scores),
            "official": False,
            "mean_recall": (
                sum(s.recall or 0.0 for s in scores) / len(scores) if scores else None
            ),
        },
        provenance={"owned_by": "JAK-40", "llm": False, "network": False},
        notes=[
            "NT holdout letter scores use Vaticanus B* at eval time only.",
            "Gold remains provisional-draft until human-checked (JAK-41 gate).",
            "Train/dev passages are not scored as holdout.",
        ],
    )
    out = output or (root / DEFAULT_NT_SCORE_REL)
    _write_json(out, report)
    typer.echo(f"Wrote {out}")
    typer.echo(f"nt_holdout_scores={len(scores)} official=False")


@eval_app.command("validate-reconstruction")
def validate_reconstruction_cmd(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="BlindedReconstructionPackage JSON.",
    ),
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Validate blinded reconstruction schema + structured integrity checks."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    report = validate_reconstruction_path(path, repo_root=root)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise typer.Exit(code=1)
    typer.echo("OK: JAK-40 blinded reconstruction validation passed")

# SPDX-License-Identifier: Apache-2.0
"""``thomas decode`` Typer subcommands for the JAK-39 minimal decoder."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.ingest.paths import require_repo_root
from zetodos_thomas.reconstruct.decoder import decode_request, request_from_surfaces
from zetodos_thomas.reconstruct.model import load_decode_model
from zetodos_thomas.reconstruct.ranking import RANKING_RULES_DOC, RANKING_RULES_ID
from zetodos_thomas.reconstruct.validate import validate_decode_result_path
from zetodos_thomas.schemas.decode import DecodeRequest
from zetodos_thomas.schemas.mappings import AblationCondition

decode_app = typer.Typer(
    name="decode",
    help="Minimal deterministic Coptic→Greek decoder (JAK-39).",
    no_args_is_help=True,
)


def _parse_condition(value: str) -> AblationCondition:
    try:
        return AblationCondition(value)
    except ValueError as exc:
        allowed = ", ".join(c.value for c in AblationCondition)
        msg = f"unknown ablation condition {value!r}; expected one of: {allowed}"
        raise typer.BadParameter(msg) from exc


@decode_app.command("run")
def run_cmd(
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
        help="Ablation arm condition (JAK-38).",
    ),
    request_path: Path | None = typer.Option(
        None,
        "--request",
        "-r",
        exists=True,
        dir_okay=False,
        readable=True,
        help="DecodeRequest JSON (span/construction_id surface).",
    ),
    surfaces: list[str] | None = typer.Option(
        None,
        "--surface",
        "-s",
        help="Coptic surface token(s) for a one-span request (repeatable).",
    ),
    unit_id: str = typer.Option("demo.unit", "--unit-id"),
    construction_id: str | None = typer.Option(
        None,
        "--construction-id",
        help="Optional JAK-38 construction_id hint for --surface mode.",
    ),
    request_id: str = typer.Option("cli-request", "--request-id"),
    max_candidates: int = typer.Option(5, "--max-candidates", min=1, max=50),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write DecodeResult JSON (default: stdout).",
    ),
    compact: bool = typer.Option(False, "--compact/--pretty"),
) -> None:
    """Decode Coptic spans using frozen JAK-37/38 tables (offline, no LLM)."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    condition = _parse_condition(ablation)
    try:
        model = load_decode_model(root, condition=condition)
    except (OSError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if request_path is not None:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        request = DecodeRequest.model_validate(payload)
        if request.ablation_condition is not condition:
            typer.secho(
                "request.ablation_condition does not match --ablation",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    elif surfaces:
        request = request_from_surfaces(
            request_id=request_id,
            unit_id=unit_id,
            surfaces=list(surfaces),
            ablation_condition=condition,
            construction_id=construction_id,
            max_candidates_per_span=max_candidates,
        )
    else:
        typer.secho(
            "provide --request JSON or one or more --surface tokens",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        result = decode_request(model, request)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    text = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(text)
    typer.echo(
        f"spans={result.summary.get('span_count')} "
        f"candidates={result.summary.get('spans_with_candidates')} "
        f"abstained={result.summary.get('spans_abstained')} "
        f"ablation={result.ablation_condition.value} "
        f"rules={RANKING_RULES_ID}",
        err=output is None,
    )


@decode_app.command("validate")
def validate_cmd(
    result_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="DecodeResult JSON to validate.",
    ),
) -> None:
    """Validate decode output schema, ranking, and leakage guards."""
    report = validate_decode_result_path(result_path)
    typer.echo(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if not report.ok:
        raise typer.Exit(code=1)
    typer.echo("OK: JAK-39 decode result validation passed")


@decode_app.command("ranking-rules")
def ranking_rules_cmd() -> None:
    """Print the stable ranking / tie-break documentation."""
    typer.echo(f"id: {RANKING_RULES_ID}")
    typer.echo(RANKING_RULES_DOC)

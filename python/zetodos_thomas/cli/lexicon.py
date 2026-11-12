# SPDX-License-Identifier: Apache-2.0
"""``thomas lexicon`` Typer subcommands for lexical candidates (JAK-37)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.align.bitext import build_bitext
from zetodos_thomas.align.splits import (
    PilotSplits,
    SplitRole,
    load_canonical_splits,
    load_splits,
)
from zetodos_thomas.ingest.paths import (
    require_repo_root,
    resolve_nt_split_path,
    resolve_omission_fixtures_path,
    resolve_pilot_path,
)
from zetodos_thomas.ingest.pilot import load_pilot_corpus
from zetodos_thomas.integrations.aligners.types import AlignerConfig, AlignerName
from zetodos_thomas.lexicon.candidates import (
    default_lexicon_dir,
    default_table_path,
    export_train_link_artifacts,
    generate_lexical_candidates,
    write_lexical_candidate_table,
)
from zetodos_thomas.lexicon.validate import validate_lexical_candidate_path

lexicon_app = typer.Typer(
    name="lexicon",
    help="Coptic→Greek lexical candidate table (JAK-37).",
    no_args_is_help=True,
)


def _resolve_runtime_splits(root: Path, splits_path: Path | None) -> PilotSplits:
    omission = resolve_omission_fixtures_path(root)
    if splits_path is not None:
        return load_splits(splits_path, omission_path=omission)
    return load_canonical_splits(resolve_nt_split_path(root), omission)


@lexicon_app.command("generate-candidates")
def generate_candidates_cmd(
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
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write lexical candidate JSON",
    ),
    export_train_links: bool = typer.Option(
        True,
        "--export-train-links/--no-export-train-links",
        help="Also write compact train-only link JSON per aligner",
    ),
    eflomal_links: Path | None = typer.Option(
        None,
        "--eflomal-links",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Reuse compact eflomal train-links JSON instead of running the aligner",
    ),
    fast_align_links: Path | None = typer.Option(
        None,
        "--fast-align-links",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Reuse compact fast_align train-links JSON instead of running the aligner",
    ),
    seed: int = typer.Option(36, "--seed"),
    compact: bool = typer.Option(
        True,
        "--compact/--pretty",
        help="Compact separators (default); --pretty indents for debugging",
    ),
    columnar: bool = typer.Option(
        True,
        "--columnar/--expanded",
        help="Columnar-v1 export (default; required to stay under large-file ceiling)",
    ),
) -> None:
    """Build the three-layer Coptic→Greek lexical candidate table (train only)."""
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
    bitext = build_bitext(corpus, splits)
    out = output or default_table_path(root)
    link_paths: dict[str, Path] | None = None
    if eflomal_links is not None or fast_align_links is not None:
        link_paths = {}
        if eflomal_links is not None:
            link_paths[AlignerName.EFLOMAL.value] = eflomal_links
        if fast_align_links is not None:
            link_paths[AlignerName.FAST_ALIGN.value] = fast_align_links

    try:
        table, aligner_links = generate_lexical_candidates(
            corpus=corpus,
            splits=splits,
            bitext_sentences=bitext.sentences_for(SplitRole.TRAIN),
            repo_root=root,
            config=AlignerConfig(seed=seed),
            link_paths=link_paths,
        )
    except (RuntimeError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    # Attach bitext checksum into provenance for audit.
    table.provenance = {
        **table.provenance,
        "bitext_checksum_sha256": bitext.checksum_sha256,
        "excluded_from_training_stack": bitext.excluded_from_training_stack,
    }
    write_lexical_candidate_table(table, out, compact=compact, columnar=columnar)
    typer.echo(f"Wrote {out}")
    typer.echo(
        f"observed_wordforms={table.summary.get('observed_wordform_types')} "
        f"inferred_lemmas={table.summary.get('inferred_lemma_types')} "
        f"confidence_rows={table.summary.get('confidence_estimate_rows')} "
        f"aligners={table.summary.get('phase2_active_aligners')}"
    )

    if export_train_links:
        lex_dir = default_lexicon_dir(root)
        written = export_train_link_artifacts(
            aligner_links,
            out_dir=lex_dir,
            provenance={
                **table.provenance,
                "bitext_checksum_sha256": bitext.checksum_sha256,
            },
        )
        for name, path in written.items():
            typer.echo(f"Wrote train links ({name}): {path}")


@lexicon_app.command("validate-candidates")
def validate_candidates_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    table_path: Path | None = typer.Option(
        None,
        "--table",
        exists=False,
        dir_okay=False,
        help="Lexical candidate JSON (default: data/sources/lexicon/...-v1.json)",
    ),
) -> None:
    """Validate layering, train-only observations, and leakage controls."""
    try:
        root = require_repo_root(repo_root)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    path = table_path or default_table_path(root)
    if not path.is_file():
        typer.secho(f"table not found: {path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    report = validate_lexical_candidate_path(path)
    typer.echo(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    if not report.ok:
        raise typer.Exit(code=1)
    typer.echo("OK: lexical candidate table validation passed")

# SPDX-License-Identifier: Apache-2.0
"""``thomas dictionary`` — parallel-text Coptic→Greek dictionary (not glosses)."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.dictionary.run import run_parallel_dictionary
from zetodos_thomas.dictionary.wider import (
    run_gospel_dictionary,
    run_nt_dictionary,
    run_ot_dictionary,
)
from zetodos_thomas.ingest.paths import require_repo_root

dictionary_app = typer.Typer(
    name="dictionary",
    help="Dictionary from Coptic–Greek parallels. Thomas Greek is evaluation only.",
    no_args_is_help=True,
)


@dictionary_app.command("build")
def build_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for dictionary, Thomas reconstruction, and metrics",
    ),
) -> None:
    """Fit parallel-dict-v1, apply it to Dilley Thomas, score fragment witnesses.

    Train evidence is the existing dual-aligner NT table (eflomal and fast_align).
    No handwritten Coptic→Greek pairs. Fragment Greek is read only at score time.
    """
    root = require_repo_root(repo_root)
    metrics = run_parallel_dictionary(root, output_dir=output, write=True)
    rate = metrics["loanword_stem_match_rate"]
    recall = metrics["content_type_recall"]
    precision = metrics["content_emission_precision"]
    typer.echo(
        "parallel-dict-v1 "
        f"loanword_stem_match_rate={rate} "
        f"content_type_recall={recall} "
        f"content_emission_precision={precision} "
        f"snap_enabled={metrics['snap_enabled']} "
        f"approach_sufficient={metrics['approach_sufficient']}"
    )
    typer.echo(str(metrics["approach_sufficient_reason"]))


@dictionary_app.command("build-gospels")
def build_gospels_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory for the Gospel re-fit artifacts",
    ),
    download: bool = typer.Option(
        True,
        "--download/--no-download",
        help="Fetch missing Vaticanus Gospel teiraw files (no images)",
    ),
) -> None:
    """Re-fit both aligners on the four Gospels, then score Thomas fragments.

    v3 dev, holdout, and omission verses stay out of training. Needs a local
    CopSahHorner module and both aligners. No handwritten glosses.
    """
    root = require_repo_root(repo_root)
    metrics = run_gospel_dictionary(root, output_dir=output, download=download)
    rate = metrics["loanword_exact_match_rate"]
    recall = metrics["content_type_recall"]
    typer.echo(
        "parallel-dict-v2 "
        f"loanword_exact_match_rate={rate} "
        f"loanword_stem_match_rate={metrics['loanword_stem_match_rate']} "
        f"content_type_recall={recall} "
        f"content_emission_precision={metrics['content_emission_precision']} "
        f"train_sentences={metrics['train_sentences']} "
        f"approach_sufficient={metrics['approach_sufficient']}"
    )
    typer.echo(str(metrics["approach_sufficient_reason"]))


@dictionary_app.command("build-nt")
def build_nt_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Re-fit both aligners on the whole Horner x Vaticanus NT, then score.

    v3 dev, holdout, and omission verses stay out of training. No handwritten
    glosses. Thomas Greek is read only at score time.
    """
    root = require_repo_root(repo_root)
    metrics = run_nt_dictionary(root, output_dir=output)
    typer.echo(
        "parallel-dict-v3 "
        f"loanword_exact_match_rate={metrics['loanword_exact_match_rate']} "
        f"loanword_stem_match_rate={metrics['loanword_stem_match_rate']} "
        f"content_type_recall={metrics['content_type_recall']} "
        f"content_emission_precision={metrics['content_emission_precision']} "
        f"train_sentences={metrics['train_sentences']} "
        f"approach_sufficient={metrics['approach_sufficient']}"
    )
    typer.echo(str(metrics["approach_sufficient_reason"]))


@dictionary_app.command("build-ot")
def build_ot_cmd(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Re-fit on Horner x Vaticanus plus Sahidic OT x Swete, then score.

    OT and Swete are CC BY-SA 4.0. Dev, holdout, and omission stay out.
    No handwritten glosses. Thomas Greek is read only at score time.
    """
    root = require_repo_root(repo_root)
    metrics = run_ot_dictionary(root, output_dir=output)
    typer.echo(
        "parallel-dict-v4 "
        f"loanword_exact_match_rate={metrics['loanword_exact_match_rate']} "
        f"loanword_stem_match_rate={metrics['loanword_stem_match_rate']} "
        f"content_type_recall={metrics['content_type_recall']} "
        f"content_emission_precision={metrics['content_emission_precision']} "
        f"train_sentences={metrics['train_sentences']} "
        f"approach_sufficient={metrics['approach_sufficient']}"
    )
    typer.echo(str(metrics["approach_sufficient_reason"]))

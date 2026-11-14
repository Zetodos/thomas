# SPDX-License-Identifier: Apache-2.0
"""``thomas dictionary`` — parallel-text Coptic→Greek dictionary (not glosses)."""

from __future__ import annotations

from pathlib import Path

import typer

from zetodos_thomas.dictionary.run import run_parallel_dictionary
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

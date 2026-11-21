# SPDX-License-Identifier: Apache-2.0
"""``thomas eval`` Typer subcommands: offline holdout & P.Oxy. benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from zetodos_thomas.cli.model_cmd import MODEL_REL, MODEL_V2_REL
from zetodos_thomas.eval.benchmark import (
    BENCHMARK_DOC_REL,
    EVAL_REPORT_REL,
    format_benchmark_markdown,
    run_full_evaluation,
)
from zetodos_thomas.ingest.paths import require_repo_root

EVAL_REPORT_V2_REL = Path("data/generated/eval-report-v2.json")
BENCHMARK_DOC_V2_REL = Path("docs/benchmark-report-v2.md")

eval_app = typer.Typer(
    name="eval",
    help="Run offline NT holdout and lacuna-aware P.Oxy. fragment evaluations.",
    no_args_is_help=True,
)


@eval_app.command("run")
def eval_run(
    repo_root: Path | None = typer.Option(
        None,
        "--repo-root",
        help="Checkout root (default: auto-detect)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    report_out: Path | None = typer.Option(
        None,
        "--report-out",
        help="Output JSON report path (default: data/generated/eval-report-v1.json)",
    ),
    doc_out: Path | None = typer.Option(
        None,
        "--doc-out",
        help="Markdown benchmark report path (default: docs/benchmark-report.md)",
    ),
    model: str = typer.Option(
        "v1",
        "--model",
        help="Model track to evaluate: v1 or v2",
    ),
) -> None:
    """Run NT Gospel holdouts + P.Oxy. 1/654/655 benchmark and emit reports."""
    root = require_repo_root(repo_root)
    is_v2 = model.lower() in {"v2", "model-v2"}
    model_dir = (root / MODEL_V2_REL) if is_v2 else (root / MODEL_REL)
    marker = "seq2seq-config.json" if is_v2 else "decoder-weights.json"
    if not (model_dir / marker).is_file():
        cmd = "thomas model build-v2" if is_v2 else "thomas model build"
        typer.secho(
            f"Missing frozen model at {model_dir}; run `{cmd}` first",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    report = run_full_evaluation(repo_root=root, model_dir=model_dir)
    json_path = report_out or (
        root / (EVAL_REPORT_V2_REL if is_v2 else EVAL_REPORT_REL)
    )
    md_path = doc_out or (root / (BENCHMARK_DOC_V2_REL if is_v2 else BENCHMARK_DOC_REL))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = (
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    )
    json_path.write_text(serialized, encoding="utf-8")
    md_path.write_text(format_benchmark_markdown(report), encoding="utf-8")

    nt = report.nt_holdout
    cv = report.poxy_convergent
    ov = report.poxy_overall
    typer.echo(
        f"NT Holdout ({nt.unit_count} verses): "
        f"Content Lemma F1={nt.content_lemma_f1:.4f} "
        f"(P={nt.content_lemma_precision:.4f}, R={nt.content_lemma_recall:.4f}), "
        f"Surface F1={nt.inflected_surface_f1:.4f}, "
        f"Morph={nt.morph_accuracy_on_matched_lemmas:.4f}, "
        f"chrF++={nt.chrf_pp:.4f}, BLEU-4={nt.bleu_4:.4f}, Cov={nt.coverage:.4f}"
    )
    typer.echo(
        f"P.Oxy. Convergent ({cv.span_count} spans / {cv.unit_count} units): "
        f"Content Lemma F1={cv.content_lemma_f1:.4f} "
        f"(R={cv.content_lemma_recall:.4f}, "
        f"Top-3 R={cv.top3_content_lemma_recall:.4f}), "
        f"Char F1 exact/itac={cv.surviving_char_f1_exact:.4f}/"
        f"{cv.surviving_char_f1_itacism:.4f}, "
        f"Loanwords={cv.loanword_exact_match_rate:.4f} ({cv.loanword_total})"
    )
    typer.echo(
        f"P.Oxy. Overall ({ov.span_count} spans / {ov.unit_count} units): "
        f"Content Lemma F1={ov.content_lemma_f1:.4f} "
        f"(R={ov.content_lemma_recall:.4f}), "
        f"Char F1 itacism={ov.surviving_char_f1_itacism:.4f}"
    )
    typer.echo(f"Wrote {json_path} ({json_path.stat().st_size} bytes) and {md_path}")

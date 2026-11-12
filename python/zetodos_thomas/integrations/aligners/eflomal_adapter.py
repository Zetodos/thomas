# SPDX-License-Identifier: Apache-2.0
"""Typed eflomal adapter (offline after explicit setup/download).

Uses the eflomal Python package when importable; otherwise raises with a
setup hint. Alignments are bidirectional (forward + reverse) then symmetrized
in-process. Supports optional Dirichlet priors for train-then-infer (holdout).
No LLM.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

from zetodos_thomas.integrations.aligners.discovery import probe_eflomal
from zetodos_thomas.integrations.aligners.pharaoh import (
    format_pharaoh_line,
    parse_pharaoh_batch,
    symmetrize_batch,
)
from zetodos_thomas.integrations.aligners.types import (
    AlignerConfig,
    AlignerName,
    AlignmentLinks,
    BitextSentence,
)


class EflomalNotAvailableError(RuntimeError):
    """Raised when eflomal has not been set up in this environment."""


def _write_side(path: Path, sentences: list[BitextSentence], *, source: bool) -> None:
    lines = [
        " ".join(s.source_tokens if source else s.target_tokens) + "\n"
        for s in sentences
    ]
    path.write_text("".join(lines), encoding="utf-8")


def write_priors_from_links(
    links: AlignmentLinks,
    priors_path: Path,
) -> None:
    """Estimate eflomal Dirichlet priors from an AlignmentLinks batch."""
    eflomal_mod: Any = importlib.import_module("eflomal")
    src_lines = [" ".join(s.source_tokens) + "\n" for s in links.sentences]
    trg_lines = [" ".join(s.target_tokens) + "\n" for s in links.sentences]
    fwd_lines = [format_pharaoh_line(a.pairs) + "\n" for a in links.forward]
    rev_lines = [format_pharaoh_line(a.pairs) + "\n" for a in links.reverse]
    priors_tuple = eflomal_mod.calculate_priors(
        src_lines, trg_lines, fwd_lines, rev_lines, reverse=False
    )
    priors_path.parent.mkdir(parents=True, exist_ok=True)
    with priors_path.open("w", encoding="utf-8") as priors_f:
        eflomal_mod.write_priors(priors_f, *priors_tuple)


def align_eflomal(
    sentences: list[BitextSentence],
    *,
    config: AlignerConfig | None = None,
    repo_root: Path | None = None,
    work_dir: Path | None = None,
    priors_path: Path | None = None,
) -> AlignmentLinks:
    """Align ``sentences`` with eflomal (src=Sahidic, trg=Greek).

    When ``priors_path`` is set, Dirichlet priors from a prior train-only run
    are applied (frozen-model inference for holdout / sealed eval).
    """
    if not sentences:
        msg = "eflomal requires at least one bitext sentence"
        raise ValueError(msg)
    cfg = config or AlignerConfig()
    availability = probe_eflomal(repo_root)
    if not availability.available or not availability.python_module:
        raise EflomalNotAvailableError(availability.detail)

    # Optional native package — load dynamically so CI ty passes without it.
    eflomal_mod: Any = importlib.import_module("eflomal")

    # Best-effort reproducibility: eflomal's C sampler reads OS entropy;
    # PYTHONHASHSEED + fixed n_samplers still reduce variance across runs.
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))

    aligner = eflomal_mod.Aligner(
        model=cfg.eflomal_model,
        n_samplers=cfg.eflomal_n_samplers,
        null_prior=cfg.eflomal_null_prior,
    )

    unit_ids = [s.unit_id for s in sentences]
    src_lens = [len(s.source_tokens) for s in sentences]
    trg_lens = [len(s.target_tokens) for s in sentences]

    with tempfile.TemporaryDirectory(prefix="thomas-eflomal-") as tmp:
        tmp_path = Path(work_dir) if work_dir is not None else Path(tmp)
        tmp_path.mkdir(parents=True, exist_ok=True)
        src_path = tmp_path / "src.txt"
        trg_path = tmp_path / "trg.txt"
        fwd_path = tmp_path / "fwd.align"
        rev_path = tmp_path / "rev.align"
        _write_side(src_path, sentences, source=True)
        _write_side(trg_path, sentences, source=False)

        priors_input = None
        if priors_path is not None:
            priors_input = priors_path.open(encoding="utf-8")
        try:
            with (
                src_path.open(encoding="utf-8") as src_f,
                trg_path.open(encoding="utf-8") as trg_f,
            ):
                aligner.align(
                    src_f,
                    trg_f,
                    links_filename_fwd=str(fwd_path),
                    links_filename_rev=str(rev_path),
                    priors_input=priors_input,
                    quiet=True,
                )
        finally:
            if priors_input is not None:
                priors_input.close()

        fwd_lines = fwd_path.read_text(encoding="utf-8").splitlines()
        rev_lines = rev_path.read_text(encoding="utf-8").splitlines()
        # Pad if eflomal omits trailing empty lines for empty alignments.
        while len(fwd_lines) < len(unit_ids):
            fwd_lines.append("")
        while len(rev_lines) < len(unit_ids):
            rev_lines.append("")
        forward = parse_pharaoh_batch(fwd_lines[: len(unit_ids)], unit_ids)
        reverse = parse_pharaoh_batch(rev_lines[: len(unit_ids)], unit_ids)

    symmetrized = symmetrize_batch(
        forward,
        reverse,
        src_lens,
        trg_lens,
        method=cfg.symmetrization,
    )
    notes = [
        "eflomal Gibbs sampler uses OS entropy; seed is recorded for "
        "config provenance but does not fully pin C RNG.",
        "Source=Sahidic (bound-group tokens), target=Greek (whitespace).",
        "No LLM; offline after setup.",
    ]
    if priors_path is not None:
        notes.append(f"Aligned with frozen Dirichlet priors from {priors_path.name}.")
    return AlignmentLinks(
        aligner=AlignerName.EFLOMAL,
        sentences=list(sentences),
        forward=forward,
        reverse=reverse,
        symmetrized=symmetrized,
        symmetrization=cfg.symmetrization,
        config={
            **cfg.as_dict(),
            "binary_path": availability.binary_path,
            "python_module": availability.python_module,
            "priors_path": str(priors_path) if priors_path else None,
        },
        notes=notes,
    )

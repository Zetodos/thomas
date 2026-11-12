# SPDX-License-Identifier: Apache-2.0
"""Typed fast_align adapter (offline after explicit binary setup).

Runs forward and reverse ``fast_align`` subprocesses on identical bitext,
then symmetrizes with the pure-Python grow-diag-final-and implementation.
No LLM.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from zetodos_thomas.integrations.aligners.discovery import probe_fast_align
from zetodos_thomas.integrations.aligners.pharaoh import (
    parse_pharaoh_batch,
    symmetrize_batch,
)
from zetodos_thomas.integrations.aligners.types import (
    AlignerConfig,
    AlignerName,
    AlignmentLinks,
    BitextSentence,
)


class FastAlignNotAvailableError(RuntimeError):
    """Raised when fast_align has not been set up in this environment."""


def _run_fast_align(
    binary: Path,
    joint_path: Path,
    *,
    reverse: bool,
    config: AlignerConfig,
) -> list[str]:
    cmd = [str(binary), "-i", str(joint_path), "-I", str(config.fast_align_iterations)]
    if config.fast_align_variational_bayes:
        cmd.append("-v")
    if config.fast_align_favor_diagonal:
        cmd.append("-d")
    if config.fast_align_optimize_diagonal:
        cmd.append("-o")
    if reverse:
        cmd.append("-r")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    # Alignments on stdout; EM diagnostics on stderr. Keep blank lines.
    return result.stdout.splitlines()


def align_fast_align(
    sentences: list[BitextSentence],
    *,
    config: AlignerConfig | None = None,
    repo_root: Path | None = None,
    work_dir: Path | None = None,
) -> AlignmentLinks:
    """Align ``sentences`` with fast_align (left=Sahidic, right=Greek)."""
    if not sentences:
        msg = "fast_align requires at least one bitext sentence"
        raise ValueError(msg)
    cfg = config or AlignerConfig()
    availability = probe_fast_align(repo_root)
    if not availability.available or not availability.binary_path:
        raise FastAlignNotAvailableError(availability.detail)

    binary = Path(availability.binary_path)
    unit_ids = [s.unit_id for s in sentences]
    src_lens = [len(s.source_tokens) for s in sentences]
    trg_lens = [len(s.target_tokens) for s in sentences]

    with tempfile.TemporaryDirectory(prefix="thomas-fast-align-") as tmp:
        tmp_path = Path(work_dir) if work_dir is not None else Path(tmp)
        tmp_path.mkdir(parents=True, exist_ok=True)
        joint_path = tmp_path / "bitext.txt"
        joint_path.write_text(
            "\n".join(s.as_pharaoh_line() for s in sentences) + "\n",
            encoding="utf-8",
        )
        fwd_lines = _run_fast_align(binary, joint_path, reverse=False, config=cfg)
        rev_lines = _run_fast_align(binary, joint_path, reverse=True, config=cfg)
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
    return AlignmentLinks(
        aligner=AlignerName.FAST_ALIGN,
        sentences=list(sentences),
        forward=forward,
        reverse=reverse,
        symmetrized=symmetrized,
        symmetrization=cfg.symmetrization,
        config={
            **cfg.as_dict(),
            "binary_path": availability.binary_path,
        },
        notes=[
            "fast_align EM is deterministic for fixed -I / -d / -o / -v flags.",
            "Reverse pass uses -r; pairs kept in left=source (Sahidic) coords.",
            "Symmetrization: pure-Python grow-diag-final-and (atools optional).",
            "No LLM; offline after setup.",
        ],
    )

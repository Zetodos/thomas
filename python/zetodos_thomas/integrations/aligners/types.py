# SPDX-License-Identifier: Apache-2.0
"""Shared types for statistical word-aligner adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AlignerName(StrEnum):
    """Supported local statistical aligners (JAK-36 bake-off)."""

    EFLOMAL = "eflomal"
    FAST_ALIGN = "fast_align"


@dataclass(frozen=True)
class BitextSentence:
    """One parallel sentence with tokenized sides and stable unit provenance."""

    unit_id: str
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    source_token_ids: tuple[str, ...]
    target_token_ids: tuple[str, ...]
    split_role: str

    def as_pharaoh_line(self) -> str:
        """fast_align joint line: ``src tokens ||| trg tokens``."""
        return f"{' '.join(self.source_tokens)} ||| {' '.join(self.target_tokens)}"


@dataclass(frozen=True)
class DirectionalAlignment:
    """0-based Pharaoh links for one sentence in one direction."""

    unit_id: str
    pairs: tuple[tuple[int, int], ...]


@dataclass
class AlignmentLinks:
    """Bidirectional + optional symmetrized links for a bitext batch."""

    aligner: AlignerName
    sentences: list[BitextSentence]
    forward: list[DirectionalAlignment]
    reverse: list[DirectionalAlignment]
    symmetrized: list[DirectionalAlignment] = field(default_factory=list)
    symmetrization: str = "grow-diag-final-and"
    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "aligner": self.aligner.value,
            "symmetrization": self.symmetrization,
            "config": self.config,
            "notes": self.notes,
            "sentence_count": len(self.sentences),
            "sentences": [
                {
                    "unit_id": s.unit_id,
                    "split_role": s.split_role,
                    "source_tokens": list(s.source_tokens),
                    "target_tokens": list(s.target_tokens),
                    "source_token_ids": list(s.source_token_ids),
                    "target_token_ids": list(s.target_token_ids),
                }
                for s in self.sentences
            ],
            "forward": [
                {"unit_id": a.unit_id, "pairs": [list(p) for p in a.pairs]}
                for a in self.forward
            ],
            "reverse": [
                {"unit_id": a.unit_id, "pairs": [list(p) for p in a.pairs]}
                for a in self.reverse
            ],
            "symmetrized": [
                {"unit_id": a.unit_id, "pairs": [list(p) for p in a.pairs]}
                for a in self.symmetrized
            ],
        }


@dataclass(frozen=True)
class AlignerConfig:
    """Reproducibility knobs shared across adapters."""

    seed: int = 36
    eflomal_model: int = 3
    eflomal_n_samplers: int = 1
    eflomal_null_prior: float = 0.2
    fast_align_iterations: int = 5
    fast_align_favor_diagonal: bool = True
    fast_align_optimize_diagonal: bool = True
    fast_align_variational_bayes: bool = True
    symmetrization: str = "grow-diag-final-and"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

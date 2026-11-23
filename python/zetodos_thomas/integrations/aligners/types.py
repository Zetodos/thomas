# SPDX-License-Identifier: Apache-2.0
"""Shared types for statistical word-aligner adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AlignerName(StrEnum):
    """Supported local statistical aligners."""

    EFLOMAL = "eflomal"


@dataclass(frozen=True)
class BitextSentence:
    """One parallel sentence with aligned tokens and unit provenance."""

    unit_id: str
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    source_token_ids: tuple[str, ...] = ()
    target_token_ids: tuple[str, ...] = ()
    split_role: str = "train"
    domain: str = "nt"


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


@dataclass(frozen=True)
class AlignerConfig:
    """Reproducibility knobs for eflomal."""

    seed: int = 36
    eflomal_model: int = 3
    eflomal_n_samplers: int = 1
    eflomal_null_prior: float = 0.2
    symmetrization: str = "grow-diag-final-and"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

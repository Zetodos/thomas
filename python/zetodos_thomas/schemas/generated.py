# SPDX-License-Identifier: Apache-2.0
"""Schemas for deterministic Category C Koine Greek reconstructions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from zetodos_thomas.schemas.linguistic import GreekMorphFeatures


class TokenAlternative(BaseModel):
    """Ranked candidate alternative for one reconstructed Greek token slot."""

    surface: str
    lemma: str
    morph_sig: str
    score: float
    provenance: str


class GeneratedToken(BaseModel):
    """One reconstructed Koine Greek token with Coptic alignment & provenance."""

    seq: int = Field(ge=0)
    surface: str
    lemma: str
    morph: GreekMorphFeatures
    coptic_seqs: list[int] = Field(default_factory=list)
    provenance: str = "C"  # L=loanword, N=NT, P=OT/LXX, C=grammatical transfer
    confidence: float = 0.0
    alternatives: list[TokenAlternative] = Field(default_factory=list)


class GeneratedUnit(BaseModel):
    """Category C Koine Greek reconstruction for one Coptic verse or logion."""

    unit_id: str
    coptic_source_id: str
    coptic_text: str
    greek_text: str
    tokens: list[GeneratedToken] = Field(default_factory=list)
    clause_alternatives: list[str] = Field(default_factory=list)


class GeneratedCorpus(BaseModel):
    """Complete deterministic reconstruction collection (e.g. Thomas 0-114)."""

    id: str
    title: str
    model_id: str
    schema_version: str = "1"
    units: list[GeneratedUnit] = Field(default_factory=list)

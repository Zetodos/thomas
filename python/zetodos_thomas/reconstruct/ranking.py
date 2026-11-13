# SPDX-License-Identifier: Apache-2.0
"""Stable ranking / tie-break rules for the JAK-39 decoder."""

from __future__ import annotations

from dataclasses import dataclass

from zetodos_thomas.schemas.decode import DecoderEvidenceKind

RANKING_RULES_ID = "jak39-stable-ranking-v1"

# Lower tier number = higher priority.
EVIDENCE_TIER: dict[DecoderEvidenceKind, int] = {
    DecoderEvidenceKind.STATISTICAL_CONSTRUCTION: 0,
    DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD: 1,
    DecoderEvidenceKind.LEXICAL_OBSERVED_WORDFORM: 2,
}

RANKING_RULES_DOC = """
JAK-39 stable ranking (identical inputs → identical outputs):

1. Evidence tier (ascending): statistical_construction <
   independently_verified_loanword < lexical_observed_wordform.
2. Dual-aligner agreement preferred (both eflomal + fast_align beats one).
3. Higher observation_count preferred.
4. Higher heuristic_ranking_score preferred when present (else treated as 0).
5. Shorter greek_surfaces sequences preferred (exact attested spans).
6. Lexicographic greek surfaces (NFC, joined by U+001F).
7. Lexicographic evidence ref_id.

Scores reported on candidates are deterministic transforms of the ranking key
(not calibrated probabilities). Uncertainty is derived from tier + aligner
agreement + observation support — never invented attestation counts.
""".strip()


@dataclass(frozen=True, slots=True)
class RankingFeatures:
    """Features used only for deterministic ordering / scoring."""

    kind: DecoderEvidenceKind
    dual_aligner: bool
    observation_count: int
    heuristic_ranking_score: float
    greek_surfaces: tuple[str, ...]
    ref_id: str


RankingKey = tuple[int, int, int, float, int, str, str]


def ranking_key(features: RankingFeatures) -> RankingKey:
    """Return a sortable key; smaller is better."""
    tier = EVIDENCE_TIER[features.kind]
    dual = 0 if features.dual_aligner else 1
    # Negate counts/scores so larger evidence sorts earlier under ascending key.
    obs = -int(features.observation_count)
    heur = -float(features.heuristic_ranking_score)
    length = len(features.greek_surfaces)
    greek = "\u001f".join(features.greek_surfaces)
    return (tier, dual, obs, heur, length, greek, features.ref_id)


def score_from_key(key: RankingKey) -> float:
    """Map ranking key to a stable descending score for display."""
    tier, dual, neg_obs, neg_heur, _length, _greek, _ref = key
    obs = -neg_obs
    heur = -neg_heur
    # Bounded positive score; construction dual-aligner high-count tops the range.
    return round(1000.0 - 100.0 * tier - 20.0 * dual + obs + heur, 6)


def uncertainty_for(
    *,
    kind: DecoderEvidenceKind,
    dual_aligner: bool,
    observation_count: int,
) -> float:
    """Explicit uncertainty in [0, 1]; higher = less certain."""
    if kind is DecoderEvidenceKind.STATISTICAL_CONSTRUCTION:
        base = 0.25 if dual_aligner else 0.4
        support = min(observation_count, 10) / 20.0
        return round(max(0.05, base - support), 4)
    if kind is DecoderEvidenceKind.INDEPENDENTLY_VERIFIED_LOANWORD:
        # Lemma-level L-clue only — inflection not entailed.
        return 0.55
    # Lexical observed wordform
    base = 0.35 if dual_aligner else 0.55
    support = min(observation_count, 20) / 40.0
    return round(max(0.1, base - support), 4)

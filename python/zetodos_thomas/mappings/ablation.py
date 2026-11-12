# SPDX-License-Identifier: Apache-2.0
"""Ablation packages: NT-aligner-only vs + verified loanwords (JAK-38)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zetodos_thomas.schemas.mappings import AblationCondition, AblationPackage


def default_ablation_paths(repo_root: Path) -> dict[str, Path]:
    """Committed ablation artifact paths."""
    base = repo_root / "data" / "sources" / "mappings"
    return {
        AblationCondition.NT_ALIGNER_ONLY.value: base
        / "ablation-nt-aligner-only-v1.json",
        AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS.value: base
        / "ablation-nt-aligner-plus-loanwords-v1.json",
    }


def build_ablation_packages(
    *,
    lexical_candidates_rel: str = (
        "data/sources/lexicon/coptic-greek-lexical-candidates-v1.json"
    ),
    statistical_phrases_rel: str = (
        "data/sources/mappings/statistical-phrase-constructions-v1.json"
    ),
    verified_loanwords_rel: str = (
        "data/sources/mappings/verified-loanwords-ccl-ddglc-v1.json"
    ),
    train_links: dict[str, str] | None = None,
) -> list[AblationPackage]:
    """Build the two Phase-2 ablation input packages."""
    links = train_links or {
        "eflomal": "data/sources/lexicon/eflomal-train-links-v1.json",
        "fast_align": "data/sources/lexicon/fast_align-train-links-v1.json",
    }
    shared_prov: dict[str, Any] = {
        "owned_by": "JAK-38",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase2_active_aligners": ["eflomal", "fast_align"],
        "adopted_aligner": None,
        "decision": (
            "Run both arms: NT-aligner-only vs NT-aligner + independently "
            "verified CCL/DDGLC loanwords."
        ),
    }
    shared_excludes = {
        "thomas_7_greek": True,
        "poxy_654": True,
        "holdout_dev_omission_as_training": True,
        "dioskoros_scrape": True,
        "statistical_pairs_as_verified_loanwords": True,
    }
    aligner_only = AblationPackage(
        id="ablation-nt-aligner-only-v1",
        title="Ablation arm A: NT aligner statistical stack only (JAK-38)",
        condition=AblationCondition.NT_ALIGNER_ONLY,
        includes={
            "lexical_candidates": lexical_candidates_rel,
            "statistical_phrase_constructions": statistical_phrases_rel,
            "train_link_artifacts": links,
            "verified_loanwords": None,
        },
        excludes={
            **shared_excludes,
            "verified_loanword_constraints": True,
        },
        provenance=shared_prov,
        notes=[
            "Arm A uses only statistical NT-aligner observations "
            "(lexical candidates + phrase/construction mappings).",
            "Verified CCL/DDGLC loanwords are intentionally omitted.",
            "Keep both eflomal and fast_align active (adopted_aligner=null).",
            "Consumer: JAK-39 minimal decoder / Phase 2 feasibility report.",
        ],
    )
    plus_loan = AblationPackage(
        id="ablation-nt-aligner-plus-loanwords-v1",
        title=(
            "Ablation arm B: NT aligner statistical stack + verified "
            "loanwords (JAK-38)"
        ),
        condition=AblationCondition.NT_ALIGNER_PLUS_VERIFIED_LOANWORDS,
        includes={
            "lexical_candidates": lexical_candidates_rel,
            "statistical_phrase_constructions": statistical_phrases_rel,
            "train_link_artifacts": links,
            "verified_loanwords": verified_loanwords_rel,
        },
        excludes={
            **shared_excludes,
            "verified_loanword_constraints": False,
        },
        provenance=shared_prov,
        notes=[
            "Arm B adds independently verified CCL/DDGLC loanwords as "
            "separate L-evidence / constraints.",
            "Do not reclassify statistical rows as verified.",
            "Keep both eflomal and fast_align active (adopted_aligner=null).",
            "Consumer: JAK-39 minimal decoder / Phase 2 feasibility report.",
        ],
    )
    return [aligner_only, plus_loan]


def write_ablation_packages(
    packages: list[AblationPackage],
    paths: dict[str, Path],
    *,
    compact: bool = False,
) -> dict[str, Path]:
    """Write ablation JSON artifacts; return written paths by condition."""
    written: dict[str, Path] = {}
    for package in packages:
        path = paths[package.condition.value]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = package.model_dump(mode="json")
        text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
        path.write_text(text + "\n", encoding="utf-8")
        written[package.condition.value] = path
    return written

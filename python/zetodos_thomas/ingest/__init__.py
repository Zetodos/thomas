# SPDX-License-Identifier: Apache-2.0
"""Source ingestion: manifests, CopSahHorner, MorphGNT, OT/Swete, Dilley, fragments."""

from zetodos_thomas.ingest.dilley_thomas import (
    load_thomas_corpus,
    parse_tt,
    write_thomas_corpus,
)
from zetodos_thomas.ingest.evidence_pilot import (
    EvidencePilotValidationError,
    load_evidence_pilot,
    validate_evidence_pilot_or_raise,
    write_evidence_pilot,
)
from zetodos_thomas.ingest.fragments import (
    FragmentValidationError,
    build_fragment_corpus,
    load_fragment_corpus,
    validate_fragment_corpus,
    write_fragment_corpus,
)
from zetodos_thomas.ingest.holdouts import (
    HoldoutValidationError,
    assert_no_train_holdout_greek_ngram_leakage,
    cross_split_greek_ngram_leaks,
    crosscheck_against_units,
    load_fragment_divergences,
    load_nt_split,
    load_omission_fixtures,
    validate_holdout_artifacts,
)
from zetodos_thomas.ingest.manifests import (
    ManifestValidationError,
    load_all_manifests,
    load_manifest,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.morphgnt import (
    build_nt_split_v1,
    load_morphgnt_nt,
    pair_horner_morphgnt,
)
from zetodos_thomas.ingest.ot_swete import (
    load_ot_swete_pairs,
    load_sahidic_ot_chapters,
)

__all__ = [
    "EvidencePilotValidationError",
    "FragmentValidationError",
    "HoldoutValidationError",
    "ManifestValidationError",
    "assert_no_train_holdout_greek_ngram_leakage",
    "build_fragment_corpus",
    "build_nt_split_v1",
    "cross_split_greek_ngram_leaks",
    "crosscheck_against_units",
    "load_all_manifests",
    "load_evidence_pilot",
    "load_fragment_corpus",
    "load_fragment_divergences",
    "load_manifest",
    "load_morphgnt_nt",
    "load_nt_split",
    "load_omission_fixtures",
    "load_ot_swete_pairs",
    "load_sahidic_ot_chapters",
    "load_thomas_corpus",
    "pair_horner_morphgnt",
    "parse_tt",
    "validate_evidence_pilot_or_raise",
    "validate_fragment_corpus",
    "validate_holdout_artifacts",
    "validate_manifests_dir",
    "write_evidence_pilot",
    "write_fragment_corpus",
    "write_thomas_corpus",
]

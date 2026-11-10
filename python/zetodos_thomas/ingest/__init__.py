# SPDX-License-Identifier: Apache-2.0
"""Source ingestion: manifests, CopSahHorner, Vaticanus B*, Dilley, fragments."""

from zetodos_thomas.ingest.dilley_thomas import (
    load_thomas_corpus,
    parse_tt,
    write_thomas_corpus,
)
from zetodos_thomas.ingest.fragments import (
    FragmentValidationError,
    build_fragment_corpus,
    load_fragment_corpus,
    validate_fragment_corpus,
    write_fragment_corpus,
)
from zetodos_thomas.ingest.manifests import (
    ManifestValidationError,
    load_all_manifests,
    load_manifest,
    validate_manifests_dir,
)
from zetodos_thomas.ingest.pilot import (
    build_pilot_corpus,
    load_pilot_corpus,
    write_pilot_corpus,
)

__all__ = [
    "FragmentValidationError",
    "ManifestValidationError",
    "build_fragment_corpus",
    "build_pilot_corpus",
    "load_all_manifests",
    "load_fragment_corpus",
    "load_manifest",
    "load_pilot_corpus",
    "load_thomas_corpus",
    "parse_tt",
    "validate_fragment_corpus",
    "validate_manifests_dir",
    "write_fragment_corpus",
    "write_pilot_corpus",
    "write_thomas_corpus",
]

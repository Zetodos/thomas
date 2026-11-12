# SPDX-License-Identifier: Apache-2.0
"""Phrase/construction mappings and verified loanwords (JAK-38)."""

from zetodos_thomas.mappings.ablation import (
    build_ablation_packages,
    default_ablation_paths,
    write_ablation_packages,
)
from zetodos_thomas.mappings.loanwords import (
    default_verified_loanwords_path,
    load_verified_loanword_table,
    verified_loanword_table,
    write_verified_loanword_table,
)
from zetodos_thomas.mappings.phrases import (
    default_mappings_dir,
    default_statistical_constructions_path,
    generate_statistical_constructions,
    load_statistical_construction_table,
    write_statistical_construction_table,
)
from zetodos_thomas.mappings.validate import (
    MappingValidationReport,
    validate_ablation_package,
    validate_statistical_construction_table,
    validate_verified_loanword_table,
)

__all__ = [
    "MappingValidationReport",
    "build_ablation_packages",
    "default_ablation_paths",
    "default_mappings_dir",
    "default_statistical_constructions_path",
    "default_verified_loanwords_path",
    "generate_statistical_constructions",
    "load_statistical_construction_table",
    "load_verified_loanword_table",
    "validate_ablation_package",
    "validate_statistical_construction_table",
    "validate_verified_loanword_table",
    "verified_loanword_table",
    "write_ablation_packages",
    "write_statistical_construction_table",
    "write_verified_loanword_table",
]

# SPDX-License-Identifier: Apache-2.0
"""Lexicon / mapping helpers (Phase 2 lexical candidates — JAK-37)."""

from zetodos_thomas.lexicon.candidates import (
    build_lexical_candidate_table,
    generate_lexical_candidates,
    load_lexical_candidate_table,
    write_lexical_candidate_table,
)
from zetodos_thomas.lexicon.validate import (
    LexicalValidationReport,
    validate_lexical_candidate_path,
    validate_lexical_candidate_table,
)

__all__ = [
    "LexicalValidationReport",
    "build_lexical_candidate_table",
    "generate_lexical_candidates",
    "load_lexical_candidate_table",
    "validate_lexical_candidate_path",
    "validate_lexical_candidate_table",
    "write_lexical_candidate_table",
]

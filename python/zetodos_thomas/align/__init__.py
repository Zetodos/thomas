# SPDX-License-Identifier: Apache-2.0
"""Coptic-Greek lemma+POS alignment modules."""

from zetodos_thomas.align.lemma_aligner import (
    LexicalCandidate,
    LexicalTranslationModel,
    filter_coptic_for_alignment,
)

__all__ = [
    "LexicalCandidate",
    "LexicalTranslationModel",
    "filter_coptic_for_alignment",
]

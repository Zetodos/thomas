# SPDX-License-Identifier: Apache-2.0
"""Lexicon, DDGLC loanwords, and Coptic multiword idioms."""

from zetodos_thomas.lexicon.ccl_lexicon import CclLexicon, LoanwordEntry
from zetodos_thomas.lexicon.idioms import CopticIdiomRule, IdiomAndValencyTable

__all__ = [
    "CclLexicon",
    "CopticIdiomRule",
    "IdiomAndValencyTable",
    "LoanwordEntry",
]

# SPDX-License-Identifier: Apache-2.0
"""Alignment utilities: tokenization helpers and schema-backed validation."""

from zetodos_thomas.align.tokenize import (
    TokenCountMismatchError,
    whitespace_token_counts,
    whitespace_tokens,
)
from zetodos_thomas.schemas.alignment import (
    AgreementDegree,
    AlignmentDocument,
    Parallel,
    ParallelRelation,
    Span,
    SpanKind,
    ThomasCorpus,
    Token,
)
from zetodos_thomas.schemas.ids import (
    ID_SCHEME_VERSION,
    parallel_id,
    span_id,
    token_id,
)

__all__ = [
    "ID_SCHEME_VERSION",
    "AgreementDegree",
    "AlignmentDocument",
    "Parallel",
    "ParallelRelation",
    "Span",
    "SpanKind",
    "ThomasCorpus",
    "Token",
    "TokenCountMismatchError",
    "parallel_id",
    "span_id",
    "token_id",
    "whitespace_token_counts",
    "whitespace_tokens",
]

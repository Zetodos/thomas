# SPDX-License-Identifier: Apache-2.0
"""Typed third-party adapters (aligners, optional offline tooling)."""

from zetodos_thomas.integrations.aligners import (
    AlignerAvailability,
    AlignerName,
    AlignmentLinks,
    discover_aligners,
)

__all__ = [
    "AlignerAvailability",
    "AlignerName",
    "AlignmentLinks",
    "discover_aligners",
]

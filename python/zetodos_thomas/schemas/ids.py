# SPDX-License-Identifier: Apache-2.0
"""Stable identifier helpers and id-scheme versioning constants.

IDs are opaque strings once minted. Presentation changes (font, HTML markup,
display orthography) must not rewrite them. Retokenization that changes token
boundaries **does** require a scheme bump — see ``docs/id-scheme.md``.
"""

from __future__ import annotations

# Bump when token boundaries or ID grammar change. v1 migration rule:
# rewrite all token/span/parallel IDs and alignment JSON; do not attempt
# in-place salvage of old IDs (documented in docs/id-scheme.md).
ID_SCHEME_VERSION = "1"

_SCHEME_PREFIX = f"v{ID_SCHEME_VERSION}:"


def token_id(source_id: str, unit_id: str, sequence: int) -> str:
    """Mint a stable token id for the current scheme version."""
    if sequence < 0:
        msg = "token sequence must be >= 0"
        raise ValueError(msg)
    return f"{_SCHEME_PREFIX}{source_id}/{unit_id}/t/{sequence}"


def span_id(source_id: str, unit_id: str, local_id: str) -> str:
    """Mint a stable span id (local_id is a short slug within the unit)."""
    local = local_id.strip()
    if not local or "/" in local or " " in local:
        msg = "span local_id must be a non-empty slug without spaces or '/'"
        raise ValueError(msg)
    return f"{_SCHEME_PREFIX}{source_id}/{unit_id}/s/{local}"


def parallel_id(local_id: str) -> str:
    """Mint a stable parallel / alignment-record id."""
    local = local_id.strip()
    if not local or "/" in local or " " in local:
        msg = "parallel local_id must be a non-empty slug without spaces or '/'"
        raise ValueError(msg)
    return f"{_SCHEME_PREFIX}parallel/{local}"


def scheme_version_of(stable_id: str) -> str | None:
    """Return the ``vN`` scheme label embedded in an id, if present."""
    if not stable_id.startswith("v") or ":" not in stable_id:
        return None
    head, _sep, _rest = stable_id.partition(":")
    return head[1:] if head.startswith("v") and head[1:].isdigit() else None

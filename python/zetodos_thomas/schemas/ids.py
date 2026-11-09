# SPDX-License-Identifier: Apache-2.0
"""Stable identifier helpers and id-scheme versioning constants.

IDs are opaque strings once minted. Presentation changes (font, HTML markup,
display orthography) must not rewrite them. Retokenization that changes token
boundaries **does** require a scheme bump — see ``docs/id-scheme.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump when token boundaries or ID grammar change. v1 migration rule:
# rewrite all token/span/parallel IDs and alignment JSON; do not attempt
# in-place salvage of old IDs (documented in docs/id-scheme.md).
ID_SCHEME_VERSION = "1"

_SCHEME_PREFIX = f"v{ID_SCHEME_VERSION}:"


@dataclass(frozen=True, slots=True)
class ParsedTokenId:
    """Parts of a minted token id."""

    scheme_version: str
    source_id: str
    unit_id: str
    sequence: int


@dataclass(frozen=True, slots=True)
class ParsedSpanId:
    """Parts of a minted span id."""

    scheme_version: str
    source_id: str
    unit_id: str
    local_id: str


@dataclass(frozen=True, slots=True)
class ParsedParallelId:
    """Parts of a minted parallel id."""

    scheme_version: str
    local_id: str


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


def _split_scheme(stable_id: str) -> tuple[str, str]:
    if not stable_id.startswith("v") or ":" not in stable_id:
        msg = f"id missing scheme prefix: {stable_id!r}"
        raise ValueError(msg)
    head, _sep, rest = stable_id.partition(":")
    if not head.startswith("v") or not head[1:].isdigit() or not rest:
        msg = f"id missing scheme prefix: {stable_id!r}"
        raise ValueError(msg)
    return head[1:], rest


def parse_token_id(stable_id: str) -> ParsedTokenId:
    """Parse ``vN:{source}/{unit}/t/{sequence}``."""
    version, rest = _split_scheme(stable_id)
    if "/t/" not in rest:
        msg = f"not a token id: {stable_id!r}"
        raise ValueError(msg)
    head, _sep, seq_text = rest.rpartition("/t/")
    if "/" not in head:
        msg = f"not a token id: {stable_id!r}"
        raise ValueError(msg)
    source_id, _sep, unit_id = head.partition("/")
    if not source_id or not unit_id or not seq_text.isdigit():
        msg = f"not a token id: {stable_id!r}"
        raise ValueError(msg)
    return ParsedTokenId(
        scheme_version=version,
        source_id=source_id,
        unit_id=unit_id,
        sequence=int(seq_text),
    )


def parse_span_id(stable_id: str) -> ParsedSpanId:
    """Parse ``vN:{source}/{unit}/s/{local}``."""
    version, rest = _split_scheme(stable_id)
    if "/s/" not in rest:
        msg = f"not a span id: {stable_id!r}"
        raise ValueError(msg)
    head, _sep, local = rest.rpartition("/s/")
    if "/" not in head or not local or "/" in local or " " in local:
        msg = f"not a span id: {stable_id!r}"
        raise ValueError(msg)
    source_id, _sep, unit_id = head.partition("/")
    if not source_id or not unit_id:
        msg = f"not a span id: {stable_id!r}"
        raise ValueError(msg)
    return ParsedSpanId(
        scheme_version=version,
        source_id=source_id,
        unit_id=unit_id,
        local_id=local,
    )


def parse_parallel_id(stable_id: str) -> ParsedParallelId:
    """Parse ``vN:parallel/{local}``."""
    version, rest = _split_scheme(stable_id)
    prefix = "parallel/"
    if not rest.startswith(prefix):
        msg = f"not a parallel id: {stable_id!r}"
        raise ValueError(msg)
    local = rest[len(prefix) :]
    if not local or "/" in local or " " in local:
        msg = f"not a parallel id: {stable_id!r}"
        raise ValueError(msg)
    return ParsedParallelId(scheme_version=version, local_id=local)

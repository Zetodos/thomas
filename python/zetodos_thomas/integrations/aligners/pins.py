# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream commit SHAs for statistical aligner builds (JAK-36)."""

from __future__ import annotations

# Immutable pins — update deliberately and regenerate eval artifacts.
EFLOMAL_GIT_URL = "https://github.com/robertostling/eflomal.git"
EFLOMAL_COMMIT_SHA = "1fe2a43e3667fb2461736ddb48783ab0cf98a171"

FAST_ALIGN_GIT_URL = "https://github.com/clab/fast_align.git"
FAST_ALIGN_COMMIT_SHA = "cab1e9aac8d3bb02ff5ae58218d8d225a039fa11"

PINNED_ALIGNER_VERSIONS: dict[str, dict[str, str]] = {
    "eflomal": {
        "git_url": EFLOMAL_GIT_URL,
        "commit_sha": EFLOMAL_COMMIT_SHA,
    },
    "fast_align": {
        "git_url": FAST_ALIGN_GIT_URL,
        "commit_sha": FAST_ALIGN_COMMIT_SHA,
    },
}


def pinned_versions_payload() -> dict[str, dict[str, str]]:
    """JSON-serializable pin table for reports / artifacts."""
    return {name: dict(meta) for name, meta in PINNED_ALIGNER_VERSIONS.items()}

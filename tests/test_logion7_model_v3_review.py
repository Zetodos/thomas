# SPDX-License-Identifier: Apache-2.0
"""The Logion 7 review page must not call pointer gaps abstentions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs" / "logion7-model-v3-review.md"


def test_logion7_review_page_exists_and_does_not_call_gaps_abstentions() -> None:
    assert REVIEW.is_file()
    page = REVIEW.read_text(encoding="utf-8")
    assert "thomas.7" in page
    assert "model-v3" in page
    assert "abstain" not in page.lower()

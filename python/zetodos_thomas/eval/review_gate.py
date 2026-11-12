# SPDX-License-Identifier: Apache-2.0
"""Gate JAK-40/41 scoring on independently reviewed evaluation packages.

JAK-43 ships the Thomas 7 blinded pack at ``ready-for-independent-review``.
That status means the scoring target and rubric are packaged, sealed, and
ready for a human reviewer — not that the review has happened.

Independent review (reviewer, date, inspected witness/source version, and
surviving-letter-vs-restoration scoring decisions) is a **blocking follow-up**
before JAK-40/41 may score against this reference. Do not relabel the pack
``human-checked`` without a real review.

Call ``assert_eval_package_human_checked`` (or
``require_human_checked_review_status``) at the start of any JAK-40/41 scoring
entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HUMAN_CHECKED = "human-checked"
READY_FOR_INDEPENDENT_REVIEW = "ready-for-independent-review"
ALLOWED_REVIEW_STATUSES = frozenset(
    {
        "provisional-draft",
        READY_FOR_INDEPENDENT_REVIEW,
        HUMAN_CHECKED,
        "sealed-holdout",
    }
)

DEFAULT_BLINDED_THOMAS7_REL = Path("data/evidence/eval/thomas-7-blinded-v1.json")


class EvalReviewGateError(ValueError):
    """Raised when scoring is attempted against an unreviewed eval package."""


def load_blinded_eval_package(path: Path) -> dict[str, Any]:
    """Load a blinded evaluation package JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"eval package must be a JSON object: {path}"
        raise EvalReviewGateError(msg)
    return payload


def require_human_checked_review_status(review_status: str, *, package_id: str) -> None:
    """Refuse scoring unless ``review_status`` is ``human-checked``."""
    if review_status == HUMAN_CHECKED:
        return
    if review_status not in ALLOWED_REVIEW_STATUSES:
        msg = f"unknown review_status {review_status!r} on eval package {package_id!r}"
        raise EvalReviewGateError(msg)
    msg = (
        f"eval package {package_id!r} has review_status={review_status!r}; "
        "JAK-40/41 scoring requires review_status='human-checked' after an "
        "independent review (reviewer, date, inspected witness/source version, "
        "surviving-letter-vs-restoration decisions). "
        f"{READY_FOR_INDEPENDENT_REVIEW!r} is packaged readiness only — not a "
        "completed review. Formal follow-up: complete independent review before "
        "scoring (tracked under JAK-43 eval-readiness; do not invent human-checked)."
    )
    raise EvalReviewGateError(msg)


def assert_eval_package_human_checked(path: Path) -> dict[str, Any]:
    """Load ``path`` and gate on ``human-checked`` review_status.

    Returns the package payload when the gate passes.
    """
    payload = load_blinded_eval_package(path)
    package_id = str(payload.get("id", path.name))
    status = payload.get("review_status")
    if not isinstance(status, str) or not status:
        msg = f"eval package {package_id!r} missing review_status"
        raise EvalReviewGateError(msg)
    require_human_checked_review_status(status, package_id=package_id)
    return payload

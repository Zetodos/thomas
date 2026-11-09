# SPDX-License-Identifier: Apache-2.0
"""Optional eflomal availability probe (PLAN.md Phase 2 consideration).

eflomal is **not** on the deterministic default path. The package on PyPI
requires a native ``make`` build of the C binary; this environment could not
install it. JAK-28 therefore ships the lexical Dice baseline and records the
eflomal outcome for Jake review rather than inventing scores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EflomalProbeResult:
    """Whether eflomal can be imported/used in this environment."""

    available: bool
    detail: str
    selected_for_default_path: bool = False


def probe_eflomal() -> EflomalProbeResult:
    """Attempt a soft import of eflomal; never raise into the CLI path."""
    try:
        import importlib

        importlib.import_module("eflomal")
    except Exception as exc:
        return EflomalProbeResult(
            available=False,
            detail=(
                f"eflomal import failed ({type(exc).__name__}: {exc}). "
                "PyPI eflomal 2.0.0 requires compiling src/ via make; no wheel "
                "was available here. Lexical Dice baseline remains the "
                "deterministic offline path. Jake decision: whether to add a "
                "native eflomal toolchain pin later (still no LLM)."
            ),
            selected_for_default_path=False,
        )
    return EflomalProbeResult(
        available=True,
        detail=(
            "eflomal imported successfully but is not wired into the default "
            "deterministic path for this pilot; compare offline before adopting."
        ),
        selected_for_default_path=False,
    )

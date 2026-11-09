# SPDX-License-Identifier: Apache-2.0
"""Ensure project-owned sources declare an SPDX license identifier."""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "SPDX-License-Identifier:"
HEAD_LINES = 12


def main(argv: list[str]) -> int:
    if not argv:
        return 0

    missing: list[str] = []
    for raw in argv:
        path = Path(raw)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error reading {path}: {exc}", file=sys.stderr)
            return 2
        head = "\n".join(text.splitlines()[:HEAD_LINES])
        if MARKER not in head:
            missing.append(raw)

    if missing:
        print("Missing SPDX-License-Identifier header in:")
        for path in missing:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

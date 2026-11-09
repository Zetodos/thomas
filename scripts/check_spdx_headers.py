# SPDX-License-Identifier: Apache-2.0
"""Ensure project-owned sources declare an SPDX license identifier."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEAD_LINES = 12
# Comment-header form only (e.g. ``# SPDX-License-Identifier: Apache-2.0``).
# Rejects empty identifiers and non-comment occurrences of the marker text.
SPDX_HEADER_RE = re.compile(
    r"^#\s*SPDX-License-Identifier:\s*(?P<id>\S+)\s*$",
)


def has_spdx_header(text: str) -> bool:
    for line in text.splitlines()[:HEAD_LINES]:
        match = SPDX_HEADER_RE.match(line)
        if match is not None and match.group("id"):
            return True
    return False


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
        if not has_spdx_header(text):
            missing.append(raw)

    if missing:
        print("Missing SPDX-License-Identifier comment header in:")
        for path in missing:
            print(f"  {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

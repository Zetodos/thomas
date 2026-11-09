# SPDX-License-Identifier: Apache-2.0
"""Download CopSahHorner + WH textonly into data/downloads/.

Prefer the Typer entrypoint:

    uv run thomas sources download

This script is a thin alias for the same logic:

    uv run python scripts/download_pilot_sources.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zetodos_thomas.ingest.download import DownloadError, download_pilot_sources
from zetodos_thomas.ingest.paths import find_repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when checksum-verified files already exist",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Checkout root (default: auto-detect from cwd)",
    )
    args = parser.parse_args(argv)
    try:
        root = find_repo_root(args.repo_root)
        sword_root, wh_dir = download_pilot_sources(root, force=args.force)
    except (DownloadError, FileNotFoundError, OSError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    print(f"OK: CopSahHorner SWORD tree at {sword_root}")
    print(f"OK: Westcott-Hort textonly at {wh_dir}")
    print("Next (optional): uv run thomas sources build-pilot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

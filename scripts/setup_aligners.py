# SPDX-License-Identifier: Apache-2.0
"""Build eflomal + fast_align into data/local/aligners (explicit setup).

Not invoked by offline reconstruction. Requires network once to clone sources,
plus a C/C++ toolchain (gcc/g++, make, cmake, libomp, python3-dev).

Upstream commits are pinned in ``zetodos_thomas.integrations.aligners.pins``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from zetodos_thomas.integrations.aligners.pins import (
    EFLOMAL_COMMIT_SHA,
    EFLOMAL_GIT_URL,
    FAST_ALIGN_COMMIT_SHA,
    FAST_ALIGN_GIT_URL,
    pinned_versions_payload,
)


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clone_pinned(url: str, sha: str, dest: Path) -> None:
    """Fetch a single pinned commit into ``dest`` and verify HEAD == sha."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _run(["git", "init"], cwd=dest)
    _run(["git", "remote", "add", "origin", url], cwd=dest)
    _run(["git", "fetch", "--depth", "1", "origin", sha], cwd=dest)
    _run(["git", "checkout", "--force", "FETCH_HEAD"], cwd=dest)
    got = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=dest, text=True
    ).strip()
    if got != sha:
        msg = f"pin mismatch for {url}: expected {sha}, got {got}"
        raise RuntimeError(msg)
    print(f"pinned {url} @ {got}", flush=True)


def setup_fast_align(dest: Path, work: Path) -> Path:
    src = work / "fast_align"
    _clone_pinned(FAST_ALIGN_GIT_URL, FAST_ALIGN_COMMIT_SHA, src)
    build = src / "build"
    build.mkdir(exist_ok=True)
    _run(["cmake", ".."], cwd=build)
    _run(["make", f"-j{os.cpu_count() or 2}"], cwd=build)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("fast_align", "atools"):
        shutil.copy2(build / name, dest / name)
        (dest / name).chmod(0o755)
    return dest / "fast_align"


def setup_eflomal(dest: Path, work: Path, *, python: str) -> Path:
    src = work / "eflomal"
    _clone_pinned(EFLOMAL_GIT_URL, EFLOMAL_COMMIT_SHA, src)
    _run(["make", "-C", "src", "clean"], cwd=src)
    env = os.environ.copy()
    env.setdefault("CC", "gcc")
    _run(["make", "-C", "src", "eflomal"], cwd=src, env=env)
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / "eflomal"
    shutil.copy2(src / "src" / "eflomal", binary)
    binary.chmod(0o755)
    # Install Python package into the selected interpreter (uv first; pip fallback).
    uv_bin = shutil.which("uv")
    if uv_bin is not None:
        _run([uv_bin, "pip", "install", "--python", python, str(src)])
    else:
        _run([python, "-m", "pip", "install", "--quiet", str(src)])
    return binary


def _write_versions(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "versions.json"
    path.write_text(
        json.dumps(pinned_versions_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Checkout root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--skip-eflomal",
        action="store_true",
        help="Only build fast_align",
    )
    parser.add_argument(
        "--skip-fast-align",
        action="store_true",
        help="Only build eflomal",
    )
    args = parser.parse_args(argv)
    root = (args.repo_root or _repo_root()).resolve()
    dest = root / "data" / "local" / "aligners"
    python = sys.executable
    with tempfile.TemporaryDirectory(prefix="thomas-aligner-setup-") as tmp:
        work = Path(tmp)
        if not args.skip_fast_align:
            path = setup_fast_align(dest, work)
            print(f"fast_align -> {path}")
        if not args.skip_eflomal:
            path = setup_eflomal(dest, work, python=python)
            print(f"eflomal -> {path}")
    versions = _write_versions(dest)
    print(f"wrote pin manifest {versions}")
    print(f"Aligner binaries in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

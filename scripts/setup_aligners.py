# SPDX-License-Identifier: Apache-2.0
"""Build eflomal + fast_align into data/local/aligners (explicit setup).

Not invoked by offline reconstruction. Requires network once to clone sources,
plus a C/C++ toolchain (gcc/g++, make, cmake, libomp, python3-dev).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_fast_align(dest: Path, work: Path) -> Path:
    src = work / "fast_align"
    if not src.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/clab/fast_align.git",
                str(src),
            ]
        )
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
    if not src.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/robertostling/eflomal.git",
                str(src),
            ]
        )
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
    print(f"Aligner binaries in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""Build eflomal into data/local/aligners and install its Python wrapper.

Not invoked by offline reconstruction. Requires network once to clone the pinned
commit, plus a C compiler (gcc/clang + make; uses Homebrew libomp on macOS when
present).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from zetodos_thomas.integrations.aligners.pins import (
    EFLOMAL_COMMIT_SHA,
    EFLOMAL_GIT_URL,
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


def _macos_omp_env(base_env: Mapping[str, str]) -> dict[str, str]:
    env = dict(base_env)
    if platform.system() != "Darwin":
        env.setdefault("CC", "gcc")
        return env
    # Apple clang supports OpenMP only via -Xpreprocessor -fopenmp + libomp.
    libomp_prefixes = (
        Path("/opt/homebrew/opt/libomp"),
        Path("/usr/local/opt/libomp"),
    )
    for prefix in libomp_prefixes:
        if (prefix / "include" / "omp.h").is_file():
            env["CC"] = "clang"
            env["CFLAGS"] = (
                f"{env.get('CFLAGS', '')} -O3 -Xpreprocessor -fopenmp "
                f"-I{prefix / 'include'}"
            ).strip()
            env["LDFLAGS"] = (
                f"{env.get('LDFLAGS', '')} -L{prefix / 'lib'} -lomp"
            ).strip()
            return env
    env.setdefault("CC", "clang")
    return env


def setup_eflomal(dest: Path, work: Path, *, python: str) -> Path:
    src = work / "eflomal"
    _clone_pinned(EFLOMAL_GIT_URL, EFLOMAL_COMMIT_SHA, src)
    _run(["make", "-C", "src", "clean"], cwd=src)
    env = _macos_omp_env(os.environ)
    # Build using Makefile; on macOS override -fopenmp if needed.
    makefile = src / "src" / "Makefile"
    if platform.system() == "Darwin" and makefile.is_file():
        text = makefile.read_text(encoding="utf-8")
        if "-fopenmp" in text and "-Xpreprocessor -fopenmp" not in text:
            libomp_inc = Path("/opt/homebrew/opt/libomp/include")
            libomp_lib = Path("/opt/homebrew/opt/libomp/lib")
            if libomp_inc.is_dir() and libomp_lib.is_dir():
                text = text.replace(
                    "-fopenmp",
                    f"-Xpreprocessor -fopenmp -I{libomp_inc}",
                )
                text = text.replace("-lgomp", f"-L{libomp_lib} -lomp")
                if "-lomp" not in text:
                    text += f"\nLDFLAGS += -L{libomp_lib} -lomp\n"
                makefile.write_text(text, encoding="utf-8")
            else:
                text = text.replace("-fopenmp", "")
                makefile.write_text(text, encoding="utf-8")
    _run(["make", "-C", "src", "eflomal"], cwd=src, env=env)
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / "eflomal"
    shutil.copy2(src / "src" / "eflomal", binary)
    binary.chmod(0o755)
    # Also ensure setup.py / python wrapper finds libomp if compiled with Cython.
    uv_bin = shutil.which("uv")
    if uv_bin is not None:
        _run([uv_bin, "pip", "install", "--python", python, str(src)], env=env)
    else:
        _run([python, "-m", "pip", "install", "--quiet", str(src)], env=env)
    # Make sure the `eflomal` binary directory is on PATH / data/local/aligners.
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
    args = parser.parse_args(argv)
    root = (args.repo_root or _repo_root()).resolve()
    dest = root / "data" / "local" / "aligners"
    python = sys.executable
    with tempfile.TemporaryDirectory(prefix="thomas-aligner-setup-") as tmp:
        work = Path(tmp)
        path = setup_eflomal(dest, work, python=python)
        print(f"eflomal -> {path}")
    versions = _write_versions(dest)
    print(f"wrote pin manifest {versions}")
    print(f"Aligner binaries in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

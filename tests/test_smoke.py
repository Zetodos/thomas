# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the Phase 0 Python package scaffold."""

from typer.testing import CliRunner

from zetodos_thomas import __version__
from zetodos_thomas.cli.app import app
from zetodos_thomas.schemas import PackageInfo


def test_package_info_roundtrip() -> None:
    info = PackageInfo(name="zetodos_thomas", version=__version__)
    assert info.name == "zetodos_thomas"
    assert info.version == "0.1.0"
    assert info.model_dump() == {"name": "zetodos_thomas", "version": "0.1.0"}


def test_thomas_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "thomas" in result.stdout.lower()

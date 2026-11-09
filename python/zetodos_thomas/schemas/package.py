# SPDX-License-Identifier: Apache-2.0
"""Minimal package metadata schema (Phase 0 stub)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PackageInfo(BaseModel):
    """Minimal package metadata used by smoke tests and future exporters."""

    name: str = Field(description="Canonical package name")
    version: str = Field(description="Package version string")

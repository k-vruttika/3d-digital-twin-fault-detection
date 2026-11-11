"""Smoke test: package and all subpackages from CLAUDE.md section 5 import."""

import importlib

import pytest

SUBPACKAGES = [
    "adapters",
    "ingest",
    "bus",
    "storage",
    "engine",
    "commands",
    "inference",
    "whatif",
    "api",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(f"twin_backend.{name}").__doc__

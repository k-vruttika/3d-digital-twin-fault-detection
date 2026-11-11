"""Smoke test: package and all subpackages from CLAUDE.md section 5 import."""

import importlib

import pytest

SUBPACKAGES = [
    "data",
    "features",
    "models",
    "train",
    "calibrate",
    "conformal",
    "explain",
    "evaluate",
    "datasets",
    "experiments",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(f"twin_ml.{name}").__doc__

"""Lint configuration guards: `poe lint` and the pre-commit hook check the same files."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _ruff() -> dict[str, object]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ruff: dict[str, object] = data["tool"]["ruff"]
    return ruff


def test_ruff_excludes_are_anchored_and_forced() -> None:
    """A bare name like "data" would exclude every directory of that name (ml/twin_ml/data/)."""
    ruff = _ruff()
    excludes = ruff.get("extend-exclude", [])
    assert isinstance(excludes, list)
    assert all(str(p).startswith("./") for p in excludes), excludes
    assert ruff.get("force-exclude") is True


def test_ruff_checks_the_package_data_modules() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--show-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.replace("\\", "/")
    assert "ml/twin_ml/data/build.py" in out
    assert "ml/twin_ml/data/perturb.py" in out


def test_pre_commit_runs_exactly_poe_lint() -> None:
    cfg = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [h for repo in cfg["repos"] for h in repo["hooks"]]
    assert len(hooks) == 1
    (hook,) = hooks
    assert hook["entry"] == "uv run --frozen poe lint"
    assert hook["pass_filenames"] is False
    assert hook["always_run"] is True

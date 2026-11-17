"""The root conftest's guard: a pytest session fails if torch and xgboost are both imported in
its process (ADR 0020). Checked in child pytest processes with stand-in modules, so the real
libraries (and their OpenMP runtimes) are never loaded here."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE = (
    "import sys, types\n"
    "def load_both():\n"
    "    for name in ('torch', 'xgboost'):\n"
    "        sys.modules.setdefault(name, types.ModuleType(name))\n"
)


def _run(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    probe = tmp_path / "test_probe.py"
    probe.write_text(FAKE + body, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "conftest",  # the repo's root conftest.py (cwd is on sys.path)
        str(probe),
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(tmp_path),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def test_guard_fails_the_test_after_which_both_are_loaded(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        "def test_one():\n    pass\n\ndef test_loads_both():\n    load_both()\n\n"
        "def test_later():\n    pass\n",
    )
    text = out.stdout + out.stderr
    assert out.returncode == 1, text
    assert "ERROR at teardown of test_loads_both" in text
    assert "0020-ml-models-training-tracking.md" in text
    assert "3 passed, 1 error" in text  # reported once, not after every later test


def test_guard_fails_collection_imports(tmp_path: Path) -> None:
    out = _run(tmp_path, "load_both()\n\ndef test_one():\n    pass\n")
    text = out.stdout + out.stderr
    assert out.returncode != 0, text
    assert "both imported in the pytest process (after test collection)" in text


def test_guard_passes_with_one_library(tmp_path: Path) -> None:
    body = (
        "import types\n"
        "def test_one():\n"
        "    sys.modules.setdefault('xgboost', types.ModuleType('xgboost'))\n"
    )
    out = _run(tmp_path, body)
    assert out.returncode == 0, out.stdout + out.stderr

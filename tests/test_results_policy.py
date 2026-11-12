"""experiments/results/ is committed (CLAUDE.md section 7): reports, summaries and plots only.

Git cannot ignore files by size, so this test checks every file under experiments/results/ that
git would commit (tracked, or untracked and not ignored): it must be a report, summary, small
table or plot, and no larger than MAX_FILE_BYTES. Large or regenerable artefacts belong in the
ignore list in .gitignore (or in data/, which is never committed).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = "experiments/results"
# Repository policy (not a domain threshold): the largest file allowed in experiments/results/.
MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".md", ".json", ".png", ".svg", ".csv", ".txt"}
ALLOWED_NAMES = {".gitkeep"}


def committable_files() -> list[Path]:
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        [git, "ls-files", "--cached", "--others", "--exclude-standard", "--", RESULTS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


def test_results_contain_only_small_reports_and_plots() -> None:
    problems: list[str] = []
    for path in committable_files():
        if not path.is_file():
            continue  # deleted in the working tree
        rel = path.relative_to(REPO_ROOT).as_posix()
        if path.name not in ALLOWED_NAMES and path.suffix.lower() not in ALLOWED_SUFFIXES:
            problems.append(f"{rel}: type not allowed (add it to .gitignore or to the policy)")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(
                f"{rel}: {size / 1e6:.1f} MB > {MAX_FILE_BYTES / 1e6:.1f} MB "
                "(ignore it in .gitignore or keep it under data/)"
            )
    assert not problems, "\n".join(problems)


def test_large_artefacts_are_ignored() -> None:
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    for name in ("run.parquet", "sub/model.pt", "sub/model.onnx", "mlruns/0/meta.yaml"):
        result = subprocess.run(
            [git, "check-ignore", "-q", f"{RESULTS}/{name}"], cwd=REPO_ROOT, check=False
        )
        assert result.returncode == 0, f"{name} should be ignored under {RESULTS}/"
    for name in ("storage_bench.md", "summary.json", "plot.png", "figure.svg"):
        result = subprocess.run(
            [git, "check-ignore", "-q", f"{RESULTS}/{name}"], cwd=REPO_ROOT, check=False
        )
        assert result.returncode == 1, f"{name} must be committed, not ignored"

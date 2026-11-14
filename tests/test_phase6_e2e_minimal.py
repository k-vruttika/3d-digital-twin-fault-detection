"""Phase 6 end to end, minimal mode: the dashboard in headless Chromium against the backend
replaying a fixed seeded file into SQLite (no Docker, broker or ML).

Runs the Playwright project ``minimal`` (frontend/e2e/minimal.spec.ts; retries 0). Skipped
when Playwright's Chromium is not installed (``uv run poe browsers``), except with
``LINETWIN_REQUIRE_SERVICES=1`` (CI on Ubuntu), where a missing browser fails the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twin_backend.e2e_stack import minimal_stack, run_playwright
from twin_backend.testing import require_chromium

pytestmark = pytest.mark.e2e


def test_dashboard_minimal_mode(tmp_path: Path) -> None:
    require_chromium()
    with minimal_stack(tmp_path) as (stack, moments):
        res = run_playwright(
            "minimal",
            stack,
            {"E2E_MOMENTS": json.dumps({k: v.as_dict() for k, v in moments.items()})},
        )
    assert res.returncode == 0, (
        f"Playwright failed (traces in {tmp_path / 'playwright'}):\n"
        f"{res.stdout[-6000:]}\n{res.stderr[-3000:]}"
    )

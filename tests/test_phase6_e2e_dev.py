"""Phase 6 end to end, dev mode: the dashboard against ``devstack --source mock`` (simulator
publisher + backend on a throw-away TimescaleDB database).

Runs the Playwright project ``dev`` (frontend/e2e/dev.spec.ts; retries 0): inject a fault,
WARNING -> FAULT and its alert, STOP, BLOCKED / STARVED and line_stoppage, acknowledge,
restart, replay the build-up. Needs the Docker Compose services (``uv run poe up``) and
Playwright's Chromium (``uv run poe browsers``); skipped without them, failed instead with
``LINETWIN_REQUIRE_SERVICES=1``.

The mock machine server listens on ``sources.yaml`` ``mock.base_url``; a ``uv run poe dev``
running at the same time would collide with it.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from twin_backend.e2e_stack import dev_stack, run_playwright
from twin_backend.testing import PgService, require_chromium

pytestmark = [pytest.mark.e2e, pytest.mark.services]


def test_dashboard_dev_mode_fault_stop_ack_replay(
    tmp_path: Path, timescale_test_db: PgService
) -> None:
    require_chromium()
    key = secrets.token_hex(16)
    with dev_stack(tmp_path, timescale_test_db.dsn, key) as stack:
        res = run_playwright("dev", stack, {"E2E_OPERATOR_KEY": key})
    assert res.returncode == 0, (
        f"Playwright failed (traces in {tmp_path / 'playwright'}; backend log "
        f"{tmp_path / 'devstack.log'}):\n{res.stdout[-6000:]}\n{res.stderr[-3000:]}"
    )

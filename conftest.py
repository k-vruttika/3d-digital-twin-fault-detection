"""Repo-wide pytest configuration: shared fixture plugins and the event loop type."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from twin_common.aio import new_event_loop

pytest_plugins = ["twin_sim.testing", "twin_backend.testing"]


def pytest_asyncio_loop_factories(config: pytest.Config, item: pytest.Item) -> Mapping[str, object]:
    """Async tests run on the same loop type as production (selector loop on Windows)."""
    return {"linetwin": new_event_loop}

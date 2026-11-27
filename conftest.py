"""Repo-wide pytest configuration: shared fixture plugins, the event loop type, and the guard
that keeps torch and XGBoost out of one pytest process (ADR 0020)."""

from __future__ import annotations

import sys
from collections.abc import Mapping

import pytest

from twin_common.aio import new_event_loop

pytest_plugins = ["twin_sim.testing", "twin_backend.testing", "twin_ml.testing"]

# Both ship an OpenMP runtime; on macOS two copies of libomp in one process segfault (exit 139).
OPENMP_CLASH = ("torch", "xgboost")
_clash_reported = False


RUN_DEMO_OPTION = "--run-demo"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        RUN_DEMO_OPTION,
        action="store_true",
        default=False,
        help="also run the real-time demo scenarios (marker 'demo', about 40 min; "
        "uv run poe demo-tests)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """The demo scenarios are opt-in: skipped unless ``--run-demo`` is given."""
    if config.getoption(RUN_DEMO_OPTION):
        return
    skip = pytest.mark.skip(
        reason=f"real-time demo scenario: opt in with {RUN_DEMO_OPTION} (uv run poe demo-tests)"
    )
    for item in items:
        if item.get_closest_marker("demo") is not None:
            item.add_marker(skip)


def pytest_asyncio_loop_factories(config: pytest.Config, item: pytest.Item) -> Mapping[str, object]:
    """Async tests run on the same loop type as production (selector loop on Windows)."""
    return {"linetwin": new_event_loop}


def openmp_clash() -> bool:
    """True if every library of ``OPENMP_CLASH`` is imported in this process."""
    return all(name in sys.modules for name in OPENMP_CLASH)


def clash_message(where: str) -> str:
    return (
        f"{' and '.join(OPENMP_CLASH)} are both imported in the pytest process ({where}). "
        "On macOS their two OpenMP runtimes crash the process with a segmentation fault "
        "(exit code 139), so tests must load at most one of them in-process and run the other "
        "in a subprocess (twin_ml.testing.run_pytest_isolated or a worker process). "
        "See docs/adr/0020-ml-models-training-tracking.md, amendment 'one worker process per "
        "training run (macOS OpenMP)'."
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Module-level imports during collection count too."""
    global _clash_reported
    if openmp_clash() and not _clash_reported:
        _clash_reported = True
        raise pytest.UsageError(clash_message("after test collection"))


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Fail the first test after which both libraries are loaded (an error at its teardown)."""
    global _clash_reported
    if openmp_clash() and not _clash_reported:
        _clash_reported = True
        pytest.fail(clash_message(f"first seen after {item.nodeid}"), pytrace=False)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Make the session fail even if the error above was somehow not counted."""
    if openmp_clash() and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

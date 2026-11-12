"""twin_common.aio: the event loop type LineTwin runs on (selector loop on Windows)."""

from __future__ import annotations

import asyncio
import sys

import pytest

from twin_common import aio


async def _loop_type() -> type[asyncio.AbstractEventLoop]:
    return type(asyncio.get_running_loop())


def test_new_event_loop_type() -> None:
    loop = aio.new_event_loop()
    try:
        if sys.platform == "win32":
            assert isinstance(loop, asyncio.SelectorEventLoop)
        else:
            assert isinstance(loop, asyncio.AbstractEventLoop)
    finally:
        loop.close()


def test_run_uses_selector_loop_on_windows() -> None:
    loop_type = aio.run(_loop_type())
    if sys.platform == "win32":
        assert issubclass(loop_type, asyncio.SelectorEventLoop)
    assert aio.run(asyncio.sleep(0, result=42)) == 42


async def test_pytest_async_tests_run_on_the_linetwin_loop() -> None:
    """The root conftest's pytest_asyncio_loop_factories hook applies to every async test."""
    loop = asyncio.get_running_loop()
    if sys.platform == "win32":
        assert isinstance(loop, asyncio.SelectorEventLoop)
    aio.require_selector_loop("test")  # must not raise on the LineTwin loop


@pytest.mark.skipif(sys.platform != "win32", reason="the Proactor loop exists on Windows only")
def test_require_selector_loop_rejects_proactor() -> None:
    async def check() -> None:
        aio.require_selector_loop("aiomqtt client")

    proactor = asyncio.ProactorEventLoop()  # type: ignore[attr-defined,unused-ignore]
    try:
        with pytest.raises(RuntimeError, match=r"aiomqtt client needs a selector event loop"):
            proactor.run_until_complete(check())
    finally:
        proactor.close()

"""One way to run asyncio code on every platform (see ADR 0008).

aiomqtt needs ``loop.add_reader``, which Windows' default Proactor event loop does not
implement. Every LineTwin entry point therefore runs its coroutine through :func:`run`, which
uses a selector event loop on Windows and the platform default elsewhere. uvicorn is started
inside that loop (``uvicorn.Server.serve``), never with ``uvicorn.run``.

Selector loops on Windows do not support asyncio subprocesses, so LineTwin code never uses the
asyncio subprocess APIs.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

IS_WINDOWS = sys.platform == "win32"


def new_event_loop() -> asyncio.AbstractEventLoop:
    """A new event loop of the type LineTwin uses on this platform."""
    if IS_WINDOWS:
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def run(main: Coroutine[Any, Any, T]) -> T:
    """Run ``main`` to completion on a LineTwin event loop (selector loop on Windows)."""
    with asyncio.Runner(loop_factory=new_event_loop) as runner:
        return runner.run(main)


def require_selector_loop(component: str) -> None:
    """Fail with a clear message if the running loop cannot do ``add_reader`` (Windows Proactor)."""
    loop = asyncio.get_running_loop()
    if IS_WINDOWS and not isinstance(loop, asyncio.SelectorEventLoop):
        raise RuntimeError(
            f"{component} needs a selector event loop, but the running loop is "
            f"{type(loop).__name__} (the Windows default). Start the program with "
            "twin_common.aio.run() instead of asyncio.run()."
        )

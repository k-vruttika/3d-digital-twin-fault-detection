"""Run an ASGI app with uvicorn inside the current event loop (see ADR 0008).

``uvicorn.run`` would create its own event loop (a Proactor loop on Windows); we always call
``uvicorn.Server.serve`` from a loop started by ``twin_common.aio.run`` instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from typing import Any

import uvicorn

# Seconds uvicorn waits for open connections (WebSocket streams) on shutdown.
GRACEFUL_SHUTDOWN_S = 1


def make_server(app: Any, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        loop="none",  # use the running loop (selector loop on Windows)
        ws="websockets-sansio",
        lifespan="on",
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    return uvicorn.Server(config)


class ServerHandle:
    """Start/stop a uvicorn server in the background of the running loop (tests, embedding).

    ``port=0`` picks a free port; ``port`` holds the bound port after ``start()``.
    """

    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 0) -> None:
        self.app = app
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> int:
        sock = socket.create_server((self.host, self.port))
        self.port = sock.getsockname()[1]
        self._server = make_server(self.app, self.host, self.port)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]), name="uvicorn")
        while not self._server.started:
            if self._task.done():
                self._task.result()  # re-raise the startup error
            await asyncio.sleep(0.01)
        return self.port

    async def stop(self) -> None:
        if self._server is None or self._task is None:
            return
        self._server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._server = None
        self._task = None

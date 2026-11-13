"""Run the API with uvicorn inside the running event loop (ADR 0008).

``uvicorn.run`` would create its own event loop (a Proactor loop on Windows), so the backend
always awaits ``uvicorn.Server.serve`` from a loop started by ``twin_common.aio.run``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from typing import Any

import uvicorn

from twin_backend.api.app import create_app
from twin_backend.runtime import Runtime
from twin_common.aio import require_selector_loop
from twin_common.logging import SecretFilter

# Seconds uvicorn waits for open connections (WebSocket streams) on shutdown.
GRACEFUL_SHUTDOWN_S = 2
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def install_log_filters() -> None:
    """Redact secrets in uvicorn's (standard-library) log records."""
    for name in UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(f, SecretFilter) for f in logger.filters):
            logger.addFilter(SecretFilter())


def make_server(app: Any, host: str, port: int, *, access_log: bool = True) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        loop="none",  # use the running loop (selector loop on Windows)
        ws="websockets-sansio",
        lifespan="on",
        log_config=None,
        access_log=access_log,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    install_log_filters()
    if access_log:
        # uvicorn turns propagation off process-wide when any server runs without access logs
        # (log_config=None: our handlers sit on the root logger, so turn it back on).
        logging.getLogger("uvicorn.access").propagate = True
    return uvicorn.Server(config)


async def serve(runtime: Runtime, *, host: str, port: int) -> bool:
    """Serve until interrupted; returns False if the app failed to start."""
    require_selector_loop("the LineTwin API")
    server = make_server(create_app(runtime, host=host), host, port)
    await server.serve()
    return bool(server.started)


class ServerHandle:
    """Start/stop a server in the background of the running loop (tests, the dev launcher).

    ``port=0`` picks a free port; ``port`` holds the bound port after ``start()``.
    """

    def __init__(
        self, app: Any, host: str = "127.0.0.1", port: int = 0, *, access_log: bool = False
    ) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.access_log = access_log
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/stream"

    async def start(self) -> int:
        sock = socket.create_server((self.host, self.port))
        self.port = sock.getsockname()[1]
        self._server = make_server(self.app, self.host, self.port, access_log=self.access_log)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]), name="uvicorn")
        while not self._server.started:
            if self._task.done():
                self._task.result()  # re-raise the startup error
                raise RuntimeError("the API failed to start (see the log)")
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

# ADR 0008: Selector event loop on Windows

- **Status:** Accepted
- **Date:** 2026-11-12
- **Phase:** 2 (adapters and ingestion)

## Context

aiomqtt (paho-mqtt underneath) needs `loop.add_reader`/`add_writer`. On Windows the default asyncio loop is the `ProactorEventLoop`, which does not implement them. On Linux and macOS the default loop is a selector loop already. The MQTT adapter and publisher run in the same process as the other adapters, the FastAPI mock server and, from Phase 4, the backend API.

## Decision

- **One helper:** `twin_common.aio.run(coro)` runs coroutines on `asyncio.Runner(loop_factory=new_event_loop)`. `new_event_loop()` returns `asyncio.SelectorEventLoop()` on Windows and the platform default elsewhere. No event loop policy is changed globally; `WindowsSelectorEventLoopPolicy` is deprecated from Python 3.14 on.
- **Every async entry point uses it:** the simulator CLI (`run`, `mock-server`, `mqtt`, `opcua`) now, and the backend server in Phase 4.
- **uvicorn is started programmatically** inside that loop: `uvicorn.Server(Config(..., loop="none")).serve()` (`twin_sim.publishers.http`). It is never started with `uvicorn.run`, which would set up its own loop.
- **pytest:** the root `conftest.py` implements pytest-asyncio's `pytest_asyncio_loop_factories` hook and returns `new_event_loop`, so every async test runs on the production loop type. Overriding the `event_loop_policy` fixture is deprecated in pytest-asyncio 1.4.
- **Guard:** `require_selector_loop(component)` raises a clear error, naming the fix, when MQTT code finds itself on a Proactor loop.
- **No asyncio subprocess APIs** anywhere: selector loops on Windows do not support them. Synchronous `subprocess.run` (tests, git hash) is unaffected.

## Consequences

- The same code runs on Windows, Linux and macOS, and tests check the loop type on Windows (`common/tests/test_aio.py`).
- Selector loops on Windows are limited to 512 sockets by `select()`. That is far above what a desktop twin needs.
- Any new async entry point must go through `twin_common.aio.run`.

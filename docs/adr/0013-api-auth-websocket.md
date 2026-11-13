# ADR 0013: Backend API, API-key auth and WebSocket backpressure

- **Status:** Accepted
- **Date:** 2026-11-13
- **Phase:** 4 (twin engine, API, commands)

## Context

The twin needs a REST API and a live stream for the dashboard and outside systems. It must start through `twin_common.aio.run()` (a selector loop on Windows, ADR 0008), work in minimal mode without services, protect write operations, and never let one slow consumer slow down ingestion or other consumers.

## Decision

### Server

`create_app(runtime)` (FastAPI) wires the routers, the `/stream` hub, CORS (origins from `api.yaml`, by default the Vite dev server) and auth. The app's lifespan starts and stops the runtime. `serve()` awaits `uvicorn.Server.serve()` inside the running loop, never `uvicorn.run`, and calls `require_selector_loop` first. `python -m twin_backend serve --mode minimal|dev|config` is the entry point:

- `minimal`: file source + SQLite;
- `dev`: the active source + TimescaleDB;
- `config`: as the files say.

`GET /health` reports the event loop class, so the selector loop can be checked from outside.

### Auth: API keys mapped to roles

- **Keys and roles:** keys come from `.env`, never from committed files: `LINETWIN_API_KEYS_OPERATOR` / `_VIEWER`, each `name:key,...`. Roles are viewer < operator. Only operators can send commands, switch sources, acknowledge alerts and inject scenarios.
- **REST:** the key travels in the `X-API-Key` header. It is compared in constant time against every configured key.
- **No keys configured:** anonymous read, no writes, and only on a loopback bind address. A non-loopback bind without keys refuses to start, with instructions. A warning is logged whenever the server runs without keys. Minimal mode works out of the box and is not exposed to the network.
- **WebSocket:** browsers cannot set headers on a WebSocket, and query strings end up in access logs. So the key is the first message, `{"type": "auth", "apiKey": ...}`, sent within `auth_timeout_s`. A wrong key, or a different first message, closes the connection with code 4401. Silence until the timeout gives anonymous read access only under the no-keys loopback rule.
- **Redaction:** no log line may contain a key. `twin_common.logging` registers every configured key as a secret. A structlog processor masks credential-named fields and every registered value, including inside the event message. A standard-library `SecretFilter` does the same for uvicorn's access and error logs, on the fully formatted message, and also masks `apiKey=` style query parameters. A test captures all logs during requests with valid and invalid keys and finds none of them.

### WebSocket backpressure

The hub has one bus subscription per topic and fans out without ever waiting. Each client has its own `Outbox`:

- **Coalesced channels** (readings per machine, the line view, data quality per machine) keep only the newest value until the client takes it.
- **Ordered channels** (state changes, alerts, commands and acks, predictions) go into a bounded queue (`client_queue_size`) that drops the oldest. The loss is reported to the client with a `dropped` message.
- **Senders:** each client has its own sender task. A send taking longer than `send_timeout_s` closes that client with code 4408.

A slow or stalled client therefore delays only itself. A test runs one client that never reads and one that reads every 0.5 s against a 160 readings/s replay. The normal client receives every reading, ingestion keeps its pace, nothing is dropped by the tracker or the writer, and the slow clients' buffers stay bounded.

## Consequences

- The dashboard (Phase 5+) connects with a viewer key. The Vite dev server works without extra setup.
- Clients must expect coalescing: a slow client sees the newest reading per machine, not every reading. A client that needs every row uses `/history`.
- The predict and what-if endpoints exist and return 501 until Phase 11.

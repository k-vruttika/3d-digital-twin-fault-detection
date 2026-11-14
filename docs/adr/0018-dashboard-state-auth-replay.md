# ADR 0018: Dashboard: lightweight state, key handling, replay and real-backend tests

- **Status:** Accepted
- **Date:** 2026-11-14
- **Phase:** 6 (dashboard)

## Context

The dashboard combines the 3D engine (ADR 0016) with panels for machines, charts, alerts, commands, scenarios, sources and replay. All of them show the same few things: machine states, the line's root causes, alerts, connection state and the operator's role. These arrive from a WebSocket at up to a few messages per second per machine, from REST calls, and in replay from `/replay`. The decisions below cover where that state lives, how the API key is handled, how replay works, which backend gaps had to be closed, and how the dashboard is tested end to end.

## Decision

### State: small external stores, no state library

- **One model per page, `DashboardModel`** (`frontend/src/dashboard/state/model.ts`).
  - It implements the `EngineSink` interface that `bindStream` already feeds, so stream data, the 3D view and the panels have one ingress.
  - It keeps a handful of typed stores (`createStore`: `get` / `set` / `subscribe`, about 30 lines) for the line, machines, line view, alerts, connection, auth, selection, mode, label mode and sources.
  - React reads them with `useSyncExternalStore`.
- **The engine attaches when it arrives.** The model forwards every message to it once it has loaded (it is a separate chunk) and replays the current state into it on attach. The engine never imports React or the dashboard.
- **High-rate data stays outside React.** Readings also go to plain listeners. The live charts subscribe to them directly, buffer readings in a ref, and flush once per animation frame.
- **Alternatives considered:**
  - **Redux Toolkit:** reducers, actions and middleware for state that is mostly "the latest value per machine". More code, a dependency, and the high-rate path would still need to bypass it.
  - **Zustand:** close to what we wrote, but a runtime dependency for ~30 lines. The project asks to avoid new runtime dependencies.
  - **React context with useReducer only:** every reading would re-render every consumer of the context.
- **Consequence:** everything is plain TypeScript. Tests drive the real model and render components against it, with fake REST, chart and engine factories (`src/test/dashboard.tsx`).

### API key: memory or sessionStorage, never localStorage or a URL

- **Entry and check:** the operator types the key into the header. `GET /auth/me` (new) checks it and returns the role, which the header shows.
- **Storage:** the key is kept in memory. With "remember in this tab" it also goes to sessionStorage, which ends with the tab. It is never written to localStorage, which survives on the device, and never put in a URL.
- **Transport:** REST sends it as `X-API-Key`, the WebSocket in its first frame (ADR 0013). Tests check that no request URL contains the key and that localStorage never holds it.
- **Role gating is a convenience, not security.** Operator controls are hidden or disabled for viewers; the backend still enforces roles on every request.

### Connection: three states the operator can act on

- **States:** *reconnecting* (down for less than 5 s, retries running) and *link lost* (longer) are told apart, so a short blip does not look like an outage. *Key rejected* covers 4401.
- **While not connected:** a banner says the data is out of date, and every operator action is disabled. The engine already shows "link lost" on the machines.

### Replay drives the same model

- **Seeking:** the replay controller (`state/replay.ts`) switches the model and engine to replay mode, so `bindStream` ignores live data. It then applies `/replay?t=` as a snapshot.
- **Playing:** play advances event time by speed × 500 ms per tick, with one request in flight. Ticks that arrive meanwhile only move the target (coalesced), so a slow store never builds a queue.
- **Return to live:** the stream is resubscribed, and its fresh snapshot replaces the replayed state.
- **Data quality is not replayed.** It is computed over a live wall-clock window and cannot be reconstructed for a past t. In replay it shows as "not replayed" instead of the current value.

### Backend gaps closed for the dashboard

All are additions; nothing existing changed behaviour:

- `GET /auth/me`: principal and role of the presented key;
- `GET /commands/limits`: per-kind limits and the state rules from `commands.yaml`, so inputs carry the backend's limits instead of copies;
- `GET /replay/range`: the stored span, for the scrubber;
- `/replay` now also returns `line`, the root causes computed by the engine's propagation on the reconstructed states, and `alerts`, those unresolved at t;
- `GET /sources` returns an `ingest` block with counters and latency from `IngestMetrics`;
- `GET /scenario/options`: fault types per machine kind, read from `simulator.yaml` without importing the simulator, and checked against `twin_sim` in a test;
- `serve --cors-origin` and `devstack --cors-origin`: extra CORS origins for a Vite server on a free port.

**No sensor thresholds were added.** They exist only in the simulator's config, perturbed per run. Showing them would invent alarm limits the twin does not have, so the charts draw none.

### Root causes in frontend-only scenes come from the backend's rule

- The synthetic "states" scene of the dev page derives its root causes with a TypeScript port of `engine/propagation.py` (`engine/rootCause.ts`).
- `uv run poe api-docs` writes `root_causes.json`, the backend propagation run on 208 state combinations. A Vitest test checks the port against it, and a pytest test checks the file is current.

### End-to-end tests and screenshots run against the real backend

- **Orchestration in pytest.** `twin_backend/e2e_stack.py` starts the backend (minimal: a fixed seeded file with `--no-loop`; dev: the mock server and `devstack --no-sim` on a throw-away TimescaleDB database) and Vite on free ports. It then runs `@playwright/test` with the URLs in the environment.
  - The `services` marker and `LINETWIN_REQUIRE_SERVICES` therefore work as for the other tests.
  - Retries are 0, so flakiness shows as failure. Traces and screenshots are kept for failures.
- **The mock server is started by the stack, not by devstack.** On Windows, terminating a process does not run its cleanup, so a child of devstack would outlive it and keep the mock port.
- **Screenshots use the same stack.** States are made deterministic by replay seeks to moments found in the generated file (first WARNING, first FAULT, first STARVED), not by recorded JSON.
- **CI:** browsers are installed and the e2e tests run only in the Ubuntu job, which already has the services. Windows and macOS keep the unit tests and the build.

## Consequences

- No new runtime dependency besides Chart.js, which is in the stack. New dev dependencies: Testing Library, jsdom, `@playwright/test` (same version as `playwright`).
- The model is the single place to extend for Phase 11 predictions: `setPrediction` already reaches it and the engine.
- The e2e dev test uses the mock server's fixed port from `sources.yaml`, so it cannot run while `uv run poe dev` is running. The stack checks the port first and says so.

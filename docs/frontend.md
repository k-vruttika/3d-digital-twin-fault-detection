# Frontend: the 3D twin engine (Phase 5)

The frontend has three parts:

- `frontend/src/engine/`: **TwinEngine**, a framework-agnostic Three.js renderer of the line. It never imports React (an ESLint rule enforces this).
- `frontend/src/api/`: a typed REST client, the WebSocket client and the glue that feeds the engine.
- `frontend/dev.html`: a full-screen dev page that shows the engine against a running backend. The React dashboard comes in a later phase.

Design decisions are in [ADR 0016](adr/0016-3d-twin-engine.md) (engine) and [ADR 0017](adr/0017-generated-api-types.md) (generated API types).

## Running it

Two terminals: the backend in one, the Vite dev server in the other.

```text
# minimal mode: file replay + SQLite, no Docker
uv run poe sim-quick          # once: writes data/sim_quick/run_0000.csv
uv run poe minimal            # terminal 1: API on http://127.0.0.1:8000
uv run poe frontend           # terminal 2: http://127.0.0.1:5173/dev.html

# dev mode: simulator publisher + TimescaleDB
uv run poe up
uv run poe dev --source mock  # terminal 1 (a live source: STOP/START work; the file source rejects commands)
uv run poe frontend           # terminal 2
```

- **Ending the replay:** `uv run poe minimal --no-loop` stops at the end of the file instead of looping (the default in `sources.yaml`). Every machine then turns semi-transparent (STALE, after 3 s) and grey (OFFLINE, after 5 s); the thresholds are in `sources.yaml` `ingest`.
- **Backend with API keys:** if `.env` sets `LINETWIN_API_KEYS_*`, the page needs a key. Put a viewer key in `frontend/.env.local` (gitignored; Vite reads it at start-up):

  ```text
  VITE_LINETWIN_API_KEY=<viewer key>
  VITE_LINETWIN_API_URL=http://127.0.0.1:8000   # optional, this is the default
  ```

  Keys never go into a URL. Without keys configured, the backend serves anonymous read access on loopback, and the page needs nothing.
- **Sending a command** to watch the twin react (operator key):

  ```text
  curl -X POST http://127.0.0.1:8000/commands -H "X-API-Key: <operator key>" -H "Content-Type: application/json" -d "{\"machineId\":\"CNC-01\",\"type\":\"STOP\",\"params\":{},\"idempotencyKey\":\"stop-1\"}"
  ```

  CNC-01 shows IDLE straight away. Over the next seconds, as the buffers fill and drain, CNV-01 turns `BLOCKED by CNC-01` and INSP-01 and CNV-02 turn `STARVED by CNC-01`, each with a yellow beacon. START brings the line back.

### Dev page options

| URL | What it shows |
|---|---|
| `/dev.html` | the live backend |
| `/dev.html?fps` | adds the FPS counter (fps and the worst frame in the last second) |
| `/dev.html?synthetic=20` | a generated 20-machine line with synthetic data, no backend (performance check) |
| `/dev.html?synthetic=line` / `single` | the shipped `line.yaml` / `line_single.yaml` layout with synthetic data |
| `/dev.html?synthetic=states` | one machine in each state (screenshots) |
| `&freeze=10` | freezes time (deterministic frames for screenshots) |

Orbit with the left mouse button, pan with the right, zoom with the wheel. Click a machine to select it; hover shows a tooltip.

### Screenshots

`uv run poe screenshots` installs Playwright's Chromium (first run only) and opens the synthetic scenes headless with software WebGL (SwiftShader). It writes PNGs and a `summary.json` to `experiments/results/frontend_screenshots/`:

- the running line;
- each state;
- the single-station line;
- the 20-machine line;
- a hover tooltip.

The script fails if the page logs an error.

## Layout from the backend

`GET /line` returns the loaded line config: machines (kind, parts, sensor → part map), buffers, the sensor catalog, and the same topology the twin engine uses for root causes. `computeLayout` (pure, in `engine/layout.ts`) places it:

- machines along +x in flow order, `layout.gap` apart; buffers sit in the gaps;
- after `layout.rowLength` machines the line turns back (a serpentine), `rowSpacing` further along +z;
- an optional per-machine override in the line config wins. Positions are world coordinates in metres, y up; either key works on its own:

  ```yaml
  - id: CNC-01
    ...
    layout: { position: [10.0, 0.0, 5.0], rotation_deg: 90 }
  ```

Adding a machine to the YAML adds it to the scene after a backend restart. The WebSocket client re-fetches `/line` on every reconnect and reloads the scene if it changed.

## Geometry

All geometry is procedural (`engine/geometry/`): unit geometries are shared and scaled per mesh, rollers and parts use `InstancedMesh`, and the belt is a `DataTexture` whose offset scrolls.

| Kind | Model | Config parts |
|---|---|---|
| CONVEYOR | frame, legs, rollers, textured belt with end drums, motor | `belt`, `motor`, `bearing` |
| CNC | bed, table, column, spindle head with rotating spindle, glass enclosure | `spindle`, `spindle_motor`, `spindle_bearing` |
| INSPECTION | base, rotating indexer table, gantry with a sweeping camera head | `indexer`, `motor`, `bearing` |
| other | a box | |

Every part named in a machine's `parts` list is a named `Group`. A part the builder does not know gets a small housing, so it is always addressable (`highlightParts`, outlines, flashing). Each part has its own materials.

## State → visual mapping

`engine/visuals.ts` (pure, unit tested). Motion follows the reported speed: rate = speed / reference RPM per kind (`config.ts`), smoothed and clamped.

| State | Motion | Look | Label |
|---|---|---|---|
| RUNNING | animated | normal | `RUNNING` |
| IDLE | stopped | normal | `IDLE` |
| WARNING | animated | orange outline on the at-risk part | `WARNING` |
| FAULT | stopped | at-risk part flashes red | `FAULT` |
| STARVED / BLOCKED | stopped | yellow beacon | `BLOCKED by CNC-01` (root cause from the WS `line` channel) |
| MAINTENANCE | stopped | blue beacon, blue tint | `MAINTENANCE` |
| STALE | stopped | semi-transparent | `STALE` |
| OFFLINE | stopped | greyed out | `OFFLINE` |
| stream down | stopped | semi-transparent | `LINK LOST` |
| no data yet | stopped | normal | `NO DATA` |

- **At-risk part:** the highest-weighted attribution of the newest prediction (`setPrediction`) that names one of the machine's parts. Without one (predictions arrive in Phase 11), the whole machine.
- **Overlays on any state:**
  - data quality DEGRADED: a faint violet ring on the floor;
  - active alerts: a floating marker in the colour of the highest severity (INFO blue, WARNING amber, CRITICAL red).
- **Labels:** camera-facing text above each machine, so colour is never the only cue. Turn them off with `labels.enabled` in `engine/config.ts`.

### Buffers: inferred, never counted

The line reports no buffer levels, so a buffer shows only what its neighbours' states imply:

| Status | When | Shown |
|---|---|---|
| FULL | the upstream machine is BLOCKED | a stack of `capacity` parts (then the count is known) |
| EMPTY | the downstream machine is STARVED | an empty tray |
| FLOWING | neither | two parts passing through, no number |
| UNKNOWN | a neighbour is STALE, OFFLINE or has no data | a greyed tray, label `BUFFER ?` |

The tooltip explains the inference. Real buffer telemetry is a backlog item (PROGRESS.md).

Parts on conveyors move with the belt and disappear when the conveyor is STARVED, STALE or OFFLINE.

## TwinEngine API

```ts
const engine = new TwinEngine(container, { fps?: boolean, tooltip?: boolean, config?: VisualConfig, ... });
engine.loadLayout(line);                       // GET /line
engine.applySnapshot(snapshot);                // WS snapshot or GET /replay
engine.applyReading(reading);
engine.applyStateChange(change);
engine.applyDataQuality(report);
engine.applyAlert(alert);
engine.applyLine(lineView);
engine.setPrediction(prediction);
engine.highlightParts("CNC-01", [{ part: "spindle_bearing", intensity: 0.8 }]);   // [] clears
engine.setMode("live" | "replay");
engine.setConnected(false);                    // shows "link lost"
const off = engine.on("select", ({ machineId }) => ...);   // also "hover": target, tooltip, x, y
engine.resize();                               // also automatic with ResizeObserver
engine.dispose();                              // frees everything; safe to call twice
```

- **Wiring:** `bindStream(streamClient, engine, { fetchLine })` (`api/bind.ts`) does all of it from the WebSocket. In replay mode it ignores live data.
- **Injectable dependencies:** the renderer, controls, label textures, clocks and the frame loop. The tests run the engine headless with fakes.
- **`dispose()`:**
  - cancels the frame loop;
  - removes every DOM listener and the ResizeObserver;
  - disposes the controls, the renderer (and forces a context loss), and every geometry, material and texture (tracked by `ResourceTracker`);
  - removes the canvas, tooltip and FPS elements.

  `resourceCounts()` reports what is still held. After `dispose()` every count is 0 (tested).

## API client

- **`RestClient`** (`api/rest.ts`): typed methods for `/line`, `/machines`, `/machines/{id}/state`, `/history`, `/replay`, `/alerts`, alert ack, `/commands`, `/health` and `/sources`.
  - Sends `X-API-Key`.
  - Throws `ApiError` with status and detail.
  - Every endpoint it calls is checked against the committed OpenAPI schema.
- **`StreamClient`** (`api/ws.ts`):
  - Handshake: `auth` is always the first frame, `subscribe` goes only after the server's `auth` reply, then the snapshot.
  - Reconnects with exponential backoff (seeded jitter) after any close except 4401, including the server's 4408 slow-client close.
  - Every reconnect re-authenticates, re-subscribes with the current selection and delivers a fresh snapshot (`reconnect: true`). The backoff resets after a snapshot.
  - An optional ping heartbeat detects half-open sockets.
- **Types** are generated from the backend (ADR 0017). `uv run poe api-docs` refreshes `src/api/openapi.json` and `schema.gen.ts`, and tests fail when they are stale.

## Performance

- **Built in:**
  - shared unit geometries;
  - instanced rollers, conveyor parts and buffer stacks;
  - one invisible pick box per machine and buffer;
  - label textures redrawn only when the text changes;
  - per-frame work limited to animation, flashing and instance matrices of moving parts;
  - pixel ratio capped at `render.maxPixelRatio`;
  - shadows only for lines up to `render.shadowMaxMachines` (8).
- **FPS:** `?fps` shows the counter. The headless screenshot run uses software rendering, so its frame rate says nothing about a real GPU. **The 60 fps target (4 machines) and smoothness at 20 machines have not been measured here;** check `dev.html?synthetic=20&fps` on your machine.
- **Bundle:** the production build is one chunk of about 600 kB (three.js); Vite warns about the size.

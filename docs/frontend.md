# Frontend: the 3D twin engine (Phase 5) and the dashboard (Phase 6)

The frontend has four parts:

- `frontend/src/engine/`: **TwinEngine**, a framework-agnostic Three.js renderer of the line. It never imports React (an ESLint rule enforces this).
- `frontend/src/api/`: a typed REST client, the WebSocket client and the glue that feeds the engine.
- `frontend/src/dashboard/` (`index.html`): the **React dashboard** around the engine (Phase 6, [below](#dashboard-phase-6)).
- `frontend/dev.html`: a full-screen dev page that shows only the engine, against a running backend or synthetic data.

Design decisions are in [ADR 0016](adr/0016-3d-twin-engine.md) (engine), [ADR 0017](adr/0017-generated-api-types.md) (generated API types) and [ADR 0018](adr/0018-dashboard-state-auth-replay.md) (dashboard).

## Running it

Two terminals: the backend in one, the Vite dev server in the other.

```text
# minimal mode: file replay + SQLite, no Docker
uv run poe sim-quick          # once: writes data/sim_quick/run_0000.csv
uv run poe minimal            # terminal 1: API on http://127.0.0.1:8000
uv run poe frontend           # terminal 2: dashboard http://127.0.0.1:5173/ (3D only: /dev.html)

# dev mode: simulator publisher + TimescaleDB
uv run poe up
uv run poe dev --source mock  # terminal 1 (a live source: STOP/START work; the file source rejects commands)
uv run poe frontend           # terminal 2
```

- **Ending the replay:** `uv run poe minimal --no-loop` stops at the end of the file instead of looping (the default in `sources.yaml`). Every machine then turns semi-transparent (STALE, after 3 s) and grey (OFFLINE, after 5 s); the thresholds are in `sources.yaml` `ingest`.
- **Backend with API keys:** if `.env` sets `LINETWIN_API_KEYS_*`, a key is needed. The dashboard asks for it in the header (memory or sessionStorage only). The dev page reads a viewer key from `frontend/.env.local` (gitignored; Vite reads it at start-up):

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
| `/dev.html?synthetic=states` | every state, in short independent segments the simulator can produce (screenshots) |
| `&freeze=10` | freezes time (deterministic frames for screenshots) |

Orbit with the left mouse button, pan with the right, zoom with the wheel. Click a machine to select it; hover shows a tooltip.

### Screenshots

`uv run poe screenshots` installs Playwright's Chromium (first run only) and opens the synthetic scenes headless with software WebGL (SwiftShader). It writes PNGs and a `summary.json` to `experiments/results/frontend_screenshots/` (the dashboard screenshots, taken against the real backend, are described in [Dashboard screenshots](#dashboard-screenshots)):

- the running line;
- each state;
- the single-station line;
- the 20-machine line;
- a hover tooltip.

The script fails if the page logs an error.

## Layout from the backend

`GET /line` returns the loaded line config: machines (kind, parts, sensor → part map), buffers, the sensor catalog, and the same topology the twin engine uses for root causes. `computeLayout` (pure, in `engine/layout.ts`) places it:

- machines along +x in flow order, `layout.gap` apart; buffers sit in the gaps;
- after `layout.rowLength` machines the line turns back (a serpentine), `rowSpacing` further along +z. Where a row continues into the next (or an override puts linked machines out of line), the link is drawn as a **transfer track** with flow chevrons (`layout.ts` `linkPath`, `geometry/transfer.ts`), and the buffer sits halfway along it;
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

- **At-risk part:** the highest-weighted attribution of the newest prediction (`setPrediction`) that names one of the machine's parts. Without one, the whole machine.
- **Overlays on any state:**
  - data quality DEGRADED: a faint violet ring on the floor;
  - active alerts: a floating marker in the colour of the highest severity (INFO blue, WARNING amber, CRITICAL red).
- **Live inference (Phase 11, ADR 0030):**
  - **Part highlight** (`highlightParts`, cyan glow, `colors.highlight`): the parts with positive attributions in the supervised model's explanation of its own alarm. The dashboard sends at most `highlightTopParts` (2) parts, with intensity relative to the strongest (`highlightsFor` in `dashboard/state/predictions.ts`). It clears when the newest +30 s prediction is NORMAL, has no attributions, or is older than the stale time. It works on top of any state, so a part can glow while the machine is still RUNNING (the warning came before the state changed).
  - **Unusual behaviour** (`VisualSpec.unusual`, purple outline, `colors.unusual`): an active `unusual_behaviour` alert (autoencoder, cause unknown) outlines **every part of the machine**, never one part. It is not drawn over a state outline (WARNING) or a flashing part (FAULT), nor on a STALE / OFFLINE machine. The backend raises it only while the supervised model has no alarm, so it never comes with a part highlight.
- **Labels:** camera-facing text above a machine, so colour is never the only cue (`engine/labelPolicy.ts`, values in `engine/config.ts` `labels`):
  - **Declutter:** by default a machine label shows only for a state other than RUNNING / IDLE (NO DATA and LINK LOST count), and a buffer label only for FULL / EMPTY / UNKNOWN. Hovering or selecting shows the label anyway; `setLabelMode("all")` (the dashboard's "All labels" toggle) shows every label.
  - **Size:** a constant height on screen (`pixelHeight`, 22 px). On a viewport smaller than `referenceViewportPx` labels shrink with it, never below `minPixelHeight` (14 px).
  - **Distance:** beyond `fadeStart` (45 m) a label fades and is hidden at `hideDistance` (80 m); hovered and selected labels never fade.
  - `labels.enabled: false` turns them off.

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
engine.setLabelMode("auto" | "all");           // label declutter (see Labels)
engine.setTheme({ background, replayBackground, floor, grid });
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

- **`RestClient`** (`api/rest.ts`): typed methods for every endpoint the frontend uses (`ENDPOINTS`): `/line`, `/machines`, `/history`, `/replay`, `/replay/range`, `/alerts` and ack, `/commands` (send, audit, limits), `/auth/me`, `/sources` and switch, `/scenario` (send, audit, options), `/health`, `/predict` and `/predict/history` (Phase 11), and the `/whatif` stub (Part C).
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
- **FPS:** `?fps` shows the counter. The headless screenshot run uses software rendering, so its frame rate says nothing about a real GPU. On the user's GPU, `dev.html?synthetic=20&fps` runs at 60 fps, worst frame 21 ms.
- **Bundle:** both pages load the engine with a dynamic `import()`, so neither three.js nor the engine is in the initial bundle.
  - `vite.config.ts` gives three.js its own chunk group. three ships as two modules (`three.core.js`, `three.module.js`), and a `maxSize` on the group keeps them in separate chunks: about 350 kB and 205 kB, plus 20 kB for the controls.
  - Chart.js is its own chunk, loaded when the first chart appears.
  - The build has no chunk-size warning; the limit was not raised.

## Dashboard (Phase 6)

The page is `index.html` → `src/main.tsx` → `src/dashboard/`, built with React 19, the Phase 5 engine and API client, and Chart.js. State handling, auth and replay are explained in [ADR 0018](adr/0018-dashboard-state-auth-replay.md).

### Running it

```text
uv run poe sim-quick; uv run poe minimal      # terminal 1 (no Docker)
uv run poe frontend                           # terminal 2 -> http://127.0.0.1:5173/
uv run poe up; uv run poe dev --source mock   # live simulator instead: commands and scenario injection work
```

The backend URL comes from `VITE_LINETWIN_API_URL` (default `http://127.0.0.1:8000`). With API keys configured, enter one in the header.

### Layout

| Area | What it holds |
|---|---|
| Header | line id, connection pill, Replay button, API key field and role, "Key labels / All labels" toggle, light / dark theme |
| Banners | connection (while not connected), REPLAY bar (timeline, play, speeds, return to live) |
| Left | machine list: state badge, root cause ("BLOCKED by CNC-01"), DQ flag, active alert count |
| Centre | 3D view; below it the tabs Live charts, History, Alerts, Sources, Scenario (sim; only with the capability), Prediction and What-if |
| Right | details of the selected machine (state with reason, readings with units, data quality, active alerts) and the command panel |

- **Widths:** the grid is tuned for windows 1280 and 1920 px wide; the side columns widen from 1700 px.
- **Themes:** light and dark use CSS tokens on `:root[data-theme]`. The choice is a display preference kept in localStorage, and the 3D scene follows it (`engine.setTheme`).
- **Keyboard:**
  - a skip link jumps to the 3D view;
  - the machine list is a listbox: arrow keys and Home / End move, Escape clears the selection;
  - tabs move with the arrow keys;
  - dialogs take the focus, and Escape cancels them.
- **Screen readers:** state badges always carry text, and the connection banner and command results are `aria-live`.

### Connection, auth and roles

- **Connection** (`state/connection.ts`):

  | State | When |
  |---|---|
  | Connected | the stream is live |
  | Reconnecting… | the StreamClient is retrying, for less than `linkLostAfterMs` (5 s) |
  | Link lost | the stream has been down longer than that; the banner says how long |
  | Key rejected | the backend closed the stream with 4401 |

  While not connected, a banner says the data may be out of date. Every operator button (commands, acknowledge, scenario, source switch) is then disabled, with the reason as its tooltip.
- **API key** (`state/auth.ts`, `state/session.ts`):
  - Entered in the header and checked with `GET /auth/me`, which shows the role: anonymous viewer, viewer or operator. The dashboard then loads the layout and connects the stream with that key.
  - Kept in memory, or in sessionStorage with "remember in this tab". Never written to localStorage and never put in a URL: REST sends it as `X-API-Key`, the WebSocket in its first frame (tested).
- **Roles:** viewers see everything read-only. The command panel, Acknowledge, scenario injection and source switching appear only for operators.

### Charts

- **Live** (`charts/LiveCharts.tsx`): one chart per sensor for the selected machine, over the last 5 minutes of event time. Prefilled from `/history` (raw), then fed from the stream outside React and flushed once per animation frame.
- **History** (`charts/HistoryPanel.tsx`): one sensor at a time. Range presets (15 min, 1 h, 6 h, 24 h, all), drag across the plot to zoom, and a reset button; no zoom plugin.
  - The resolution follows the visible span: raw up to 30 min, 1-minute buckets up to 24 h, 10-minute buckets beyond. Buckets show the mean with a min–max band.
  - Ranges are anchored at the newest event time (or the replay time), never at the wall clock, and do not reach back before the first stored event.
  - The chart fills the rest of the bottom panel (flex column), so its axis labels stay visible at 1280 and 1920 px; the e2e and screenshot specs check that the canvas lies inside the panel.
- **Missing values are gaps, never zero** (`charts/series.ts`):
  - A missing sensor value (null, or absent from the JSON) and an empty bucket become `y: null`.
  - A time gap longer than 3 median sample intervals, or a missing bucket, inserts a null point.
  - Every dataset has `spanGaps: false`.
- **Units:** the y axes take them from the `/line` sensor catalog, e.g. "temp (°C)".
- **No threshold lines.** The backend provides no alarm limits: the warning and fault thresholds live in the simulator's config and are perturbed per run, and the `/line` catalog ranges are plausibility bounds for ingest validation, not alarm limits. The charts therefore draw none, and say so.

### Alerts

The panel shows the ISA-18.2 groups (`state/alerts.ts`):

- active, unacknowledged;
- active, acknowledged;
- cleared, unacknowledged.

Resolved alerts (cleared and acknowledged) leave the list. Within a group, alerts sort by severity, then newest first.

- **Times:** each alert shows when it was **raised**, and when it was **cleared** and **acked** (with by whom) once those happened, in the alerts panel and in the details panel (`alertTimes` in `state/alerts.ts`). An Alert event's `ts` is the time of its latest transition, so the dashboard keeps every lifecycle event per alert. For an alert first seen at a later rev (raised before the dashboard connected, after a restart, in replay), it fetches that machine's events up to the alert's current transition from `/alerts/history`, once per alert (`useAlertBackfill.ts`). In replay only events up to the replayed rev count, so a later clear or ack never shows early.

- **Filters:** machine and severity.
- **Acknowledge** (operator, connected): `POST /alerts/{id}/ack`; the new rev arrives through the stream.
- Clicking a machine id selects that machine everywhere.

### Commands (operator only)

- **STOP** is always shown and enabled while connected, since no rule can block it. It asks for confirmation.
- **START, RESET, Maintenance, Set speed, Set load:**
  - the parameter inputs carry the limits for the machine's kind from `GET /commands/limits`, and an out-of-range value cannot be sent;
  - one line lists the commands the state rules would reject right now; each button's tooltip gives the reason. The backend still decides.
- **Idempotency:** every click gets a new key (`crypto.randomUUID`).
- **Result** (`state/commandFlow.ts`): PENDING → ACCEPTED → EXECUTED, REJECTED (with the reason) or TIMEOUT.
  - ACCEPTED comes from the stream's ack, matched to the click through the `command` frame's idempotency key.
  - The final status comes from the POST response or the stream, whichever arrives first.
  - HTTP errors show their detail.
- **Recent commands:** the machine's audit trail for the last hour (`GET /commands`).

### Scenario injection (simulation only)

The panel appears only when the active source has the `scenario` capability, under a "SIMULATION ONLY" heading.

- It injects a fault on a chosen machine. The fault types offered are those of the simulator config that apply to the machine's kind (`GET /scenario/options`).
- An optional progression time must stay within the configured range.
- The request's ack status is shown, with recent requests from the audit trail.

### Sources

- **Source:** the active adapter, its capabilities and health.
- **Ingestion counters** since the backend started (`GET /sources` `ingest`, polled every 5 s):
  - received and published messages;
  - late %;
  - quarantined, by reason;
  - partial readings;
  - duplicates dropped;
  - end-to-end latency: mean and p95 (the upper bound of the histogram bucket).
- **Per-machine data quality** from the stream.
- **Switching** (operators, after a confirmation): if the new source fails, the backend keeps the old one and the panel shows why.

### Replay

**Replay** opens the REPLAY bar:

- **Scrubber:** spans the stored range (`GET /replay/range`) and shows `GET /replay?t=`: machine states, root causes and the alerts unresolved at t. The 3D view (engine replay mode), the list, the details and the alerts panel all follow it.
- **Play** at 1×, 5× or 20× advances event time every 500 ms. Only one request is in flight; later ticks just move the target.
- **Return to live** resubscribes the stream, and its fresh snapshot replaces the replayed state.

While replaying:

- live stream data is ignored;
- operator actions are disabled;
- live charts pause, and the History tab centres on the replay time;
- data quality shows as "not replayed": it is measured over a live wall-clock window, so it cannot be reconstructed for a past time;
- the 3D view gets an amber frame with a soft vignette. In the dark theme the scene background also turns a warmer dark; in the light theme the scene is not tinted (a warm tint over a light scene looked muddy), so the banner and the frame carry the cue.

### Prediction (Phase 11)

Live predictions from the backend's inference service (ADR 0030; `uv run poe dev --source mock --inference`, or `serve --inference`). Without inference the panel says "Live inference is off". Minimal mode needs none of it.

- **Data:**
  - predictions arrive on the WS `predictions` channel and in the snapshot; they are kept per machine and horizon in the model's `predictions` store;
  - `GET /predict` is polled every `predictPollMs` (2 s) in live mode (`usePredictPolling.ts`). It gives the inference state, the bundles with their model card, and the backend's per-machine gate;
  - in replay, `GET /replay` returns the stored predictions at t, and the gate is derived locally from the machine state and the prediction's age.
- **Table, one row per machine:** the current state, then the +10 s and +30 s predictions (the top class with its calibrated confidence), the conformal set at the primary horizon in words ("WARNING or FAULT", "empty set", "any state"), time to fault with its interval (or "beyond 600 s (no fault expected sooner)" at the model's cap), reliability, and an `alarm` badge.
  - The alarm is the evaluated rule (`Prediction.alarm`: uncalibrated +30 s argmax WARNING or FAULT for the sustain time), never the confidence.
  - **Sets can look odd.** Mondrian LAC gives each class its own threshold. The deployed GRU bundle's +30 s thresholds admit NORMAL only at p ≥ 0.997, and FAULT already at p ≥ 0.0014. So the live run showed healthy conveyors at "NORMAL 1.00" (0.996) with the set "FAULT". This is the evaluated Phase 8 method, not a display bug. The "About" note says so.
- **Gate texts instead of a prediction:** "No prediction (machine stopped)", "No prediction (machine waiting)", "Warming up (n/30 s of running data)", "Predictions unavailable (inference not ready)", "No recent prediction". A prediction older than the backend's `predictionStaleS` (5 s, relative to the newest data on screen) is never shown as current.
- **Selected machine:** the alarm, the top three attributions as `sensor → part (weight)` with the window they explain (`attributionsTs`; explanations run beside the prediction path, so the window can be slightly older than the prediction), and the autoencoder score against its threshold ("above threshold · cause unknown"). Once the machine stops or waits it shows the gate text instead, never the last alarm or score.
- **Model line:** the inference state (`ready`, `starting`, ...) and the model version, e.g. `linetwin-gru v1 (gru seed 0) + linetwin-autoencoder v1 (autoencoder seed 0)`.
- **"About these predictions"** (collapsed): the models were trained, validated and calibrated on simulator data only; Phase 10 did not demonstrate early warning on real machines; the coverage of the sets; what the alarm, unusual behaviour and highlights mean; plus the bundle's model card.
- **Details panel:** one line with the +30 s prediction (`PredictionSummary`) or the gate text.
- **Alerts panel:** inference kinds have badges (`AlertKindBadge`): `predicted WARNING`, `predicted FAULT`, `model drift`, and `unusual behaviour · cause unknown` in purple with a tinted row.
- **3D view:** part highlights and the unusual-behaviour outline, see [State → visual mapping](#state--visual-mapping).

### What-if

Still a placeholder until Phase 11 Part C: it calls `POST /whatif` and shows the backend's 501 detail; any other error shows as an error.

### Tests

**Vitest + React Testing Library (jsdom).** Chart.js and the engine are injected as fakes (`src/test/dashboard.tsx`). Covered:

- role gating;
- the connection banner (connected, reconnecting, link lost, key rejected), with operator buttons disabled;
- alerts: ISA-18.2 groups, filters, acknowledge, click to select;
- the command flow (pending → ACCEPTED → EXECUTED; REJECTED with its reason; TIMEOUT; HTTP errors), the STOP confirmation, limits, a fresh idempotency key per click;
- the scenario panel: visibility, label and request body;
- sources metrics;
- replay: banner, engine replay mode, return to live, request coalescing;
- predictions (`predictions.test.tsx`): the view helpers, the unusual-behaviour visual (whole machine, own colour, never over a state outline or on a stale machine), part highlights only from supervised attributions (cleared on NORMAL and when stale), the panel (gate texts, sets in words, time to fault, the honest note, inference off), no stale alarm in the details of a stopped machine, replay of stored predictions, and the distinct unusual-behaviour alert;
- the what-if 501 placeholder;
- missing values drawn as chart gaps;
- the API key never in a URL or localStorage.

**Playwright end to end** (`frontend/e2e/`; retries 0, trace and screenshot kept on failure). pytest runs both suites (`tests/test_phase6_e2e_*.py`, `twin_backend/e2e_stack.py`): it starts the backend and Vite on free ports, then runs `playwright test --project ...`.

- **minimal** (no Docker): a fixed seeded file (quick scenario, run 2) replayed with `--no-loop` into SQLite.
  - The dashboard loads and lists the four machines.
  - Selecting a machine shows its details, a live chart and a history chart.
  - Replay scrubs to RUNNING, FAULT and STARVED moments found in the file, and the engine's labels follow.
- **dev** (`services` marker): the mock machine server (quiet scenario `config/scenarios/e2e_quiet.yaml`, 30× speed) and `devstack --no-sim` on a throw-away TimescaleDB database, with an operator key. The test:
  1. injects `bearing_wear` on CNC-01 and sees WARNING, then FAULT, and `machine_fault`;
  2. sends STOP with confirmation (EXECUTED) and sees CNV-01 `BLOCKED by CNC-01`, INSP-01 STARVED and `line_stoppage`;
  3. acknowledges it;
  4. restarts the machine with maintenance, which clears the fault (afterwards the simulator runs the machine again), and sees the line RUNNING;
  5. replays back to the warning and fault times taken from `/alerts/history`.
- **Skipping:** without Chromium (install it with `uv run poe browsers`) both suites skip; with `LINETWIN_REQUIRE_SERVICES=1` they fail instead. In CI they run on Ubuntu only.

### Dashboard screenshots

`uv run poe screenshots` also runs `python -m twin_backend.e2e_stack screenshots`. It uses the same orchestration as the e2e tests; no recorded or hand-written data is involved. The PNGs go to `experiments/results/frontend_screenshots/dashboard/`:

- **minimal mode**, with replay seeks to moments found in the file: `overview_{dark,light}_{1280,1920}`, `machine_details`, `alerts`, `replay_mode`;
- **dev mode** (mock source, TimescaleDB): `sources`, `scenario_warning`, `command_confirm_stop`, `command_result`. These need `uv run poe up`; `--minimal-only` skips them.
- **prediction panel** (Phase 11; Playwright project `shots-predict`): a second dev-mode stack with `--inference`. The simulator runs in real time, and `bearing_wear` is injected on CNC-01 with a 720 s progression (the shortest training progression). These shots need `up` and the deployed bundles (`uv run poe deploy-models`); without the bundles they are skipped with a message (`summary.json` `predictMode.skipped`). `--no-predict` skips them and `--predict-only` reruns only them (about 12 minutes). The spec checks in the running app:
  1. `prediction_normal`: the healthy line, every machine predicted NORMAL after its warm-up;
  2. `prediction_unusual`, `alerts_unusual`, `twin_unusual_closeup`: the autoencoder's unusual-behaviour alarm while the supervised model still predicts NORMAL. Every part of CNC-01 is outlined in the unusual colour and no part is highlighted;
  3. `prediction_alarm`, `twin_alarm_closeup`, `alerts_predicted`: the supervised predicted alarm, with `spindle_bearing` as the strongest highlighted part;
  4. `prediction_warning`: the simulator's own WARNING, with the bearing outlined in the warning colour.
- **Why real time:** an explanation is attached to predictions only while it is younger than `prediction_stale_s` on the event-time axis (ADR 0030). With Chromium's software WebGL on the same CPU, Integrated Gradients took 1–5 s per explanation in this run. At 4× that is 4–20 s of event time, so no part was ever highlighted. The close-ups move the camera with `engine.focus` for the picture only.

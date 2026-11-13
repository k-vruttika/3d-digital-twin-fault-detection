# ADR 0016: 3D twin engine: framework-agnostic Three.js, procedural geometry, inferred buffers

- **Status:** Accepted
- **Date:** 2026-11-14
- **Phase:** 5 (3D twin engine, frontend)

## Context

The company brief asks for a desktop 3D station whose running, idle and fault states are driven by a file or a mock server. The twin must also show what later phases add: predictions, explanations on 3D parts, alerts and replay. It has to render whatever line the config describes (four machines, one station, or more) and stay testable without a GPU.

## Decision

### Engine shape

- **`TwinEngine`** (`frontend/src/engine/`) is a plain TypeScript class on Three.js. It does not use react-three-fiber (CLAUDE.md §4) and never imports React (ESLint `no-restricted-imports`). The React dashboard of a later phase will host it in a component.
- **Data in, nothing out:** the engine never fetches anything. All data arrives through public methods (`loadLayout`, `applySnapshot`, `applyReading`, ...). `api/bind.ts` connects a `StreamClient` to it. This keeps the engine usable for live data, `/replay` and synthetic data alike.
- **Injectable dependencies:** the renderer, controls, label textures, clocks and the frame loop. Unit tests run the real engine headless (Vitest, node) with a fake renderer, and check the scene graph, materials, picking and disposal.

### Layout from the backend

- **`GET /line`** returns the line config and the topology the backend engine uses (`Topology.from_line`), so the frontend never derives neighbours differently.
- **Placement:** the frontend computes it (pure `computeLayout`). Machines go along the flow, wrap into a serpentine after `rowLength` machines, and follow an optional per-machine `layout` override (`MachineLayout` in `twin_common`, new and optional).

### Procedural geometry

- **No model files:** all geometry is built in code from shared unit primitives.
- **Addressable parts:** each config part is a named group with its own materials, and unknown parts get a generic housing. Highlighting, outlines and flashing then work for any line config, including Phase 11 attributions (`sensor_parts`).
- **Outlines** are inverted hulls (BackSide copies grown by a fixed thickness), not a post-processing pass. They cost one extra draw per outlined mesh, and only while shown.

### State → visual mapping

- **Where:** one pure function (`machineVisual`), unit tested for every state.
- **Precedence:** OFFLINE (grey) > stream down or STALE (semi-transparent) > the machine state. Data-quality and alert overlays combine with any state.
- **Not colour alone:** every state also has a camera-facing text label, which names the root cause for STARVED / BLOCKED. It can be turned off in config.
- **At-risk part:** the top attribution of the newest prediction when it names a part of the machine; otherwise the whole machine. There is no guessing from sensor ranges: the catalog ranges are plausibility bounds, not alarm limits.
- **Link down:** while the WebSocket is down, the engine shows every machine as "link lost" rather than its last known state.

### Buffers are inferred, never counted

The Reading contract carries no buffer level. The simulator records one, but only as a dataset label column. The twin therefore shows what the data implies:

- **FULL** when the upstream machine is BLOCKED (the simulator blocks a machine exactly when its output buffer is full);
- **EMPTY** when the downstream machine is STARVED;
- **FLOWING** otherwise, with no number;
- **UNKNOWN** when a neighbour has no live data.

The tooltip states the inference. Real buffer telemetry (a BufferLevel event through publishers, adapters, store and WebSocket) is deferred and recorded in the PROGRESS.md backlog.

### Picking and performance

- **Picking:** one invisible box per machine and buffer instead of every mesh. It is cheaper, and open structures (the inspection gantry, the CNC glass) can be hovered anywhere.
- **Instancing:** rollers, parts on conveyors and buffer stacks are instanced.
- **Shadows** only for lines up to `render.shadowMaxMachines` machines.
- **Disposal:** `ResourceTracker` and `ListenerRegistry` count what the engine holds, so `dispose()` is verifiable.

## Consequences

- Any line config renders without frontend changes. A new machine kind renders as a generic box until it gets a builder.
- Buffer visuals are honest but coarse. A FLOWING buffer may in fact be almost full or almost empty.
- The engine's tests check the scene graph and materials, not pixels. Pixel-level checks come from the Playwright screenshots (`uv run poe screenshots`), which are reviewed by eye.
- The new optional `layout` field on machines is ignored by the simulator.

## Alternatives considered

- **react-three-fiber:** excluded by CLAUDE.md, and it would tie the engine to React.
- **glTF models:** rejected; the brief and the plan ask for geometry in code, and part addressability would depend on the model's naming.
- **Outline post-processing (OutlinePass):** looks nicer, but costs a full-screen pass per frame and an extra render target.
- **Estimating buffer counts from speeds:** rejected; it would show invented numbers.

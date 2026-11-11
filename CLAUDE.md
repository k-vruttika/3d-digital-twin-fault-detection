# LineTwin: Multi-Source 3D Digital Twin with Reliable Early Fault Warning

This file is the permanent project context for Claude Code. Read it fully, and read `docs/PROGRESS.md`, at the start of every session.

## 1. Company brief (must always stay satisfied)

> Machine Digital Twin Dashboard: A 3D conveyor or CNC station on desktop shows running, idle and fault states driven by a JSON/CSV file or mock data server. It tests architecture, data handling and connecting the engine to outside systems.

**Minimal mode rule:** at every phase it must be possible to run the 3D twin from a plain JSON/CSV file or the mock server, with NO database, broker or ML running. Never break this. It is tested in CI.

## 2. What we are building

A desktop 3D digital twin of a small production line (conveyor → CNC → inspection station → output conveyor, with buffers between them). It:

- ingests data from interchangeable sources: JSON/CSV file replay, mock REST/WebSocket server, MQTT, OPC UA
- validates, deduplicates and reorders messy data, and stores everything as events (time travel / replay)
- is bidirectional: operators send commands (stop, set speed, maintenance) back to the machine, and can run what-if simulations
- predicts faults early (state at +10 s / +30 s, time-to-fault), with calibrated confidence, conformal prediction sets, anomaly detection for unseen faults, explanations highlighted on 3D parts, and live drift detection
- is evaluated rigorously for robustness under noise, missing data, operating-condition shift and unseen fault types, and validated on real public time-series datasets

**Research question:** How reliably can a data-driven digital twin warn of machine faults when its input data contains noise or missing values, when the machine operates under conditions different from training, or when a fault type was never seen in training, and does the reported confidence remain trustworthy in these cases?

## 3. Architecture

```
SIMULATOR (4-machine line, physics-based degradation, snapshot/restore/fast-forward)
  ├─ File export (CSV/JSON/Parquet) ─┐
  ├─ Mock REST/WS machine server     ├─► ADAPTER LAYER ─► Normalizer ─► Validator ─► Dedup/Reorder ─► EVENT BUS
  ├─ MQTT (native Mosquitto)         │   (+ Chaos wrapper)               │ invalid → quarantine         │
  └─ OPC UA server (asyncua)        ─┘                                                                  │
                                                                                                        ├─► EVENT STORE (PostgreSQL | SQLite)
                                                                                                        ├─► TWIN ENGINE (state machines, line propagation)
                                                                                                        ├─► INFERENCE SERVICE (models, calibration, conformal,
                                                                                                        │                      anomaly, attributions, drift)
                                                                                                        └─► WS GATEWAY ─► FRONTEND (TwinEngine Three.js + React dashboard)

COMMAND PATH: Dashboard ─► POST /commands ─► safety rules ─► Command bus ─► active adapter.send_command ─► machine ─► ack
WHAT-IF:      Dashboard ─► POST /whatif ─► simulator snapshot ─► fast-forward copy ─► model ─► projected trajectory
```

## 4. Tech stack (do not add other major dependencies without asking)

- **Python 3.11**, managed with `uv`. FastAPI, Uvicorn, Pydantic v2, pydantic-settings, asyncio
- **Messaging:** Mosquitto installed natively on Windows (config in `config/mosquitto.conf`) + `aiomqtt`; OPC UA via `asyncua`
- **Storage:** PostgreSQL installed natively, via `asyncpg`/SQLAlchemy 2. Plain PostgreSQL features only (B-tree and BRIN indexes, materialized views for downsampling); no TimescaleDB or other extensions. SQLite implementation for minimal mode and tests; Parquet (pyarrow) for datasets
- **ML:** PyTorch, scikit-learn, XGBoost, SHAP, Captum, `river` (ADWIN), MLflow (local file store), ONNX / onnxruntime
- **Frontend:** TypeScript (strict), Vite, React, Three.js (used directly, NOT react-three-fiber), Chart.js
- **Quality:** ruff, mypy, pytest, pytest-asyncio, Vitest, ESLint, pre-commit, GitHub Actions
- **Ops:** Prometheus metrics endpoint (`/metrics`), Locust for load testing. Dev tasks via `scripts/dev.ps1` (Windows) and an equivalent `Makefile`

## 5. Repository layout

```
linetwin/
├─ CLAUDE.md
├─ Makefile                  # dev, minimal, test, lint, fmt, sim, train, experiments, loadtest
├─ scripts/dev.ps1           # same targets as the Makefile, for Windows PowerShell
├─ .github/workflows/ci.yml
├─ config/
│  ├─ line.yaml              # line layout, machines, sensors, buffers, part mappings
│  ├─ sources.yaml           # active adapter + per-adapter settings
│  ├─ mosquitto.conf         # local native Mosquitto broker config
│  └─ models/*.yaml          # per-model training configs
├─ common/twin_common/       # shared Pydantic contracts (Reading, Command, ...), config loaders
├─ simulator/twin_sim/       # physics, machines, line, scenarios, publishers (mqtt, opcua, mock server), CLI
├─ backend/twin_backend/
│  ├─ adapters/              # base protocol, file, mock, mqtt, opcua, chaos
│  ├─ ingest/                # normalizer, validator, dedup/reorder, staleness
│  ├─ bus/                   # in-process async pub/sub
│  ├─ storage/               # event store interface, postgres, sqlite, snapshots
│  ├─ engine/                # state machines, line propagation, data quality
│  ├─ commands/              # command bus, safety rules, audit
│  ├─ inference/             # online inference, windows, drift, registry
│  ├─ whatif/
│  └─ api/                   # REST routers, WS gateway, auth
├─ ml/twin_ml/
│  ├─ data/                  # windowing, labels, splits, perturbations
│  ├─ features/
│  ├─ models/                # rf, xgb, lstm, gru, transformer, autoencoder, iforest
│  ├─ train/  calibrate/  conformal/  explain/  evaluate/
│  ├─ datasets/              # metropt3, azure_pdm, cmapss loaders
│  └─ experiments/           # E1–E5, ablation runners, report generators
├─ frontend/src/
│  ├─ engine/                # framework-agnostic TwinEngine (no React imports)
│  ├─ dashboard/             # React components
│  └─ api/                   # REST + WS client
├─ data/                     # sim/, external/ (gitignored)
├─ experiments/results/      # generated reports and plots only
├─ tests/                    # cross-package integration / e2e tests
└─ docs/
   ├─ PROGRESS.md            # updated at end of every phase
   ├─ architecture.md  api.md  simulator.md  experiments.md  setup_windows.md
   └─ adr/                   # architecture decision records (NNNN-title.md)
```

## 6. Canonical data contracts (defined once in `twin_common`)

**Reading** (schemaVersion 2):
`schemaVersion, machineId, seq (int, per machine), ts (ISO-8601 UTC), state, speed (RPM), temp (°C), vib (mm/s RMS), load (%), current (A), runTime (s), quality (OK | FILLED | LATE | SUSPECT)`

**MachineState:** `RUNNING, IDLE, WARNING, FAULT, STARVED, BLOCKED, MAINTENANCE, OFFLINE, STALE`

**Prediction:** `machineId, ts, horizonS, probs {NORMAL, WARNING, FAULT}, calibrated (bool), conformalSet [classes], ttfS, ttfInterval [lo, hi], anomalyScore, attributions [{sensor, part, weight}], reliability (OK | DEGRADED), modelVersion`

**Command:** `commandId, idempotencyKey, machineId, type (START | STOP | SET_SPEED | SET_LOAD | SCHEDULE_MAINTENANCE | RESET), params, issuedBy, ts`
**CommandAck:** `commandId, status (ACCEPTED | REJECTED | EXECUTED | TIMEOUT), reason, ts`

**MQTT topics:** `plant/line1/{machineId}/telemetry`, `.../state`, `.../cmd`, `.../cmd/ack`

ML target mapping: RUNNING/IDLE/STARVED/BLOCKED/MAINTENANCE → NORMAL (machine health target); WARNING → WARNING; FAULT → FAULT. Document any change in an ADR.

## 7. Conventions

- Type hints everywhere; mypy clean on `common`, `backend`, `simulator`. Pydantic for all external data.
- No magic constants: thresholds, rates, windows, horizons come from YAML config.
- All randomness seeded and reproducible. Same seed → identical output.
- Async I/O in the backend; no blocking calls in the event loop (use executors for ML).
- Every module gets unit tests; every phase adds at least one integration test.
- Logging: structured (JSON) via `structlog`, never `print` in library code.
- Frontend `engine/` must not import React; pure state→visual mapping functions are unit tested.
- Generated results go to `experiments/results/` and are produced by scripts, never typed by hand.
- Do not use Docker. All services run natively on Windows.

## 8. Working agreement for Claude

1. At session start: read this file and `docs/PROGRESS.md`. Work only on the phase you are given.
2. Before coding, present a short plan (files to create/change, key design choices, open questions). Wait for approval if anything is ambiguous or deviates from this file.
3. Implement in small steps. Run `make lint` and `make test` before declaring done.
4. At phase end: update `docs/PROGRESS.md` (what was built, how to run it, known limitations, deviations), add ADRs for significant decisions, and print a summary plus a suggested commit message. Then stop.
5. **Never fabricate experiment numbers, citations, or dataset contents.** Results only come from running code. If something can't be run here (e.g. a dataset is missing), say so and leave a clearly marked TODO.
6. Do not download datasets automatically; loaders expect files in `data/external/<name>/` and fail with a helpful message.
7. Never break minimal mode (section 1).

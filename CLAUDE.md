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
  ├─ MQTT (Mosquitto)                │   (+ Chaos wrapper)               │ invalid → quarantine         │
  └─ OPC UA server (asyncua)        ─┘                                                                  │
                                                                                                        ├─► EVENT STORE (TimescaleDB | SQLite)
                                                                                                        ├─► TWIN ENGINE (state machines, line propagation)
                                                                                                        ├─► INFERENCE SERVICE (models, calibration, conformal,
                                                                                                        │                      anomaly, attributions, drift)
                                                                                                        └─► WS GATEWAY ─► FRONTEND (TwinEngine Three.js + React dashboard)

COMMAND PATH: Dashboard ─► POST /commands ─► safety rules ─► Command bus ─► active adapter.send_command ─► machine ─► ack
WHAT-IF:      Dashboard ─► POST /whatif ─► simulator snapshot ─► fast-forward copy ─► model ─► projected trajectory

SERVICES:     Docker Compose (uv run poe up): TimescaleDB (PostgreSQL 16) + Mosquitto 2, ports on 127.0.0.1 only.
              Minimal mode (file / mock server, SQLite) needs none of them.
```

## 4. Tech stack (do not add other major dependencies without asking)

- **Python 3.11**, managed with `uv`. FastAPI, Uvicorn, Pydantic v2, pydantic-settings, asyncio
- **Messaging:** Mosquitto 2 via Docker Compose (shared settings in `config/mosquitto.d/`, container entry `config/mosquitto.docker.conf`; native fallback entry `config/mosquitto.conf`) + `aiomqtt`; OPC UA via `asyncua`
- **Storage:** PostgreSQL 16 with the TimescaleDB extension via Docker Compose (hypertables and continuous aggregates allowed), via `asyncpg`/SQLAlchemy 2. SQLite implementation for minimal mode and tests; Parquet (pyarrow) for datasets
- **ML:** PyTorch (CPU build: PyTorch CPU index on Linux/Windows, PyPI on macOS), scikit-learn, XGBoost (`xgboost-cpu` on Linux/Windows), SHAP, Captum, `river` (ADWIN), MLflow (`mlflow-skinny`, local file store `mlruns/`), ONNX / onnxruntime (ADR 0020)
- **Frontend:** TypeScript (strict), Vite, React, Three.js (used directly, NOT react-three-fiber), Chart.js
- **Quality:** ruff, mypy, pytest, pytest-asyncio, Vitest, ESLint, pre-commit, GitHub Actions; frontend dev tools `openapi-typescript` (API types, ADR 0017), Playwright with `@playwright/test` (screenshots, dashboard e2e) and Testing Library + jsdom (component tests)
- **Ops:** Prometheus metrics endpoint (`/metrics`), Locust for load testing. Dev tasks are poethepoet (`poe`) tasks in the root `pyproject.toml`, run as `uv run poe <task>`, identical on Windows, Linux and macOS

## 5. Repository layout

```
linetwin/
├─ CLAUDE.md
├─ pyproject.toml            # uv workspace + poe tasks: dev, minimal, test, lint, fmt, sim, up/down, train, experiments, loadtest
├─ docker-compose.yml        # services: TimescaleDB (PostgreSQL 16) + Mosquitto 2, 127.0.0.1 only
├─ .env.example              # service credentials/ports (copy to .env, gitignored); read by Compose and TwinSettings
├─ .github/workflows/ci.yml
├─ config/
│  ├─ line.yaml              # line layout, machines, sensors, buffers, part mappings
│  ├─ sources.yaml           # active adapter + per-adapter settings
│  ├─ storage.yaml           # event store backend (sqlite | timescale), writer, snapshots, policies
│  ├─ engine.yaml            # state transitions, data-quality thresholds, alert kinds
│  ├─ api.yaml               # bind address, CORS, WebSocket buffers
│  ├─ commands.yaml          # safety rules (STOP always allowed), limits, idempotency
│  ├─ inference.yaml         # Phase 11 live inference: deployed model + version, gate, workers (off by default)
│  ├─ whatif.yaml            # Phase 11 what-if: projection lengths, history, ghost horizons (needs the source capability `whatif`)
│  ├─ mosquitto.d/           # Mosquitto settings shared by container and native fallback
│  ├─ mosquitto.docker.conf  # Mosquitto entry config inside the container
│  ├─ mosquitto.conf         # Mosquitto entry config for the native fallback
│  ├─ datasets/              # Phase 10 facts about the public datasets: metropt3.yaml (failure list + source), azure_pdm.yaml, metropt3_replay.yaml (replay demo window)
│  └─ models/                # ML: dataset.yaml (+ dataset_e4/_aug), eval.yaml, robustness.yaml (Phase 8), e5.yaml (Phase 9, pre-registered) + e5_followup.yaml (exploratory), real.yaml (Phase 10, pre-registered) + real_<dataset>.yaml (real-data windows/labels/splits), one <model>.yaml per model/baseline, iforest.yaml + autoencoder.yaml (anomaly detectors), deploy.yaml (Phase 11 deploy bundles, ADR 0029)
├─ common/twin_common/       # shared Pydantic contracts (Reading, Command, ...), config loaders
├─ simulator/twin_sim/       # physics, machines, line, scenarios, publishers (mqtt, opcua, mock server), CLI
├─ backend/twin_backend/
│  ├─ adapters/              # base protocol, file, mock, mqtt, opcua, chaos
│  ├─ ingest/                # normalizer, validator, dedup/reorder, staleness
│  ├─ bus/                   # in-process async pub/sub
│  ├─ storage/               # event store interface, timescale, sqlite, writer, snapshots, replay, export, migrations/
│  ├─ engine/                # state machines, line propagation, data quality
│  ├─ commands/              # command bus, safety rules, audit
│  ├─ inference/             # Phase 11: service (bus -> worker processes -> Predictions, alert conditions), workers, alarms (evaluated rule)
│  ├─ whatif/                # Phase 11: what-if service (fork via the source, safety rules, scoring in its own worker pool)
│  └─ api/                   # REST routers, WS gateway, auth
├─ ml/twin_ml/
│  ├─ data/                  # windowing, labels, splits, build (python -m twin_ml.data build), perturbations
│  ├─ features/              # channel schema (ADR 0028), allowed inputs, normalisation, window summaries (no twin_sim, no torch)
│  ├─ baselines/             # persistence, static threshold; rules.py (real-data threshold, rolling z-score, persistence, error-log rules)
│  ├─ models/                # rf, xgb, lstm, gru, transformer; anomaly.py (normal windows, thresholds, iforest), autoencoder.py
│  ├─ explain/               # explain() (Phase 11 API), trees.py (TreeSHAP, Saabas), ig.py (Integrated Gradients), aggregate.py (→ sensor → part)
│  ├─ train/  calibrate/  conformal/  evaluate/
│  ├─ datasets/              # metropt3, azure_pdm loaders, real-data build and audit, replay.py (minimal-mode replay CSV) (python -m twin_ml.datasets build|audit|replay-csv; no twin_sim)
│  ├─ serve/                 # Phase 11: deploy bundles + registry (bundle.py), buffers + gate, calib (exported maps), live.py, worker process, latency report (python -m twin_ml.serve deploy|worker|latency)
│  └─ experiments/           # E1–E5, ablation runners, report generators (E5: e5.py / e5_worker.py, explain_eval.py, report_e5.py; Phase 10: real.py / real_worker.py, real_eval.py, real_explain.py, report_real.py + report_real_results.py)
├─ frontend/
│  ├─ index.html             # the React dashboard (uv run poe frontend → /)
│  ├─ dev.html               # full-screen 3D twin dev page (uv run poe frontend → /dev.html)
│  ├─ playwright.config.ts   # Playwright e2e projects (run by pytest via twin_backend/e2e_stack.py)
│  ├─ e2e/                   # Playwright specs: minimal, dev (services), dashboard screenshots
│  ├─ scripts/               # screenshots.mjs (uv run poe screenshots)
│  └─ src/
│     ├─ engine/             # framework-agnostic TwinEngine (no React imports); __fixtures__/ generated
│     ├─ dashboard/          # React dashboard: state/ (model, session, replay), components/, charts/
│     ├─ api/                # REST + WS client; openapi.json + schema.gen.ts generated (poe api-docs)
│     └─ dev/                # dev page entry and synthetic data mode
├─ data/                     # sim/, ml/ (built datasets), external/ (gitignored)
├─ experiments/results/      # generated reports and plots only
├─ tests/                    # cross-package integration / e2e tests
└─ docs/
   ├─ PROGRESS.md            # updated at end of every phase
   ├─ architecture.md  api.md  frontend.md  simulator.md  experiments.md  setup.md
   └─ adr/                   # architecture decision records (NNNN-title.md)
```

## 6. Canonical data contracts (defined once in `twin_common`)

**Reading** (schemaVersion 2):
`schemaVersion, machineId, seq (int, per machine), ts (ISO-8601 UTC), state, speed (RPM), temp (°C), vib (mm/s RMS), load (%), current (A), runTime (s), quality (OK | FILLED | LATE | SUSPECT)`
Sensor values (speed, temp, vib, load, current) are nullable: null = missing. Partially missing readings are kept with quality SUSPECT; readings with no sensor value are quarantined (ADR 0007).

**MachineState:** `RUNNING, IDLE, WARNING, FAULT, STARVED, BLOCKED, MAINTENANCE, OFFLINE, STALE`

**Prediction:** `machineId, ts, horizonS, probs {NORMAL, WARNING, FAULT}, calibrated (bool), conformalSet [classes], ttfS, ttfInterval [lo, hi], anomalyScore, attributions [{sensor, part, weight}], reliability (OK | DEGRADED), modelVersion`; optional since Phase 11 (ADR 0030): `alarm` (the evaluated supervised alarm, primary horizon), `anomalyFlag`, `anomalyThreshold`, `attributionsTs` (window the attributions explain)

**Command:** `commandId, idempotencyKey, machineId, type (START | STOP | SET_SPEED | SET_LOAD | SCHEDULE_MAINTENANCE | RESET), params, issuedBy, ts`
**CommandAck:** `commandId, status (ACCEPTED | REJECTED | EXECUTED | TIMEOUT), reason, ts`
**StateChange:** `machineId, state, previousState, ts, reason` (ts on the event-time axis, ADR 0010)
**Alert:** `alertId, machineId, ts, severity (INFO | WARNING | CRITICAL), kind, message, source, active, acknowledged, ackedBy, ackedAt, rev` (one append-only event per lifecycle transition, ISA-18.2; current alert = highest rev; ADR 0011). Inference kinds (source `inference`, ADR 0030): `predicted_warning`, `predicted_fault`, `unusual_behaviour` (autoencoder only, cause unknown, no part), `model_drift`

**ScenarioRequest / ScenarioAck** (simulation only, ADR 0015): `requestId, action (INJECT_FAULT | LOAD_SCENARIO), machineId, faultType, progressionS, scenario, seed, requestedBy, source, ts` / `requestId, status, reason, ts`

**MQTT topics:** `plant/line1/{machineId}/telemetry`, `.../state`, `.../cmd`, `.../cmd/ack`; line level (simulation only): `plant/line1/scenario`, `plant/line1/scenario/ack`

ML target mapping: RUNNING/IDLE/STARVED/BLOCKED/MAINTENANCE → NORMAL (machine health target); WARNING → WARNING; FAULT → FAULT. Document any change in an ADR.

## 7. Conventions

- Type hints everywhere; mypy clean on `common`, `backend`, `simulator`. Pydantic for all external data.
- No magic constants: thresholds, rates, windows, horizons come from YAML config.
- All randomness seeded and reproducible. Same seed → identical output.
- Async I/O in the backend; no blocking calls in the event loop (use executors for ML).
- Every module gets unit tests; every phase adds at least one integration test.
- Logging: structured (JSON) via `structlog`, never `print` in library code.
- Frontend `engine/` must not import React; pure state→visual mapping functions are unit tested.
- Generated results go to `experiments/results/` and are produced by scripts, never typed by hand. They are **committed**: reports (`*.md`), JSON summaries (`*.json`), plots (`*.png`, `*.svg`) and small tables (`*.csv`). Large or regenerable artefacts (Parquet/Arrow/NumPy dumps, model checkpoints, ONNX files, databases, `mlruns/`, `raw/` folders) are ignored in `.gitignore`. No file there may exceed 5 MB; `tests/test_results_policy.py` enforces the types and the size limit. Datasets stay in `data/`, which is never committed.
- Services (TimescaleDB, Mosquitto) run via Docker Compose (`uv run poe up`). Native installs are a documented fallback. Minimal mode must never require Docker.
- Developer tasks are added **only** as poe tasks in the root `pyproject.toml` (`[tool.poe.tasks]`). Use cross-platform commands only (tools from the uv environment, `uv`, `npm --prefix frontend`, `docker compose`); no Makefile, shell scripts, or bash/PowerShell syntax. CI calls the same poe tasks on Ubuntu, Windows and macOS.

## 8. Working agreement for Claude

1. At session start: read this file and `docs/PROGRESS.md`. Work only on the phase you are given.
2. Before coding, present a short plan (files to create/change, key design choices, open questions). Wait for approval if anything is ambiguous or deviates from this file.
3. Implement in small steps. Run `uv run poe lint` and `uv run poe test` before declaring done.
4. At phase end: update `docs/PROGRESS.md` (what was built, how to run it, known limitations, deviations), add ADRs for significant decisions, and print a summary plus a suggested commit message. Then stop.
5. **Never fabricate experiment numbers, citations, or dataset contents.** Results only come from running code. If something can't be run here (e.g. a dataset is missing), say so and leave a clearly marked TODO.
6. Do not download datasets automatically; loaders expect files in `data/external/<name>/` and fail with a helpful message.
7. Never break minimal mode (section 1).
8. Never add "Co-Authored-By", "Generated with Claude", or any other AI attribution to commit messages, suggested commit messages, PR descriptions, code comments, or docs.

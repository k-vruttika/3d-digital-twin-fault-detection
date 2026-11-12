# LineTwin progress

## Phase 2: Adapters and ingestion (done, 2026-11-12)

### What was built

- **Shared code (`twin_common`):**
  - `Reading` sensor fields are nullable (`null` = missing); `schemaVersion` stays 2. Helpers: `sensor_values()`, `missing_sensors()`.
  - New `StateChange` contract (bus topic `states`).
  - New enums: `Capability`, `AdapterStatus`, `QuarantineReason` (including `NO_SENSOR_DATA`).
  - `units.py`: unit-conversion table into the catalog units.
  - `opcua_names.py`: OPC UA browse names, event type and method table, shared by server and adapter.
  - `aio.py`: `run()` and `new_event_loop()` use a selector loop on Windows; `require_selector_loop()` gives a clear error otherwise ([ADR 0008](adr/0008-windows-selector-event-loop.md)).
- **`config/sources.yaml` / `SourcesConfig`:**
  - A field mapping per adapter (`fields`, `units`, `drop_unmapped`).
  - New required sections: `reconnect`, `ingest`, `bus`, `chaos`, plus `command_timeout_s`.
  - Per-adapter settings:
    - `mqtt.qos: 1` and `mqtt.session_expiry_s`;
    - `opcua.publishing_interval_ms`, `event_queue_size` and `watchdog_s`;
    - `file.format` accepts `csv | jsonl | json`, plus `start_offset_s`;
    - `mock.ws_path` is now `/stream`, plus `history_path`, `command_path`, `rest_poll_s` and `history_limit`.
  - The default replay file is now `data/sim_quick/run_0000.csv`; the old default never existed.
  - Validation rules: STALE before OFFLINE, `seq_reset_threshold <= dedup_window`, and a JSONL quarantine needs a path.
- **Backend (`twin_backend`):**
  - **`adapters/`:**
    - `base.py`: `SourceAdapter` protocol, `RawMessage`, `AdapterHealth`, `BaseAdapter` (I/O task, bounded queue, re-cancelling `stop()`) and `Backoff`.
    - Adapters: `file.py` (standard library only), `mock.py` (WebSocket, `/history` backfill, REST fallback), `mqtt.py` (aiomqtt, MQTT v5 persistent session, commands via `cmd`/`cmd/ack`) and `opcua.py` (ReadingEvent subscription, watchdog, method calls).
    - `chaos.py`: seeded drop, duplicate, delay, reorder and corrupt faults.
    - `registry.py` (lazy imports) and `manager.py` (`SourceManager`: active adapter, pump and ticker, hot `switch()`, `send_command`).
  - **`ingest/`:** normalizer, validator (missing values, ranges), quarantine (memory or JSONL), dedup, reorder buffer, staleness monitor, Prometheus metrics, and the pipeline that ties them together ([ADR 0007](adr/0007-adapters-and-ingest.md)).
  - **`bus/`:** `EventBus` protocol and `InMemoryEventBus` (bounded per-subscriber queues, drop-oldest) ([ADR 0006](adr/0006-event-bus-interface.md)).
  - **`clock.py`:** `SystemClock` and `FakeClock`.
- **Simulator publishers (`twin_sim.publishers`, never imported by `twin_sim` itself):**
  - `session.py`: `SimSession` wraps the RealtimeRunner, with history and listeners.
  - `mock_server.py`: FastAPI app with `/info`, `/state`, `/history`, `WS /stream` and `POST /command`.
  - `mqtt.py`: telemetry plus a retained state topic, commands and acks, a buffered publish loop and reconnect.
  - `opcua.py`: objects, variables and methods per machine, plus one `ReadingEventType` event per reading.
  - `http.py`: uvicorn started inside the running loop.
- **CLI and poe tasks:**
  - New subcommands `mock-server`, `mqtt` and `opcua`, which take the same `--line`, `--scenario`, `--seed`, `--run-index`, `--speed`, `--duration` and `--history-size` options as `run`. Their defaults come from `sources.yaml`.
  - New poe tasks: `sim-mock`, `sim-mqtt`, `sim-opcua`.
  - `run` now starts through `aio.run`.
  - `minimal-data-check` also runs `tests/test_phase2_minimal.py`.
- **Config:** the load catalog maximum was raised from 120 to 130 % in `line.yaml` and `line_single.yaml`, after the sensor-range check (approved). The line.yaml comment now describes the configurable out-of-range handling.
- **Docs:**
  - [api.md](api.md) is new: Reading, the mock server, MQTT topics and the OPC UA address space.
  - [architecture.md](architecture.md) has a new ingest section; [simulator.md](simulator.md) a new §11 on the publishers.
  - ADRs [0006](adr/0006-event-bus-interface.md), [0007](adr/0007-adapters-and-ingest.md) and [0008](adr/0008-windows-selector-event-loop.md).
  - CLAUDE.md §6 notes the nullable sensor fields.
- **Dependencies (approved):**
  - simulator: fastapi, uvicorn, websockets, aiomqtt, asyncua;
  - backend: asyncua, websockets, httpx, prometheus-client.

### How to run

```text
uv run poe sim-quick                      # replay file for the default file adapter
uv run poe minimal-data-check             # file -> adapter -> ingest -> bus, no services
uv run poe sim-mock                       # mock server on http://127.0.0.1:8100
uv run poe up; uv run poe sim-mqtt        # MQTT publisher (needs the broker)
uv run poe sim-opcua                      # OPC UA server on opc.tcp://127.0.0.1:4840/linetwin/
uv run poe sim-opcua --line config/line_single.yaml --scenario config/scenarios/quick_single.yaml
```

The backend entry point (API, `/metrics`, hot switch over HTTP) arrives in Phase 4. Until then, `SourceManager` is used from code and tests.

### Tests

pytest now runs 416 tests, up from 226. The most important ones:

- **Identical sequences** (`tests/test_phase2_integration.py`): one seeded 120 s run through File JSONL, File CSV, Mock, OPC UA and MQTT gives identical canonical Reading sequences. The MQTT case asserts QoS 1 and that the publisher dropped nothing.
- **Commands** through MQTT, OPC UA and Mock reach the simulator (SET_SPEED applied, EXECUTED) and return acks (an invalid RESET is REJECTED with the simulator's reason).
- **OPC UA:**
  - the adapter reconnects after a server restart;
  - the single-station line works;
  - a partially missing reading travels as NaN and arrives as `null` with SUSPECT.
- **Chaos:**
  - with 10 % duplicates and reordering, the output equals the simulated run exactly (no LATE readings, no gaps);
  - with 5 % corruption, every corrupted message is quarantined with a reason and nothing else is lost;
  - same seed gives the same output, and the fault rates match.
- **Mock adapter:**
  - backfill closes the startup gap;
  - a server restart mid-stream loses no seq;
  - REST fallback works while the WebSocket is unavailable;
  - commands, a server that is down, and the single-station line are covered.
- **Hot switch** from file to a mock server streaming the same run is seamless (overlap deduplicated). A failing switch keeps the current source.
- **Unit tests:**
  - normalizer: mapping and units;
  - validator: every quarantine reason, partially missing readings, suspect vs quarantine mode;
  - dedup and source reset; reorder (in order, gap filled, window timeout, LATE, initial hold);
  - staleness at exact fake-clock times; bus fan-out and drop-oldest; backoff; clock;
  - file replay: CSV/JSONL/JSON, pacing, speed, offset, loop, missing-file hint;
  - pipeline metrics.
- **Contracts:** round trips with null sensors; missing keys are accepted; non-sensor fields stay required.
- **Loop type:** on Windows, async tests and `aio.run()` run on a SelectorEventLoop, and a Proactor loop is rejected.
- **Sensor ranges** (`simulator/tests/test_sim_sensor_ranges.py`): analytic worst cases per machine kind, plus an empirical sweep (quick, sensor_faults, e5_lubrication_loss, cnc_stop, quick_single; 3 seeds × 1800 s) and a commanded 120 % overload. Everything lies inside the catalog.
- **Minimal mode** (`tests/test_phase2_minimal.py`): in a fresh interpreter, the `sim-quick` CSV and JSONL go through `SourceManager` with no broker, DB, HTTP or ML module imported.

### Verified here (Windows 11)

| Check | Result |
|---|---|
| `uv run poe lint` | passes |
| `uv run poe test`, services up, `LINETWIN_REQUIRE_SERVICES=1` | 416 passed |
| `uv run poe test`, after `uv run poe down` | 412 passed, 4 skipped (MQTT/DB) with the start hint |
| `uv run poe minimal-data-check` | passes |
| `mock-server` | smoke-run |
| `mqtt` | streamed through its adapter (full line) |
| `opcua` | streamed through its adapter (full and single-station line) |
| OPC UA test group | passed 8 runs in a row |
| Mock and source-manager tests | passed 5 runs in a row |

### Known limitations

- **Startup delay:** each machine's first readings wait up to `lateness_window_s` (2 s) after start or a detected source restart, so early out-of-order messages are not flagged LATE.
- **OPC UA outages lose data:** readings fired while the adapter is disconnected are lost and show up as a gap. Mock (history backfill) and MQTT (persistent session) recover short outages.
- **No MQTT broker-outage test:** stopping the shared Docker broker from a test is not done. Publisher buffering and the adapter's persistent session are covered by code review only.
- **Dropout rows in CSV:** the CSV export contains sensor-dropout rows with every sensor NaN; they are quarantined as `NO_SENSOR_DATA`. JSONL omits them.
- **Looped file replay** restarts `runTime` on every pass.
- **Load and CSV order:** the file adapter loads the whole file into memory and sorts CSV rows by (ts, line order). That is fine for simulator exports; very large files would need streaming.
- **Metrics not yet served:** they are collected per pipeline, but nothing serves `/metrics` before Phase 4.
- **OPC UA test speed:** the OPC UA tests take 3 to 8 s each, mostly asyncua connection setup and teardown.
- **CI:** the new jobs (Ubuntu with services, Windows and macOS without) have not run on GitHub yet; there is no remote.

### Deviations from the plan / CLAUDE.md

- **`Reading` contract:** sensor fields are nullable while `schemaVersion` stays 2 (requested in review). CLAUDE.md §6 was updated.
- **`QuarantineReason.NO_SENSOR_DATA`** was added. Sensor NaN/inf now means "missing", not `NON_FINITE`, which now applies only to `runTime`.
- **Load catalog max** raised from 120 to 130 % (approved).
- **New files outside the CLAUDE.md layout:**
  - `common/twin_common/aio.py`, `units.py` and `opcua_names.py`;
  - `backend/twin_backend/clock.py`;
  - `simulator/twin_sim/publishers/` (CLAUDE.md §5 lists publishers under the simulator without a folder).
- **Root `conftest.py`** now also sets the event loop type, through pytest-asyncio's `pytest_asyncio_loop_factories` hook; overriding `event_loop_policy` is deprecated in pytest-asyncio 1.4.
- **Two workarounds:**
  - A pytest filter ignores Starlette's "use httpx2" TestClient deprecation warning.
  - `OpcUaServer.stop()` closes asyncua's listening socket first. Otherwise a client reconnecting during shutdown keeps `wait_closed()` (Python 3.11.15) waiting forever.
- **Python 3.11 cancellation:** `BaseAdapter.stop()` re-cancels its task until it ends, because on Python 3.11 `asyncio.wait_for` (used inside asyncua) can swallow a cancellation. Both of these hangs were seen in testing and are fixed.

## Infrastructure: Docker Compose services (2026-11-12)

### What was built

- **`docker-compose.yml`** (project `linetwin`). Both services publish their ports on 127.0.0.1 only.
  - **`timescaledb`:** `timescale/timescaledb:2.31.0-pg16`, i.e. PostgreSQL 16 with TimescaleDB. 2.32.0 was published the same day, so I took the previous release. Named volume `timescale-data`, health check `pg_isready` over TCP.
  - **`mosquitto`:** `eclipse-mosquitto:2.1.2-alpine`, matching the native 2.1.2 (2.1.x is only published as alpine tags). Named volume `mosquitto-data`, health check `mosquitto_pub`.
- **`.env.example`** (committed; `.env` stays gitignored): `LINETWIN_PG_DB`, `LINETWIN_PG_USER`, `LINETWIN_PG_PASSWORD`, `LINETWIN_PG_PORT`. Docker Compose and `TwinSettings` both read it.
- **Mosquitto config split:**
  - `config/mosquitto.d/common.conf`: shared settings;
  - `config/mosquitto.docker.conf`: container entry, all interfaces inside the container;
  - `config/mosquitto.conf`: native entry, `listener 1883 127.0.0.1` plus `include_dir`.
- **poe tasks:** `up` (starts and waits until healthy), `down`, `reset` (also deletes volumes), `logs`, `ps`. All are plain `docker compose` commands.
- **`TwinSettings`:** new fields `pg_host`, `pg_port`, `pg_db`, `pg_user` and `pg_password` (SecretStr), and new properties `pg_dsn` and `database_url`. Reads `.env`; real environment variables take precedence.
- **Backend dependencies:** `asyncpg`, and `aiomqtt` (which brings `paho-mqtt`).
- **Test fixtures** in `twin_backend/testing.py` (plugin registered in the root `conftest.py`): `mqtt_broker` and `timescale_db`.
  - When a service can't be reached, they skip with "… not reachable at … - start services with: uv run poe up".
  - With `LINETWIN_REQUIRE_SERVICES=1` they fail instead.
  - The pytest marker `services` is registered.
- **Tests:**
  - `tests/test_compose_services.py`: an MQTT publish/subscribe round trip, and a PostgreSQL 16 check that the `timescaledb` extension is available and installed;
  - `common/tests/test_settings.py`: settings defaults, `.env` and environment overrides, URL escaping, and `.env.example` matching the defaults.
- **CI:** the Ubuntu job runs `uv run poe up` before the tests and sets `LINETWIN_REQUIRE_SERVICES=1`. Windows and macOS don't start services, and the service tests are skipped there.
- **Docs:**
  - **CLAUDE.md:** architecture diagram, §4 storage and messaging, §5 layout, and the §7 rule (services via Docker Compose, native as the fallback, minimal mode never requires Docker; the poe rule now allows `docker compose`).
  - **ADRs:** [ADR 0005](adr/0005-docker-compose-services.md) is new; [ADR 0002](adr/0002-native-services-no-docker.md) is marked superseded.
  - **Setup guide:** `docs/setup_windows.md` was renamed to [setup.md](setup.md). Docker Desktop (Windows, macOS) and Docker Engine (Linux) are the main path; native Mosquitto and PostgreSQL with TimescaleDB are the fallback.
  - **architecture.md:** updated.

### How to run

```text
uv run poe up      # start + wait until healthy (first run downloads the images)
uv run poe ps      # status
uv run poe test    # service tests run when the services are up, otherwise they are skipped
uv run poe down    # stop (data kept);  uv run poe reset  # stop and delete volumes
```

### Verified here (Windows 11, Docker 29.9, Compose v5.7)

- `uv run poe up`: both services healthy. The first run spent about 11 minutes downloading the images; later starts take seconds.
- **With services up:** `uv run poe lint` passes, and `uv run poe test` with `LINETWIN_REQUIRE_SERVICES=1` gives 226 passed (both service tests ran).
- **After `uv run poe down`:** `uv run poe lint` passes, and `uv run poe test` gives 224 passed and 2 skipped with the start hint.
- With services down, `LINETWIN_REQUIRE_SERVICES=1` makes the service tests error, as intended.
- `uv run poe reset` followed by `up`: on a fresh volume the extension is recreated and the service tests pass. `uv run poe logs` works.

### Known limitations

- **Port 5432 is taken on this machine** by a native `postgres` process (PID at the time: 6716, not a visible Windows service), so Docker could not bind it. A local `.env` (not committed) sets `LINETWIN_PG_PORT=5433`. If that native PostgreSQL is no longer needed, stop it and delete the `.env` line to go back to 5432.
- **Mosquitto 2.1 quirk:** a per-listener option such as `protocol` in a file without a listener makes mosquitto 2.1 create an extra default listener, which collides on 1883. `common.conf` therefore contains no per-listener options; a comment there explains why.
- **The native fallback configs were not run here.** `mosquitto -c config/mosquitto.conf` and a native PostgreSQL with TimescaleDB are documented but untested, because the native Mosquitto service is disabled.
- The Ubuntu CI job with services has not run on GitHub yet (there is no remote).

## Tooling: poethepoet task runner (2026-11-12)

### What changed

- **One OS-neutral task runner.** The `Makefile` and `scripts/dev.ps1` were removed. Every developer task is now a [poethepoet](https://poethepoet.natn.io/) task in the root `pyproject.toml` (`[tool.poe.tasks]`), run as `uv run poe <task>`. The same commands work on Windows, Linux and macOS. See [ADR 0004](adr/0004-poe-task-runner.md).
- **Tasks:**
  - kept from before: `install`, `lint`, `fmt`, `test`, `dev`, `minimal` (`dev` and `minimal` are still stubs that exit with code 1), `sim`, `sim-quick`, `sim-quick-single`, `sim-plots`;
  - new, moved out of CI: `minimal-data-check` and `build-frontend`;
  - the old `help` target is replaced by `uv run poe`, which lists every task with its help text.
- **No shell syntax.** Tasks run without a shell and call only tools from the uv environment, `uv`, and `npm --prefix frontend`. The environment variable for `minimal-data-check` is set through poe's `env` option.
- **CI:** one job with an OS matrix (`ubuntu-latest`, `windows-latest`, `macos-latest`, `fail-fast: false`). Every check step is `uv run --frozen poe <task>`: lint, test, minimal-data-check, build-frontend.
- **CLAUDE.md:** §4, §5 and §8 now refer to poe. §7 has a new rule: developer tasks are added only as poe tasks, using cross-platform commands only.
- **Docs:** `setup_windows.md`, `simulator.md` and `config/line_single.yaml` use `uv run poe …`. The CLI examples in `simulator.md` no longer use PowerShell line continuations.
- **Dependency:** `poethepoet` (0.49) added to the root dev group.

### How to run

```text
uv sync --all-packages    # fresh clone only (installs poe)
uv run poe install        # uv sync + npm ci
uv run poe lint
uv run poe test
uv run poe                # list all tasks
```

### Known limitations

- The new three-OS CI matrix has not run on GitHub yet (there is no remote). Only Windows was exercised locally.
- The Phase 0 and Phase 1 sections below are historical: wherever they mention `make` or `dev.ps1`, the equivalent is now `uv run poe <same target>`.

## Phase 1: Simulator (done, 2026-11-11)

### What was built

- **`twin_sim` package:**
  - `physics.py`: first-order thermal model (exact integration), load-dependent vibration, current and speed.
  - `faults` (in `physics.py` and `line.py`): bearing_wear, overheating, spindle_imbalance (CNC only), belt_jam, and the held-out lubrication_loss.
  - `thresholds.py`: per-run perturbed thresholds with hysteresis.
  - `feasibility.py`: reachability and minimum-lead-time validator.
  - `line.py`: `LineSimulator` with discrete parts, buffers, STARVED/BLOCKED with root cause, commands, auto-recovery, sensor faults, `step` / `snapshot` / `restore` / `fast_forward`.
  - `scenario.py`: `RunPlan`.
  - `recorder.py` and `labels.py`: Arrow table, time-to-fault with a censored flag.
  - `batch.py`: multiprocessing, Parquet/CSV/JSONL, `manifest.json`.
  - `runner.py`: asyncio `RealtimeRunner`.
  - `plots.py`: example plots.
  - `__main__.py`: CLI with `generate`, `run` and `plot-examples`.
  - `testing.py`: pytest fixtures.
- **Config:**
  - `config/simulator.yaml`: every physics, fault, threshold, noise and label parameter. Loading it runs the feasibility validator.
  - `config/scenarios/`: `default`, `quick`, `e5_lubrication_loss`, `sensor_faults`, `cnc_stop`.
- **Docs:** [simulator.md](simulator.md) covers the equations, parameter tables, threshold sources (ISO 10816 zones, IEC 60034-1 insulation classes, IEC 60947-4-1 overload), how lubrication_loss differs, the feasibility numbers, label definitions and measured performance. Design decisions are in [ADR 0003](adr/0003-simulator-design.md).
- **Tooling:**
  - `sim`, `sim-quick` and `sim-plots` targets. These were originally in `Makefile` and `dev.ps1`; they are now poe tasks.
  - CI step "minimal-mode data path": `generate --quick --export csv,jsonl`, then `tests/test_phase1_integration.py` on the output. This is now the poe task `minimal-data-check`.

### How to run

```text
uv run poe sim         # 200 runs x 3600 s, seed 42 -> data/sim/ (Parquet + manifest.json)
uv run poe sim-quick   # 5 x 1800 s fault-dense runs + CSV/JSONL -> data/sim_quick/
uv run poe sim-plots   # experiments/results/sim_examples/*.png
uv run python -m twin_sim run --speed 10 --duration 120    # live Readings as JSON lines on stdout
uv run python -m twin_sim generate --help
```

Measured: the full 200-run dataset (2.88 M rows, 132 MB) generated in 9.8 s with 17 workers. One worker does 10 runs in 2.7 s, which puts the full dataset at about 1 min on a single core.

### Tests

pytest now runs 221 tests: 76 in `simulator/tests` and 3 in `tests/test_phase1_integration.py`.

- **Determinism:** same seed gives an identical table, a different seed or run gives a different one, and output is identical with 1 or 2 workers.
- **Physics:**
  - higher load gives higher steady-state temperature and current (pure functions and a full simulation)
  - exact thermal integration
  - degradation is monotone in progress
  - the cooling-loss cap holds
- **Faults:**
  - healthy line never warns at the worst case
  - every gradual fault reaches FAULT at the best case within its progression time (several seeds)
  - every gradual fault has lead time ≥ 300 s, WARNING before FAULT and a rising primary precursor at the worst case (several seeds)
  - jam drops speed and spikes current
  - sensor faults change observations but not states
  - held-out fault excluded by default
  - imbalance occurs only on the CNC
- **Feasibility validator:** rejects the old overheating model, too-fast progression and a too-low healthy margin. Readable `ConfigError`s.
- **Propagation:**
  - stopping CNC-01 makes CNV-01 BLOCKED and INSP-01/CNV-02 STARVED, all with root cause CNC-01, then recovers
  - fault propagation follows the root-cause chain
  - SOURCE starvation
- **Commands:** every type, rejections, idempotency, maintenance resets wear, RESET keeps wear, auto-recovery.
- **Snapshot:** snapshot → JSON → restore is bit-identical (including mid-jam and mid-fault). fast_forward equals repeated steps.
- **Labels:** `time_to_fault` unit tests. On generated data, ttf counts down by 1 s to onset, censored rows have no fault ahead, and non-censored rows do.
- **Runner:** order, seq and timestamps, pacing by speed, async callback, commands, stop.
- **Batch and CLI:** files, schema, manifest, JSONL parses as `Reading`, `--quick`, config errors.
- **Integration:** JSONL replays into `Reading`s time-ordered per machine. CSV agrees with Parquet. Importing the simulator pulls in no DB, broker or ML module.

### Known limitations

- `dev` and `minimal` are still stubs; they need the backend file adapter and the frontend (Phase 2+). The CI minimal-mode step checks only the file-based data path for now.
- Parquet stores observed values only. With sensor faults on, the true values are not in the dataset; they are available in `StepFrame` (`true_vib/temp/current`).
- Ambient temperature is constant per run, and speed has process noise but no separate measurement noise.
- With the default scenario, few spindle_imbalance faults reach FAULT (17 of 35 planned in the 200-run dataset), because only CNC-01 is eligible. Late or low-load gradual faults are often censored, and onsets that hit an already-faulted machine are skipped (60 of 383). All of this is recorded in the manifest.
- At the time, only `dev.ps1` was run locally (no `make`). Both have since been replaced by poe tasks (see the Tooling section above). The CI steps have not run on GitHub yet.
- I checked the example plots visually (bearing_wear, lubrication_loss, line coupling). Please review all seven PNGs.

### Enhancement: CNC spindle speed changes between jobs (2026-11-12)

- **New scenario option, `speed_changes`.** Every U(300, 900) s it issues a SET_SPEED for CNC-01 to U(7500, 10000) RPM, through the same command path as the auto-recovery operator.
  - It is in `default`, and in `quick`, `sensor_faults` and `e5_lubrication_loss`, which mirror the default operating conditions. It is not in `cnc_stop`.
  - The setpoints come from a separate seeded stream (`speed_plan`), so fault plans are unchanged: the regenerated 200-run dataset has the same planned faults as before.
- **Validated speed range.** A new per-kind `speed_operating_rpm` in `simulator.yaml` (CNC 7500–10000). The validator runs the healthy, reachability and lead-time checks at both ends of the range.
  - No parameter change was needed.
  - The tightest case is bearing wear at 10000 RPM, with a worst-case lead time of 315 s.
  - spindle_imbalance needs 380 s at 10000 RPM.
- **Scenario checks.** `check_scenario_compat` now takes the `LineConfig`. It rejects speeds outside the validated range, and speeds whose CNC cycle time would not stay below the arrival interval (7500 RPM gives 6.4 s, against 6.5 s arrivals).
- **New plot:** `imbalance_vs_bearing_speed_steps.png`. Vibration divided by the speed ratio is smooth for bearing wear and steps with speed for imbalance.
- **Tests (213 in total):**
  - healthy run at both CNC speed ends
  - reachability and lead time at both speed ends
  - validator rejects a 12000 RPM range
  - scenario speed and cycle-time rules
  - speed changes planned, applied, and independent of the fault plan
  - imbalance ∝ speed² vs bearing ∝ speed
- **Regenerated:** `sim-quick`, the plots and `data/sim`. The 200 runs took 16.2 s this time (9.8 s before), within normal run-to-run variation on this machine.

### Enhancement: single-station mode (2026-11-12)

- **`config/line_single.yaml`:** CNV-01 only, no buffers. This is the "conveyor or CNC station" from the original synopsis. The simulator needed no code change: the source feeds the machine directly and parts leave to the sink.
- **`config/scenarios/quick_single.yaml`:** `quick` without the CNC-only parts. Full-line scenarios reference CNC-01 and are refused on the single line with a clear message (`... not on line 'line1_single'`).
- **CLI:** `--line` on `generate` and `run`, defaulting to `config/line.yaml`. It replaces the earlier global `--line-config` option, which had never been documented. `plot-examples` always uses the full line.
- **Targets:** `sim-quick-single` (now the poe task `uv run poe sim-quick-single`) writes to `data/sim_quick_single/`. Ran it: 5 runs, 9000 rows, 1.6 s.
- **Tests:** 8 new in `simulator/tests/test_sim_single_line.py`:
  - healthy station shows only RUNNING with null buffers
  - gradual fault lead time and root cause
  - SOURCE starvation, STOP
  - snapshot round trip
  - full-line scenario rejected
  - batch output
  - CLI `generate`/`run` with `--line`
- **Docs:** [simulator.md](simulator.md) has a new §5 "Single-station mode"; the later sections are renumbered.

### Deviations from the plan / CLAUDE.md

- **Minimum progression time:** `D_min` is 720 s instead of 600 s, so the worst-case bearing-wear lead time stays ≥ 300 s.
- **New current thresholds** (1.15 / 1.30 × rated). They are needed so lubrication_loss can trip on overcurrent.
- **Model details changed from the original plan:**
  - bearing-wear friction is 0.2·w (was 0.3·w), to keep current flat;
  - bearing vibration scales with the speed ratio;
  - lubrication_loss has no deterministic vibration term, only stick-slip bursts.
- **New files outside the CLAUDE.md layout:**
  - `config/simulator.yaml`
  - `twin_sim/feasibility.py`
  - `twin_sim/testing.py`, a fixture plugin registered by a new root `conftest.py`, because `conftest.py` names must be unique repo-wide (ADR 0001)
- **`twin_common.logging.configure_logging`** has a new optional `stream` argument, so the `run` CLI can keep stdout for Readings.
- **Dependencies:** `numpy` and `pyarrow` were added to `twin-sim`, and `matplotlib` to the root dev group (approved). A mypy override ignores missing pyarrow type information.
- **Targets:** `sim-quick` and `sim-plots` were added on top of `sim`.

## Phase 0: Foundation (done, 2026-11-11)

### What was built

- **CLAUDE.md updated for the no-Docker decision:**
  - native Mosquitto
  - native PostgreSQL (plain features only: BRIN indexes and materialized views, no TimescaleDB)
  - Docker Compose and Grafana removed
  - `config/mosquitto.conf`, `scripts/dev.ps1` and `setup_windows.md` added to the layout
  - new rule in §7: "Do not use Docker."

  See [ADR 0002](adr/0002-native-services-no-docker.md).
- **uv workspace** with members `common/`, `simulator/`, `backend/` and `ml/`, one `uv.lock`, and Python 3.11 pinned in `.python-version`. See [ADR 0001](adr/0001-monorepo.md). The empty subpackages from CLAUDE.md §5 exist in `twin_backend` and `twin_ml`.
- **`twin_common`:**
  - `enums.py`: MachineState, Quality, HealthClass, SensorName, CommandType, AckStatus, Reliability, MachineKind, AdapterKind. It also has `to_health_class()` implementing the §6 ML target mapping. OFFLINE and STALE raise `UnlabelableStateError`.
  - `contracts.py`: `Reading` (schemaVersion 2), `Prediction`, `Attribution`, `Command`, `CommandAck`.
    - JSON is camelCase, Python is snake_case.
    - Models are frozen, `extra="forbid"`, and use UTC-normalized aware timestamps.
    - Prediction probs must cover all three classes and sum to 1. `conformalSet` is unique and in canonical order, and `ttfInterval` must satisfy lo ≤ hi.
    - SET_SPEED / SET_LOAD require a numeric `speed` / `load` param.
  - `config/`: `LineConfig`, `SourcesConfig`, `TwinSettings` (env prefix `LINETWIN_`), and `load_line_config` / `load_sources_config`. The loaders raise a single `ConfigError` that names the file and each failing location.
  - `topics.py`: MQTT topic builders. `logging.py`: structlog JSON setup. `testing.py`: sample contracts for tests in every package.
- **Config:**
  - `config/line.yaml`: CNV-01 → CNC-01 → INSP-01 → CNV-02, buffers B1/B2/B3 (capacities 10/8/10), a sensor catalog with physical ranges, per-machine sensor→part mappings, and load ranges low 30–50, medium 50–75, high 75–95.
  - `config/sources.yaml`: `active: file`, the minimal-mode default.
  - `config/mosquitto.conf`.
- **Frontend:** Vite 8, React 19, TypeScript 6 (strict, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`), ESLint 10 with typescript-eslint strict-type-checked, and Vitest 5. It has empty `src/engine`, `src/dashboard` and `src/api` folders. An ESLint rule forbids React or dashboard imports inside `src/engine`.
- **Tooling:**
  - `Makefile` and `scripts/dev.ps1`, with targets help, install, lint, fmt, test, dev, minimal and sim
  - `.pre-commit-config.yaml` (ruff, ruff-format, mypy, eslint), with local hooks run through uv/npm so versions match the lockfiles
  - `.github/workflows/ci.yml`: Linux runs `make`, Windows runs `dev.ps1` under PowerShell 5.1
- **Docs:** `architecture.md` (Mermaid), `setup_windows.md`, ADRs 0001 and 0002.

### How to run

```powershell
.\scripts\dev.ps1 install
.\scripts\dev.ps1 lint      # ruff, ruff format --check, mypy (strict), eslint, tsc
.\scripts\dev.ps1 test      # pytest (143 tests), vitest
```

### Tests

- Round trips (model → JSON → model, and via a dict) for every contract, every MachineState, Quality, CommandType and AckStatus value, and optional-field variants. The wire keys are checked to be camelCase.
- Contract rejections: unknown field, naive timestamp, schemaVersion ≠ 2, negative sensor values, bad probs, unordered ttf interval, duplicate conformal classes, missing or non-numeric command params, empty ids.
- Config errors:
  - unknown sensor
  - catalog sensor that isn't a Reading field
  - buffer from/to a machine that doesn't exist
  - buffer self-link
  - negative and zero capacity
  - duplicate machine id
  - part not on the machine
  - sensor without a part
  - overlapping, gapped or out-of-bounds load ranges
  - unknown key
  - missing file, YAML syntax error, non-mapping top level
  - active adapter without its section, unknown adapter, bad port
- Integration (`tests/test_phase0_integration.py`): the shipped configs load through `TwinSettings`. Every configured sensor is a Reading field with a part mapped, and a Reading can be built and round-tripped for every machine.

### Known limitations

- `dev`, `minimal` and `sim` are stubs that print "available from Phase 1" and exit with code 1. There is no minimal-mode CI job yet because there is nothing to run. Add it in Phase 1.
- Field-level config errors (e.g. a negative capacity) are reported before the cross-reference checks run. Cross-reference errors are all reported together, but they only appear once the field-level errors are fixed.
- `make` is not installed on the development machine. The Makefile is exercised by the Linux CI job only, and only `dev.ps1` was run locally.
- The CI workflow has not run on GitHub yet, since there's no remote.
- `pre-commit run --all-files` only sees files tracked by git. Nothing is committed yet, so only `pre-commit validate-config` was run.
- `npm audit` reports a moderate minimatch ReDoS advisory. It comes from typescript-eslint and ESLint (dev-only lint tooling), and the suggested fix downgrades ESLint, so it was left as is for now.
- No frontend unit tests yet (`vitest --passWithNoTests`). The first ones arrive with the engine's state→visual mapping.

### Deviations from the plan / CLAUDE.md

- **React 19** instead of the React 18 named in the plan, because 19 is the current stable release.
- **TypeScript ~6.0** rather than 7.x, because typescript-eslint supports TS < 6.1.
- `to_health_class` raises for OFFLINE/STALE. §6 does not map them, and they describe the data link rather than machine health.
- `Prediction.ttfS` and `ttfInterval` are nullable, where `null` means no fault is expected within the horizon.
- Test directories have no `__init__.py`, and test file names are unique repo-wide (see ADR 0001).
- `help` and `install` targets were added on top of the requested dev/minimal/test/lint/fmt/sim.

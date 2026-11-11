# LineTwin progress

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
  - `sim`, `sim-quick` and `sim-plots` targets in `Makefile` and `dev.ps1`.
  - CI step "minimal-mode data path" in both jobs: `generate --quick --export csv,jsonl`, then `tests/test_phase1_integration.py` on the output.

### How to run

```powershell
.\scripts\dev.ps1 sim         # 200 runs x 3600 s, seed 42 -> data/sim/ (Parquet + manifest.json)
.\scripts\dev.ps1 sim-quick   # 5 x 1800 s fault-dense runs + CSV/JSONL -> data/sim_quick/
.\scripts\dev.ps1 sim-plots   # experiments/results/sim_examples/*.png
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
- `make` is still not installed locally. Only `dev.ps1` was run here, and the new CI steps have not run on GitHub yet.
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
- **Targets:** `sim-quick-single` in the Makefile and dev.ps1 writes to `data/sim_quick_single/`. Ran it: 5 runs, 9000 rows, 1.6 s.
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

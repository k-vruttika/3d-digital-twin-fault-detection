# LineTwin progress

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

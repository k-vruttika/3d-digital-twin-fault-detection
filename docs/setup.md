# Setup

LineTwin runs on Windows, macOS and Linux:

- **Minimal mode and the simulator** need only **uv** and **Node.js**.
- **Services:** the MQTT broker (Mosquitto, from Phase 2) and the event store (TimescaleDB, i.e. PostgreSQL 16 with the TimescaleDB extension, from Phase 3) run via **Docker Compose**. That is the main path; native installs are a fallback (section 3). See [ADR 0005](adr/0005-docker-compose-services.md).

Run all commands from the repository root. They work the same in PowerShell, cmd, bash and zsh.

## 1. Core tools (required)

| Tool | Version | Windows | macOS | Linux |
|---|---|---|---|---|
| uv | ≥ 0.5 | `winget install --id astral-sh.uv -e` | `brew install uv` | the official installer from docs.astral.sh/uv |
| Node.js | ≥ 22 (CI uses 24) | `winget install --id OpenJS.NodeJS.LTS -e` | `brew install node` | your distribution's package or nodejs.org |
| Git | any recent | `winget install --id Git.Git -e` | `brew install git` | your distribution's package |

You don't need to install Python yourself; uv downloads the pinned Python 3.11. If winget reports that an id was not found, search with `winget search <name>`.

```text
uv sync --all-packages      # first time only: installs the Python packages, including poe
uv run poe install          # uv sync + npm ci
uv run poe lint
uv run poe test
uv run poe                  # list every task
uv run pre-commit install   # optional: run hooks on every commit
```

`uv run poe lint` and `uv run poe test` are the main checks: run both before every commit. The real-time Phase 11 demo scenarios (about 40 min) are not part of `poe test`; run them on purpose with `uv run poe demo-tests` (needs `uv run poe up` and `uv run poe deploy-models`, see [demo_script.md](demo_script.md)).

**CI is manual.** The GitHub Actions workflow (`.github/workflows/ci.yml`) does not run on push or pull request. Start it from the Actions tab (CI → Run workflow), or with `gh workflow run ci.yml`. A run uses Ubuntu only (with the Docker Compose services and the e2e tests); tick **Also run on Windows** / **Also run on macOS** in the run form (or pass `-f windows=true -f macos=true` to `gh`) to add those jobs. Every job runs the same poe tasks as locally.

Developer tasks are [poethepoet](https://poethepoet.natn.io/) tasks in the root `pyproject.toml` ([ADR 0004](adr/0004-poe-task-runner.md)). They run without a shell, so no execution-policy changes and no `make` are needed.

## 2. Services via Docker Compose (main path)

### Install Docker

- **Windows and macOS:** install **Docker Desktop** from docker.com (on Windows also `winget install --id Docker.DockerDesktop -e`). Start it once and wait until it reports that the engine is running. On Windows it uses the WSL 2 backend.
- **Linux:** install **Docker Engine** and the **Docker Compose plugin** following the official instructions for your distribution on docs.docker.com. To run `docker` without `sudo`, follow the "post-installation steps" there (adding your user to the `docker` group).

Check: `docker compose version` prints a v2 or newer version.

### Configure (optional)

The defaults work without any configuration. To change credentials or the database port, copy `.env.example` to `.env` (gitignored) and edit it. Docker Compose and the Python settings (`twin_common.config.TwinSettings`, prefix `LINETWIN_`) both read this file.

| variable | default | meaning |
|---|---|---|
| `LINETWIN_PG_DB` / `LINETWIN_PG_USER` / `LINETWIN_PG_PASSWORD` | `linetwin` / `linetwin` / `linetwin-dev` | database created on first start (local development values only) |
| `LINETWIN_PG_PORT` | `5432` | host port of TimescaleDB. Change it, e.g. to `5433`, if a native PostgreSQL already uses 5432. |
| `LINETWIN_API_KEYS_OPERATOR` / `LINETWIN_API_KEYS_VIEWER` | empty | backend API keys, `name:key,name2:key2`. Empty: read-only API, loopback only. Generate keys with `uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`. |
| `LINETWIN_LINE_FILE` | `line.yaml` | line layout in `config/` (`line_single.yaml` for the single station) |

The MQTT port is fixed at **1883**, to match `config/sources.yaml`. Every port is published on **127.0.0.1** only, so nothing is reachable from the network.

### Run

```text
uv run poe up      # start TimescaleDB + Mosquitto and wait until both are healthy
uv run poe ps      # status and health
uv run poe logs    # last 200 log lines; extra arguments pass through: uv run poe logs -f mosquitto
uv run poe down    # stop (data is kept in the named volumes)
uv run poe reset   # stop and DELETE the data volumes
```

The first `up` downloads the images. Images are pinned to exact tags in `docker-compose.yml`:
- `timescale/timescaledb:2.31.0-pg16`
- `eclipse-mosquitto:2.1.2-alpine`

Connection details:

- **MQTT:** `127.0.0.1:1883`, anonymous; topics `plant/line1/...`.
- **TimescaleDB:** `postgresql://linetwin:linetwin-dev@127.0.0.1:5432/linetwin` (or your `.env` values). The `timescaledb` extension is created in the database on first start.

### Tests and services

Tests that need a service (marked `services`, e.g. `tests/test_compose_services.py`) run when the services are reachable. Otherwise they are skipped with "start services with: uv run poe up". Setting `LINETWIN_REQUIRE_SERVICES=1` makes them fail instead of skip; CI (manual runs, see section 1) does this on Ubuntu, the only CI job that starts the services.

Event-store tests run on SQLite and, when TimescaleDB is reachable, on TimescaleDB too. They never touch your configured database:
- each test session creates its own database `linetwin_test_<random hex>` on the same server, runs the migrations, and drops it at the end (`DROP DATABASE ... WITH (FORCE)`);
- the database user needs the right to create databases, which the Compose user has;
- `uv run poe storage-bench` does the same with a `linetwin_bench_<hex>` database.

To use TimescaleDB as the live store, set `backend: timescale` in `config/storage.yaml` and run `uv run poe db-migrate` once (the backend also migrates on start).

## 3. Native fallback (without Docker)

Use this only if Docker is not an option. The services must then match what the code expects: Mosquitto 2 with the project config, and PostgreSQL 16 **with the TimescaleDB extension**.

### Mosquitto

The config is split into a shared settings file and two small entry files:

| file | role |
|---|---|
| `config/mosquitto.d/common.conf` | shared settings (anonymous access, persistence, logging, limits) |
| `config/mosquitto.conf` | native entry: `listener 1883 127.0.0.1` (loopback only) plus `include_dir config/mosquitto.d` |
| `config/mosquitto.docker.conf` | container entry (used by Docker Compose only) |

1. Install Mosquitto 2:
   - **Windows:** the installer from mosquitto.org → Download, or `winget search mosquitto`. It installs to `C:\Program Files\mosquitto\`.
   - **macOS:** `brew install mosquitto`.
   - **Linux:** your distribution's `mosquitto` package.
2. Stop any Mosquitto service that already holds port 1883, and stop the Docker services (`uv run poe down`).
3. Run in the foreground **from the repository root**, because `include_dir` is relative to the working directory:

   ```text
   mosquitto -c config/mosquitto.conf -v
   ```

   On Windows, without `mosquitto` on `PATH`: `& "C:\Program Files\mosquitto\mosquitto.exe" -c config\mosquitto.conf -v`. Persistence is written to `./mosquitto.db`, which is gitignored.
4. Check it from a second terminal: `mosquitto_sub -h 127.0.0.1 -t "plant/line1/#" -v`, and from a third: `mosquitto_pub -h 127.0.0.1 -t plant/line1/CNC-01/telemetry -m hello`.

**As a Windows service:** the service reads `C:\Program Files\mosquitto\mosquitto.conf` and does not start in the repository folder. Write that file (from an admin shell) with `listener 1883 127.0.0.1` and an **absolute** `include_dir` pointing at this repository's `config\mosquitto.d`, then run `Restart-Service mosquitto`.

Anonymous access is enabled only because the listener is bound to loopback. Do not expose the broker on a network interface without `allow_anonymous false` and a `password_file`.

### PostgreSQL 16 + TimescaleDB

1. Install PostgreSQL 16 or newer:
   - **Windows:** the EDB installer from postgresql.org, or `winget search PostgreSQL`.
   - **macOS and Linux:** your package manager.

   Keep port 5432, set a password for the `postgres` superuser, and skip Stack Builder.
2. Install the **TimescaleDB extension** for your PostgreSQL version, following Timescale's official "self-hosted" installation documentation for your OS. Then make sure `shared_preload_libraries` contains `timescaledb` and restart PostgreSQL.
3. Create the role, the database and the extension. On Windows, add the client tools to `PATH` first, e.g. `$env:Path += ";C:\Program Files\PostgreSQL\17\bin"`.

   ```text
   psql -U postgres -h 127.0.0.1 -c "CREATE ROLE linetwin LOGIN PASSWORD 'linetwin-dev';"
   psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE linetwin OWNER linetwin;"
   psql -U postgres -h 127.0.0.1 -d linetwin -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
   psql -U linetwin -h 127.0.0.1 -d linetwin -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"
   ```

4. The code reads the same `LINETWIN_PG_*` settings as Docker Compose (from `.env` or the environment). If you used other credentials or a different port, set them in `.env`.

A native PostgreSQL on 5432 and the Docker TimescaleDB cannot both use 5432. Set `LINETWIN_PG_PORT` in `.env` to choose which one the code connects to.

## 4. Running the backend

```text
uv run poe sim-quick; uv run poe minimal   # minimal mode: replay file + SQLite, nothing else
uv run poe up; uv run poe dev              # dev: simulator publisher for sources.yaml `active`
                                           # + backend with TimescaleDB (Mosquitto for mqtt)
```

The API listens on `http://127.0.0.1:8000` (`config/api.yaml`). `GET /health` needs no key. For anything else set API keys in `.env` and send `X-API-Key`; without keys the API is read-only. `uv run poe dev` checks that the services are reachable and stops the simulator publisher on exit (Ctrl+C). See [api.md](api.md).

## 5. Running backend and frontend together

The dashboard (and the 3D twin's dev page) needs the backend running. Use two terminals:

```text
# terminal 1: a backend (either one)
uv run poe sim-quick; uv run poe minimal     # file replay, no Docker
uv run poe up; uv run poe dev --source mock  # live simulator, commands work

# terminal 2: the Vite dev server
uv run poe frontend                          # dashboard: http://127.0.0.1:5173/  (3D only: /dev.html)
```

- **API keys:** if `.env` configures them, enter a key in the dashboard header; it stays in memory, or in sessionStorage if you tick "remember in this tab". For the dev page, put a viewer key in `frontend/.env.local` as `VITE_LINETWIN_API_KEY=...` (gitignored). Without keys the backend serves anonymous read access on loopback, and neither page needs a key.
- **Screenshots:** `uv run poe screenshots` renders the dev page and the dashboard headless. Playwright downloads its Chromium on the first run. The dashboard's dev-mode shots need `uv run poe up`.
- **End-to-end tests:** they need Playwright's Chromium. Run `uv run poe browsers` once; on Linux it also installs the system libraries and asks for sudo. Without the browser the e2e tests are skipped. The dev-mode e2e test also needs the services, and must not run alongside `uv run poe dev`, because both use the mock server's port.

Details: [frontend.md](frontend.md).

## 6. Minimal mode reminder

None of sections 2 and 3 is needed to run the twin from a JSON/CSV file or the mock server. `config/sources.yaml` ships with `active: file` and `config/storage.yaml` with `backend: sqlite`, and minimal mode never requires Docker (CLAUDE.md §1 and §7). `uv run poe minimal` is the entry point.

## 7. ML (Milestone B)

- **Dependencies:** `uv sync --all-packages` installs the ML stack with the rest:
  - PyTorch, CPU build (PyTorch's CPU index on Windows and Linux, PyPI on macOS); no CUDA wheels;
  - scikit-learn and XGBoost (`xgboost-cpu` on Windows and Linux);
  - **macOS:** XGBoost needs the OpenMP runtime: run `brew install libomp` once (without it, importing XGBoost fails with `Library not loaded: @rpath/libomp.dylib`). The two XGBoost-dependent tests skip with that hint when it is missing locally; CI installs it and sets `LINETWIN_REQUIRE_XGBOOST=1` so they must run there;
  - **macOS caveat, two OpenMP runtimes:** torch ships its own `libomp.dylib`, while XGBoost uses Homebrew's. If both load in one Python process, it can crash with a segmentation fault (exit code 139). For that reason `uv run poe train`, `uv run poe train-all` and the e2e tests train every (model, seed) in its own worker process (`twin_ml.train.isolated`), so torch and XGBoost never meet. Do the same in your own scripts and notebooks: don't import torch and xgboost in the same process on macOS, and don't set `KMP_DUPLICATE_LIB_OK`. Each worker logs `train.openmp` with the OpenMP runtimes it loaded ([ADR 0020](adr/0020-ml-models-training-tracking.md));
  - `mlflow-skinny`.
  
  The first sync downloads a few hundred MB. No GPU is needed: model sizes and the training subsample are chosen for a laptop CPU (measured times in [PROGRESS.md](PROGRESS.md)).
- **Run E1:** `uv run poe dataset`, `uv run poe train-all`, `uv run poe report-e1`. Details: [experiments.md](experiments.md).
- **MLflow UI:** `uv run poe mlflow-ui` runs the full MLflow UI in an isolated `uv tool` environment, on the local store `mlruns/` (gitignored).
- **Isolation:** the backend and minimal mode never import torch, scikit-learn, XGBoost, MLflow or `twin_ml`; tests enforce this.

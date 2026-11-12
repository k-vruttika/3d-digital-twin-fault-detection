# ADR 0005: Services via Docker Compose; TimescaleDB instead of plain PostgreSQL

- **Status:** Accepted. Supersedes [ADR 0002](0002-native-services-no-docker.md).
- **Date:** 2026-11-12
- **Phase:** 1 (infrastructure)

## Context

[ADR 0002](0002-native-services-no-docker.md) chose natively installed Windows services (Mosquitto, plain PostgreSQL) and ruled out Docker. Since then:

- The task runner became OS-neutral ([ADR 0004](0004-poe-task-runner.md)), and CI runs on Ubuntu, Windows and macOS. Native installs differ per OS, with different packages, paths and service managers, and each one had to be documented and maintained.
- Plain PostgreSQL meant re-implementing time-series features by hand: materialized views refreshed by the backend, retention by `DELETE`, BRIN tuning. TimescaleDB provides these as hypertables, continuous aggregates and retention policies, but installing the extension natively is a separate step on every OS.
- Docker Desktop is now available on the development machine.

## Decision

- **Docker Compose** (`docker-compose.yml` at the repo root) runs the services:
  - **TimescaleDB:** `timescale/timescaledb:2.31.0-pg16`, i.e. PostgreSQL 16 with the TimescaleDB extension. Named volume `timescale-data`, health check `pg_isready` over TCP. Database, user and password come from `.env` (template `.env.example`), with defaults in the compose file.
  - **Mosquitto:** `eclipse-mosquitto:2.1.2-alpine`, matching the natively installed 2.1.2. Named volume `mosquitto-data`, health check with `mosquitto_pub`.
  - Images are pinned to exact tags, and every published port is bound to **127.0.0.1** only.
- **Storage:** PostgreSQL 16 with TimescaleDB. Hypertables and continuous aggregates are allowed. SQLite remains the event store for minimal mode and tests.
- **Mosquitto config:** split, so the native fallback keeps its loopback-only listener.
  - `config/mosquitto.d/common.conf` holds the shared settings, without a listener or paths.
  - `config/mosquitto.docker.conf` is the container entry: it listens on all interfaces inside the container, and the host publishes the port on 127.0.0.1 only.
  - `config/mosquitto.conf` is the native entry: `listener 1883 127.0.0.1`.
- **Settings:** `.env` is read by Compose and by `TwinSettings` (prefix `LINETWIN_`), so both use the same credentials and ports.
- **Tasks** (poe, cross-platform `docker compose` commands):
  - `up` starts the services and waits until they are healthy;
  - `down` stops them;
  - `reset` also deletes the volumes;
  - `logs` and `ps` show logs and status.
- **Native installs** of Mosquitto and PostgreSQL with TimescaleDB are a documented fallback in [setup.md](../setup.md).
- **Minimal mode must never require Docker.** CLAUDE.md §7 records this rule.

## Consequences

- One command (`uv run poe up`) gives identical service versions on Windows, macOS and Linux, with no per-OS install steps.
- Docker Desktop (Windows, macOS) or Docker Engine (Linux) is required for the service-dependent parts: MQTT adapter, event store, and later phases. Minimal mode, the simulator and most tests do not need it.
- **Tests:** service tests use the `mqtt_broker` / `timescale_db` fixtures (`twin_backend.testing`) and are marked `services`. They run when the services are reachable and are otherwise skipped with "start services with: uv run poe up". In CI only the Ubuntu job starts the services, and it sets `LINETWIN_REQUIRE_SERVICES=1` so the service tests fail rather than skip there. macOS runners have no Docker, and Windows runners only run Windows containers.
- **Pinned images** must be upgraded deliberately. A TimescaleDB major upgrade on an existing volume needs `ALTER EXTENSION timescaledb UPDATE` or a fresh volume (`uv run poe reset`).
- **Data** lives in named Docker volumes and survives `down`; `reset` deletes it.
- **Ports:** if a host port is taken (e.g. by a native PostgreSQL on 5432), it is changed in `.env` (`LINETWIN_PG_PORT`). The MQTT port is fixed at 1883 to match `config/sources.yaml`.
- **Native fallback:** PostgreSQL must have the TimescaleDB extension installed (see Timescale's installation documentation), because the backend may use hypertables.

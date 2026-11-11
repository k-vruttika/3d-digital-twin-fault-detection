# ADR 0002: Native Windows services instead of Docker; plain PostgreSQL instead of TimescaleDB

- **Status:** Accepted
- **Date:** 2026-11-11
- **Phase:** 0

## Context

The original plan ran Mosquitto, TimescaleDB and optional Grafana/Prometheus under Docker Compose. Development happens on Windows 11, and the project owner decided not to use Docker at all.

## Decision

- **No Docker.** All services run natively on Windows. This is recorded as a rule in CLAUDE.md section 7.
- **MQTT:** Mosquitto, installed natively, started with the repo's `config/mosquitto.conf` (loopback-only listener on 1883).
- **Storage:** native **PostgreSQL** replaces TimescaleDB. Only plain PostgreSQL features are used, with no extensions:
  - append-only event tables with B-tree indexes on `(machine_id, ts)` and **BRIN** indexes on `ts` for cheap time-range scans
  - **materialized views** for downsampled aggregates (e.g. 1 s / 1 min buckets via `date_bin`), refreshed by the backend
  - optional declarative range **partitioning** by time if volumes require it
- **SQLite** remains the event store for minimal mode and tests.
- **Ops:** Docker Compose and Grafana are dropped. The backend still exposes a Prometheus `/metrics` endpoint, and Locust is still used for load testing.

## Consequences

- Fewer moving parts on a Windows laptop. There's no Docker Desktop, WSL2 or licensing concern.
- Setup is manual and documented in `docs/setup_windows.md`. CI does not need these services until Phases 2/3. When it does, the jobs will install them natively or use the SQLite path.
- There are no Timescale hypertables, continuous aggregates or compression. Materialized views must be refreshed explicitly, and retention is a scheduled `DELETE` or partition drop. The event-store interface hides this from callers.
- Minimal mode (CLAUDE.md section 1) is unaffected. It never needs PostgreSQL or Mosquitto.

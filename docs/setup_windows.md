# Windows setup (no Docker)

LineTwin runs natively on Windows 11. Minimal mode and Phases 0–1 need only **uv** and **Node.js**. You need **Mosquitto** from Phase 2 (MQTT adapter) and **PostgreSQL** from Phase 3 (event store).

All commands are for PowerShell, run from the repository root unless stated otherwise.

## 1. Core tools (required now)

| Tool | Version | Install |
|---|---|---|
| uv | ≥ 0.5 | `winget install --id astral-sh.uv -e` |
| Node.js | ≥ 22 (CI uses 24) | `winget install --id OpenJS.NodeJS.LTS -e` |
| Git | any recent | `winget install --id Git.Git -e` |

You don't need to install Python yourself. uv downloads the pinned Python 3.11 automatically.

If winget reports that an id was not found, search for it with `winget search <name>`. Package ids occasionally change.

```powershell
.\scripts\dev.ps1 install   # uv sync + npm ci
.\scripts\dev.ps1 lint
.\scripts\dev.ps1 test
uv run pre-commit install   # optional: run hooks on every commit
```

If PowerShell refuses to run the script ("running scripts is disabled on this system"), allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or run a single command without changing policy:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 lint
```

`make` is optional on Windows. `scripts\dev.ps1` has the same targets as the `Makefile`.

## 2. Mosquitto MQTT broker (Phase 2)

### Install

1. Download the 64-bit Windows installer from the official Eclipse Mosquitto download page (mosquitto.org → Download). Alternatively, use `winget search mosquitto` and install the Eclipse Mosquitto package it lists.
2. Install to the default location, `C:\Program Files\mosquitto\`. The installer can also register a Windows service called **Mosquitto Broker**.
3. Optionally add `C:\Program Files\mosquitto` to your `PATH`, so you can run `mosquitto`, `mosquitto_pub` and `mosquitto_sub` directly.

### Run with the project config (recommended for development)

The repo's [config/mosquitto.conf](../config/mosquitto.conf) listens on `127.0.0.1:1883` only and logs to the console.

```powershell
# If the Windows service is running it already holds port 1883; stop it first (admin shell):
Stop-Service mosquitto -ErrorAction SilentlyContinue

& "C:\Program Files\mosquitto\mosquitto.exe" -c config\mosquitto.conf -v
```

Leave that window open. In a second window, check that it works:

```powershell
& "C:\Program Files\mosquitto\mosquitto_sub.exe" -h 127.0.0.1 -t "plant/line1/#" -v
# in a third window:
& "C:\Program Files\mosquitto\mosquitto_pub.exe" -h 127.0.0.1 -t "plant/line1/CNC-01/telemetry" -m "hello"
```

The subscriber prints `plant/line1/CNC-01/telemetry hello`.

### Run as a Windows service (optional)

The service reads `C:\Program Files\mosquitto\mosquitto.conf`. To use the project settings, copy the project file over it (from an admin shell), then restart the service:

```powershell
Copy-Item config\mosquitto.conf "C:\Program Files\mosquitto\mosquitto.conf"
Restart-Service mosquitto
```

Anonymous access is enabled only because the listener is bound to loopback. Do not expose the broker on a network interface without `allow_anonymous false` and a `password_file`.

## 3. PostgreSQL (Phase 3)

LineTwin uses **plain PostgreSQL**, with no TimescaleDB or other extensions (see [ADR 0002](adr/0002-native-services-no-docker.md)).

### Install

Either:

- download the Windows installer from postgresql.org → Download → Windows (the EDB installer), or
- use `winget search PostgreSQL` and install the latest `PostgreSQL.PostgreSQL.<major>` package it lists (version 16 or newer).

During installation:

- set a password for the `postgres` superuser and remember it
- keep port **5432**
- you can skip Stack Builder; no add-ons are needed

The installer registers a Windows service named like `postgresql-x64-<major>`, which starts automatically.

Add the client tools to `PATH` for the current session (adjust the version number):

```powershell
$env:Path += ";C:\Program Files\PostgreSQL\17\bin"
psql --version
```

### Create the LineTwin database and user

```powershell
psql -U postgres -h 127.0.0.1 -c "CREATE ROLE linetwin LOGIN PASSWORD 'linetwin-dev';"
psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE linetwin OWNER linetwin;"
psql -U linetwin -h 127.0.0.1 -d linetwin -c "SELECT version();"
```

`linetwin-dev` is a local development password only. The backend will read the connection URL from the `LINETWIN_DATABASE_URL` environment variable (added in Phase 3), e.g.:

```powershell
$env:LINETWIN_DATABASE_URL = "postgresql+asyncpg://linetwin:linetwin-dev@127.0.0.1:5432/linetwin"
```

### Start and stop the service

```powershell
Get-Service postgresql*              # status
Start-Service postgresql-x64-17      # admin shell
Stop-Service  postgresql-x64-17
```

## 4. Minimal mode reminder

None of section 2 or 3 is needed to run the 3D twin from a JSON/CSV file or the mock server. `config/sources.yaml` ships with `active: file`.

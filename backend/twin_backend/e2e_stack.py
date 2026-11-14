"""End-to-end stack for the dashboard: backend + Vite + Playwright (test tooling, ADR 0018).

Used by the pytest e2e modules (``tests/test_phase6_e2e_*.py``) and by the screenshots task
(``python -m twin_backend.e2e_stack screenshots``). Everything is started with
``subprocess.Popen`` and the current interpreter or ``node`` (no shell), so it behaves the same
on Windows, Linux and macOS.

- **Minimal mode:** a fixed, seeded simulator run (``generate --quick``, run 2) replayed by the
  file source with ``--no-loop`` into SQLite. The moments worth seeking to (first WARNING,
  first FAULT, first STARVED / BLOCKED, ...) are read from the generated file, never typed by
  hand (``key_moments``).
- **Dev mode:** ``devstack --source mock`` with the quiet scenario, an operator key and a
  TimescaleDB database supplied by the caller (a throw-away test database).
- **Frontend:** the Vite dev server on a free port with ``VITE_LINETWIN_API_URL``; the backend
  gets that origin through ``--cors-origin``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"
SCREENSHOT_DIR = REPO_ROOT / "experiments" / "results" / "frontend_screenshots" / "dashboard"

# The fixed minimal-mode run: the quick scenario's run 2 (seed from the scenario), replayed fast.
MINIMAL_RUN_INDEX = 2
MINIMAL_RUNS = 3
MINIMAL_REPLAY_SPEED = 60.0
# Dev mode: the quiet scenario at a simulator speed that brings a 720 s fault in well under 30 s.
DEV_SCENARIO = REPO_ROOT / "config" / "scenarios" / "e2e_quiet.yaml"
DEV_SIM_SPEED = 30.0
E2E_OPERATOR = "e2e"

READY_TIMEOUT_S = 90.0
STOP_TIMEOUT_S = 15.0
POLL_S = 0.25


@dataclass(frozen=True)
class KeyMoment:
    """An event time in the replay file at which a machine is in a given state."""

    machine: str
    state: str
    ts: str  # ISO-8601 UTC

    def as_dict(self) -> dict[str, str]:
        return {"machine": self.machine, "state": self.state, "ts": self.ts}


# ---------------------------------------------------------------- process helpers
def free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def node() -> str:
    exe = shutil.which("node")
    if exe is None:
        raise RuntimeError("node is not on PATH (needed for Vite and Playwright)")
    return exe


def chromium_missing() -> str | None:
    """None when Playwright's Chromium is installed, else the reason (with the install hint)."""
    script = (
        "const {chromium}=require('playwright');process.stdout.write(chromium.executablePath())"
    )
    try:
        out = subprocess.run(
            [node(), "-e", script], cwd=FRONTEND, capture_output=True, text=True, timeout=60
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return f"cannot query Playwright ({exc}); install it with: uv run poe browsers"
    path = out.stdout.strip()
    if out.returncode != 0 or not path or not Path(path).exists():
        return "Playwright's Chromium is not installed; install it with: uv run poe browsers"
    return None


def stop(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=STOP_TIMEOUT_S)


def wait_http(
    url: str, proc: subprocess.Popen[Any], log: Path, timeout_s: float = READY_TIMEOUT_S
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"process exited with {proc.returncode} before {url} was up; see {log}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(POLL_S)
    raise RuntimeError(f"{url} not up after {timeout_s:.0f} s; see {log}")


def get_json(url: str, key: str | None = None) -> Any:
    req = urllib.request.Request(url, headers={"X-API-Key": key} if key else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def clean_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The current environment without API keys (the stack sets its own), plus `extra`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LINETWIN_API_KEYS_")}
    # Real environment variables win over .env, so empty values switch keys off reliably.
    env["LINETWIN_API_KEYS_OPERATOR"] = ""
    env["LINETWIN_API_KEYS_VIEWER"] = ""
    env.update(extra or {})
    return env


# ---------------------------------------------------------------- minimal mode data
def generate_replay_file(out_dir: Path) -> Path:
    """The fixed seeded run as CSV (deterministic: same seed, same file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"run_{MINIMAL_RUN_INDEX:04d}.csv"
    if not target.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "twin_sim",
                "generate",
                "--quick",
                "--runs",
                str(MINIMAL_RUNS),
                "--workers",
                "1",
                "--export",
                "csv",
                "--out",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    return target


def _iso(ts: str) -> str:
    """CSV timestamps ('2026-01-01 00:00:01.000Z') as ISO-8601."""
    return ts.replace(" ", "T")


def key_moments(csv_path: Path) -> dict[str, KeyMoment]:
    """Moments of the replay file to seek to, found by scanning its states.

    - ``warning`` / ``fault``: the first WARNING and the first FAULT of the machine that faults
      first after a warning (the build-up);
    - ``waiting``: the first STARVED or BLOCKED machine after that fault;
    - ``normal``: a time before ``warning`` when every machine was RUNNING.
    """
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows.extend(csv.DictReader(fh))
    first: dict[tuple[str, str], str] = {}
    for row in rows:
        first.setdefault((row["machine_id"], row["state"]), _iso(row["ts"]))
    candidates = [
        (first[(m, "FAULT")], m)
        for (m, st) in first
        if st == "WARNING" and (m, "FAULT") in first and first[(m, "WARNING")] < first[(m, "FAULT")]
    ]
    if not candidates:
        raise RuntimeError(f"{csv_path} has no WARNING -> FAULT build-up to replay")
    fault_ts, machine = min(candidates)
    warning_ts = first[(machine, "WARNING")]
    waiting = sorted(
        (ts, m, st)
        for (m, st), ts in first.items()
        if st in ("STARVED", "BLOCKED") and ts > fault_ts
    )
    if not waiting:
        raise RuntimeError(f"{csv_path} has no STARVED / BLOCKED machine after the fault")
    by_ts: dict[str, dict[str, str]] = {}
    for row in rows:
        by_ts.setdefault(_iso(row["ts"]), {})[row["machine_id"]] = row["state"]
    machines = {row["machine_id"] for row in rows}
    normal = next(
        ts
        for ts in sorted(by_ts, reverse=True)
        if ts < warning_ts
        and len(by_ts[ts]) == len(machines)
        and all(s == "RUNNING" for s in by_ts[ts].values())
    )
    w_ts, w_m, w_st = waiting[0]
    return {
        "normal": KeyMoment(machine, "RUNNING", normal),
        "warning": KeyMoment(machine, "WARNING", warning_ts),
        "fault": KeyMoment(machine, "FAULT", fault_ts),
        "waiting": KeyMoment(w_m, w_st, w_ts),
    }


# ---------------------------------------------------------------- stacks
@dataclass
class Stack:
    api_url: str
    web_url: str
    workdir: Path
    procs: list[subprocess.Popen[Any]]
    env: dict[str, str]

    def close(self) -> None:
        for p in reversed(self.procs):
            stop(p)


def start_vite(workdir: Path, api_url: str, port: int) -> tuple[subprocess.Popen[Any], str]:
    log = workdir / "vite.log"
    env = clean_env({"VITE_LINETWIN_API_URL": api_url})
    vite = FRONTEND / "node_modules" / "vite" / "bin" / "vite.js"
    proc = subprocess.Popen(
        [node(), str(vite), "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
        cwd=FRONTEND,
        env=env,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}/"
    wait_http(url, proc, log)
    return proc, url


@contextlib.contextmanager
def minimal_stack(workdir: Path) -> Iterator[tuple[Stack, dict[str, KeyMoment]]]:
    """Minimal mode (file + SQLite, no Docker), the replay fully ingested, plus Vite."""
    workdir.mkdir(parents=True, exist_ok=True)
    data = generate_replay_file(workdir / "data")
    moments = key_moments(data)
    api_port, web_port = free_port(), free_port()
    origin = f"http://127.0.0.1:{web_port}"
    log = workdir / "backend.log"
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "twin_backend",
            "serve",
            "--mode",
            "minimal",
            "--file",
            str(data),
            "--replay-speed",
            str(MINIMAL_REPLAY_SPEED),
            "--no-loop",
            "--sqlite-path",
            str(workdir / "store.sqlite"),
            "--port",
            str(api_port),
            "--cors-origin",
            origin,
        ],
        cwd=REPO_ROOT,
        env=clean_env(),
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
    )
    stack = Stack(f"http://127.0.0.1:{api_port}", "", workdir, [backend], {})
    try:
        wait_http(f"{stack.api_url}/health", backend, log)
        wait_for_eof(stack.api_url, backend, log)
        vite, web = start_vite(workdir, stack.api_url, web_port)
        stack.procs.append(vite)
        stack.web_url = web
        yield stack, moments
    finally:
        stack.close()


def wait_for_eof(api_url: str, proc: subprocess.Popen[Any], log: Path) -> None:
    """Until the file source has replayed the whole file (status EOF)."""
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited with {proc.returncode}; see {log}")
        status = (get_json(f"{api_url}/health").get("source") or {}).get("status")
        if status == "EOF":
            return
        time.sleep(POLL_S)
    raise RuntimeError(f"the replay did not finish within {READY_TIMEOUT_S:.0f} s; see {log}")


def pg_env(dsn: str) -> dict[str, str]:
    u = urlparse(dsn)
    return {
        "LINETWIN_PG_HOST": u.hostname or "127.0.0.1",
        "LINETWIN_PG_PORT": str(u.port or 5432),
        "LINETWIN_PG_DB": u.path.lstrip("/"),
        "LINETWIN_PG_USER": u.username or "",
        "LINETWIN_PG_PASSWORD": u.password or "",
    }


@contextlib.contextmanager
def dev_stack(workdir: Path, dsn: str, operator_key: str) -> Iterator[Stack]:
    """Mock machine server + backend (dev mode, TimescaleDB) + Vite.

    The mock server is started here, not by devstack (``--no-sim``): a child of devstack would
    outlive it on Windows, where terminating a process does not run its cleanup.
    """
    from twin_backend.devstack import port_open
    from twin_common.config import load_sources_config

    workdir.mkdir(parents=True, exist_ok=True)
    sources = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
    if sources.mock is None:
        raise RuntimeError("sources.yaml has no mock section")
    mock = urlparse(sources.mock.base_url)
    mock_host, mock_port = mock.hostname or "127.0.0.1", mock.port or 80
    if port_open(mock_host, mock_port):
        raise RuntimeError(
            f"{sources.mock.base_url} is in use (a running `uv run poe dev`?); stop it first"
        )
    api_port, web_port = free_port(), free_port()
    origin = f"http://127.0.0.1:{web_port}"
    env = clean_env({**pg_env(dsn), "LINETWIN_API_KEYS_OPERATOR": f"{E2E_OPERATOR}:{operator_key}"})
    sim_log = workdir / "mock-server.log"
    sim = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "twin_sim",
            "mock-server",
            "--scenario",
            str(DEV_SCENARIO),
            "--speed",
            str(DEV_SIM_SPEED),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=sim_log.open("w"),
        stderr=subprocess.STDOUT,
    )
    stack = Stack("", "", workdir, [sim], env)
    try:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not port_open(mock_host, mock_port):
            if sim.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"the mock server did not start; see {sim_log}")
            time.sleep(POLL_S)
        log = workdir / "devstack.log"
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "twin_backend.devstack",
                "--source",
                "mock",
                "--no-sim",
                "--port",
                str(api_port),
                "--cors-origin",
                origin,
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
        )
        stack.procs.append(backend)
        stack.api_url = f"http://127.0.0.1:{api_port}"
        wait_http(f"{stack.api_url}/health", backend, log)
        vite, web = start_vite(workdir, stack.api_url, web_port)
        stack.procs.append(vite)
        stack.web_url = web
        yield stack
    finally:
        stack.close()


def run_playwright(
    project: str, stack: Stack, extra_env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one Playwright project against `stack` (retries 0; trace on failure)."""
    cli = FRONTEND / "node_modules" / "@playwright" / "test" / "cli.js"
    env = clean_env(
        {
            "E2E_BASE_URL": stack.web_url,
            "E2E_API_URL": stack.api_url,
            "E2E_OUTPUT_DIR": str(stack.workdir / "playwright"),
            **(extra_env or {}),
        }
    )
    return subprocess.run(
        [node(), str(cli), "test", "--project", project],
        cwd=FRONTEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


# ---------------------------------------------------------------- screenshots CLI
def screenshots(minimal_only: bool) -> int:
    import tempfile

    if (reason := chromium_missing()) is not None:
        print(reason, file=sys.stderr)
        return 1
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    with tempfile.TemporaryDirectory(prefix="linetwin_shots_") as tmp:
        work = Path(tmp)
        with minimal_stack(work / "minimal") as (stack, moments):
            res = run_playwright(
                "shots-minimal",
                stack,
                {
                    "E2E_MOMENTS": json.dumps({k: v.as_dict() for k, v in moments.items()}),
                    "E2E_SHOTS_DIR": str(SCREENSHOT_DIR),
                },
            )
            print(res.stdout[-4000:])
            if res.returncode != 0:
                print(res.stderr[-4000:], file=sys.stderr)
                failed = True
        if not minimal_only:
            failed = _dev_screenshots(work / "dev") or failed
    summary = {
        "source": "real backend via twin_backend.e2e_stack (no recorded data)",
        "minimalReplayFile": f"generate --quick, run {MINIMAL_RUN_INDEX}",
        "minimalMoments": {k: v.as_dict() for k, v in moments.items()},
        "devMode": not minimal_only,
        "screenshots": sorted(f.stem for f in SCREENSHOT_DIR.glob("*.png")),
        "ok": not failed,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (SCREENSHOT_DIR / "summary.json").write_text(text, encoding="utf-8")
    return 1 if failed else 0


def _dev_screenshots(work: Path) -> bool:
    import secrets

    from twin_backend.testing import (
        TEST_DB_PREFIX,
        _create_test_db,
        _drop_test_db,
        probe_pg,
        run_in_thread,
    )
    from twin_common.config.settings import TwinSettings

    settings = TwinSettings(_env_file=REPO_ROOT / ".env")
    if (reason := probe_pg(settings.pg_dsn)) is not None:
        print(
            f"TimescaleDB not reachable ({reason}); start services with: uv run poe up "
            "(or pass --minimal-only)",
            file=sys.stderr,
        )
        return True
    name = f"{TEST_DB_PREFIX}{secrets.token_hex(6)}"
    test_dsn = settings.model_copy(update={"pg_db": name}).pg_dsn
    run_in_thread(_create_test_db(settings.pg_dsn, name, test_dsn))
    key = secrets.token_hex(16)
    try:
        with dev_stack(work, test_dsn, key) as stack:
            res = run_playwright(
                "shots-dev", stack, {"E2E_OPERATOR_KEY": key, "E2E_SHOTS_DIR": str(SCREENSHOT_DIR)}
            )
            print(res.stdout[-4000:])
            if res.returncode != 0:
                print(res.stderr[-4000:], file=sys.stderr)
                return True
    finally:
        run_in_thread(_drop_test_db(settings.pg_dsn, name))
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m twin_backend.e2e_stack", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("screenshots", help="dashboard screenshots from the real backend")
    s.add_argument(
        "--minimal-only", action="store_true", help="skip the dev-mode (TimescaleDB) shots"
    )
    args = p.parse_args(argv)
    if args.command == "screenshots":
        return screenshots(args.minimal_only)
    return 2


if __name__ == "__main__":
    sys.exit(main())

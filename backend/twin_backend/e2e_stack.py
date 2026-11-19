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
# Dev mode with live inference: real time. Explanations run beside the prediction path and are
# attached only while younger than inference.yaml prediction_stale_s on the event-time axis
# (ADR 0030). With Chromium's software WebGL on the same CPU they took 1-5 s each here (vs
# 0.3-0.9 s alone, live_inference.md); at 4x that is 4-20 s of event time and every explanation
# was dropped, so no part was highlighted. A 720 s fault (the shortest training progression,
# simulator.yaml) reaches WARNING in about 9 minutes.
PREDICT_SIM_SPEED = 1.0
PREDICT_PROGRESSION_S = 720.0
PREDICT_PLAYWRIGHT_TIMEOUT_S = 1500
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


def wait_for_inference(
    api_url: str, proc: subprocess.Popen[Any], log: Path, key: str | None = None
) -> None:
    """Until ``GET /predict`` reports the inference workers ready (they load torch / XGBoost)."""
    from twin_common.config.inference import load_inference_config

    start_s = load_inference_config(REPO_ROOT / "config" / "inference.yaml").worker.start_timeout_s
    deadline = time.monotonic() + start_s + READY_TIMEOUT_S
    state = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited with {proc.returncode}; see {log}")
        info = get_json(f"{api_url}/predict", key).get("inference") or {}
        state = info.get("state")
        if state == "ready":
            return
        if state == "failed":
            raise RuntimeError(f"live inference failed ({info.get('reason')}); see {log}")
        time.sleep(POLL_S)
    raise RuntimeError(f"live inference not ready (state {state}); see {log}")


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
def dev_stack(
    workdir: Path,
    dsn: str,
    operator_key: str,
    *,
    inference: bool = False,
    sim_speed: float = DEV_SIM_SPEED,
) -> Iterator[Stack]:
    """Mock machine server + backend (dev mode, TimescaleDB) + Vite.

    The mock server is started here, not by devstack (``--no-sim``): a child of devstack would
    outlive it on Windows, where terminating a process does not run its cleanup.
    ``inference`` adds ``--inference`` (Phase 11 live inference with the deployed bundles of
    ``config/inference.yaml``); the stack is yielded once ``GET /predict`` reports it running.
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
            str(sim_speed),
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
                *(["--inference"] if inference else []),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
        )
        stack.procs.append(backend)
        stack.api_url = f"http://127.0.0.1:{api_port}"
        wait_http(f"{stack.api_url}/health", backend, log)
        if inference:
            wait_for_inference(stack.api_url, backend, log, operator_key)
        vite, web = start_vite(workdir, stack.api_url, web_port)
        stack.procs.append(vite)
        stack.web_url = web
        yield stack
    finally:
        stack.close()


def run_playwright(
    project: str,
    stack: Stack,
    extra_env: Mapping[str, str] | None = None,
    timeout_s: float = 900,
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
        timeout=timeout_s,
    )


# ---------------------------------------------------------------- screenshots CLI
def screenshots(minimal_only: bool, predict: bool = True, predict_only: bool = False) -> int:
    """All dashboard shots; ``predict_only`` reruns only the prediction panel shots and keeps
    the rest of ``summary.json``."""
    import tempfile

    if (reason := chromium_missing()) is not None:
        print(reason, file=sys.stderr)
        return 1
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SCREENSHOT_DIR / "summary.json"
    predict_skipped: str | None = None
    if minimal_only or not predict:
        predict_skipped = "not requested"
    elif (missing := bundles_missing()) is not None:
        predict_skipped = missing
        print(f"prediction shots skipped: {missing}", file=sys.stderr)
        if predict_only:
            return 1
    prev: dict[str, Any] = {}
    if predict_only and summary_path.exists():
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
    moments_out = prev.get("minimalMoments")
    base_failed = False
    predict_failed = False
    # Inference workers exit on stdin EOF just after their backend; on Windows they may still hold
    # the backend log for a moment when the folder is removed.
    with tempfile.TemporaryDirectory(prefix="linetwin_shots_", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        if not predict_only:
            with minimal_stack(work / "minimal") as (stack, moments):
                base_failed = _report(
                    run_playwright(
                        "shots-minimal",
                        stack,
                        {
                            "E2E_MOMENTS": json.dumps({k: v.as_dict() for k, v in moments.items()}),
                            "E2E_SHOTS_DIR": str(SCREENSHOT_DIR),
                        },
                    )
                )
            moments_out = {k: v.as_dict() for k, v in moments.items()}
            if not minimal_only:
                base_failed = _dev_screenshots(work / "dev") or base_failed
        if predict_skipped is None:
            predict_failed = _predict_screenshots(work / "predict")
    # ok per part, so --predict-only replaces only the prediction result
    base_ok = bool(prev.get("baseOk", prev.get("ok", True))) if predict_only else not base_failed
    predict_ok = None if predict_skipped is not None else not predict_failed
    summary = {
        "source": "real backend via twin_backend.e2e_stack (no recorded data)",
        "minimalReplayFile": f"generate --quick, run {MINIMAL_RUN_INDEX}",
        "minimalMoments": moments_out,
        "devMode": prev.get("devMode", False) if predict_only else not minimal_only,
        "predictMode": (
            {
                "simSpeed": PREDICT_SIM_SPEED,
                "fault": "bearing_wear on CNC-01",
                "progressionS": PREDICT_PROGRESSION_S,
            }
            if predict_skipped is None
            else {"skipped": predict_skipped}
        ),
        "screenshots": sorted(f.stem for f in SCREENSHOT_DIR.glob("*.png")),
        "baseOk": base_ok,
        "predictOk": predict_ok,
        "ok": base_ok and predict_ok is not False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if (base_failed or predict_failed) else 0


@contextlib.contextmanager
def throwaway_db() -> Iterator[str | None]:
    """A fresh TimescaleDB test database (dropped afterwards); None when it is not reachable."""
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
        yield None
        return
    name = f"{TEST_DB_PREFIX}{secrets.token_hex(6)}"
    test_dsn = settings.model_copy(update={"pg_db": name}).pg_dsn
    run_in_thread(_create_test_db(settings.pg_dsn, name, test_dsn))
    try:
        yield test_dsn
    finally:
        run_in_thread(_drop_test_db(settings.pg_dsn, name))


def _report(res: subprocess.CompletedProcess[str]) -> bool:
    """Print a Playwright run's output; True when it failed."""
    print(res.stdout[-4000:])
    if res.returncode != 0:
        print(res.stderr[-4000:], file=sys.stderr)
        return True
    return False


def _dev_screenshots(work: Path) -> bool:
    import secrets

    with throwaway_db() as dsn:
        if dsn is None:
            return True
        key = secrets.token_hex(16)
        with dev_stack(work, dsn, key) as stack:
            return _report(
                run_playwright(
                    "shots-dev",
                    stack,
                    {"E2E_OPERATOR_KEY": key, "E2E_SHOTS_DIR": str(SCREENSHOT_DIR)},
                )
            )


def bundles_missing() -> str | None:
    """None when the bundles of config/inference.yaml are registered, else the reason."""
    from twin_common.config.inference import load_inference_config

    cfg = load_inference_config(REPO_ROOT / "config" / "inference.yaml")
    models = REPO_ROOT / cfg.tracking_dir / "models"
    names = [cfg.models[cfg.model].registered, cfg.anomaly.registered]
    absent = [n for n in names if not (models / n).is_dir()]
    if absent:
        return (
            f"deployed bundles not registered ({', '.join(absent)}); build them with: "
            "uv run poe deploy-models"
        )
    return None


def _predict_screenshots(work: Path) -> bool:
    """Prediction panel shots: dev mode + live inference, bearing_wear on CNC-01."""
    import secrets

    with throwaway_db() as dsn:
        if dsn is None:
            return True
        key = secrets.token_hex(16)
        with dev_stack(work, dsn, key, inference=True, sim_speed=PREDICT_SIM_SPEED) as stack:
            return _report(
                run_playwright(
                    "shots-predict",
                    stack,
                    {
                        "E2E_OPERATOR_KEY": key,
                        "E2E_SHOTS_DIR": str(SCREENSHOT_DIR),
                        "E2E_PROGRESSION_S": f"{PREDICT_PROGRESSION_S:g}",
                    },
                    timeout_s=PREDICT_PLAYWRIGHT_TIMEOUT_S,
                )
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m twin_backend.e2e_stack", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("screenshots", help="dashboard screenshots from the real backend")
    s.add_argument(
        "--minimal-only", action="store_true", help="skip the dev-mode (TimescaleDB) shots"
    )
    s.add_argument(
        "--no-predict",
        action="store_true",
        help="skip the prediction panel shots (dev mode + live inference, needs deploy-models)",
    )
    s.add_argument(
        "--predict-only",
        action="store_true",
        help="only the prediction panel shots (still runs the minimal-mode shots first)",
    )
    args = p.parse_args(argv)
    if args.command == "screenshots":
        return screenshots(args.minimal_only, not args.no_predict, args.predict_only)
    return 2


if __name__ == "__main__":
    sys.exit(main())

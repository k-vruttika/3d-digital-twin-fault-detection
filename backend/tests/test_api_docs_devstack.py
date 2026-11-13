"""docs/api.md matches the OpenAPI schema, and the dev launcher (``uv run poe dev``)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from twin_backend import devstack
from twin_backend.api.docs import BEGIN, END, build_app, render, splice
from twin_backend.testing import REPO_ROOT, free_port
from twin_common.config import TwinSettings, load_sources_config
from twin_common.enums import AdapterKind


def test_api_md_is_up_to_date() -> None:
    path = REPO_ROOT / "docs" / "api.md"
    current = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert BEGIN in current
    assert END in current
    expected = splice(current, render(build_app()))
    assert current == expected, "docs/api.md is stale: run `uv run poe api-docs`"


def test_generated_reference_lists_every_route_with_its_role() -> None:
    text = render(build_app())
    for line in (
        "##### `POST /commands`",
        "##### `POST /scenario`",
        "##### `GET /replay`",
        "##### `POST /alerts/{alert_id}/ack`",
    ):
        assert line in text
    assert "Send a command (operator); answers with the final ack. Role: **operator**." in text
    assert "Liveness and wiring (no key needed). Role: **none (no key needed)**." in text


# ---------------------------------------------------------------- dev launcher
SOURCES = load_sources_config(REPO_ROOT / "config" / "sources.yaml")
SETTINGS = TwinSettings(_env_file=None)


def test_publisher_commands_per_source() -> None:
    line = Path("config/line.yaml")
    for kind, sub in (
        (AdapterKind.MOCK, "mock-server"),
        (AdapterKind.MQTT, "mqtt"),
        (AdapterKind.OPCUA, "opcua"),
    ):
        cmd = devstack.publisher_command(
            kind, line=line, scenario=None, speed=5.0, duration=None, seed=None
        )
        assert cmd == [sys.executable, "-m", "twin_sim", sub, "--line", str(line), "--speed", "5"]
    full = devstack.publisher_command(
        AdapterKind.OPCUA, line=line, scenario=Path("s.yaml"), speed=1, duration=60, seed=3
    )
    assert full is not None
    assert full[-6:] == ["--scenario", "s.yaml", "--duration", "60", "--seed", "3"]
    assert (
        devstack.publisher_command(
            AdapterKind.FILE, line=line, scenario=None, speed=1, duration=None, seed=None
        )
        is None
    )


def test_required_services_and_ready_addresses() -> None:
    assert [n for n, _, _ in devstack.required_services(AdapterKind.FILE, SOURCES, SETTINGS)] == [
        "TimescaleDB"
    ]
    names = [n for n, _, _ in devstack.required_services(AdapterKind.MQTT, SOURCES, SETTINGS)]
    assert names == ["TimescaleDB", "Mosquitto (MQTT broker)"]
    assert devstack.ready_address(AdapterKind.MOCK, SOURCES) == ("127.0.0.1", 8100)
    assert devstack.ready_address(AdapterKind.OPCUA, SOURCES) == ("127.0.0.1", 4840)
    assert devstack.ready_address(AdapterKind.MQTT, SOURCES) is None


class FakeProc:
    def __init__(self, cmd: list[str], *, exits: bool = False) -> None:
        self.cmd = cmd
        self.exits = exits
        self.terminated = False

    def poll(self) -> int | None:
        return 1 if self.exits or self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


def test_main_starts_publisher_serves_and_always_stops_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    procs: list[FakeProc] = []

    def popen(cmd: list[str]) -> FakeProc:
        procs.append(FakeProc(cmd))
        return procs[-1]

    seen: list[argparse.Namespace] = []

    def serve(ns: argparse.Namespace) -> int:
        seen.append(ns)
        assert not procs[0].terminated  # the publisher runs while the backend serves
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        devstack.main(
            ["--source", "opcua", "--speed", "4"],
            popen=popen,  # type: ignore[arg-type]
            serve=serve,
            probe=lambda _h, _p: True,
        )
    assert procs[0].cmd[3] == "opcua"
    assert procs[0].terminated  # stopped in finally, even on an exception
    assert (seen[0].mode, seen[0].source) == ("dev", "opcua")
    assert "starting the opcua simulator publisher" in capsys.readouterr().err


def test_main_refuses_without_services_and_reports_dead_publisher(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[Any] = []

    def record(cmd: list[str]) -> Any:
        calls.append(cmd)

    rc = devstack.main(
        ["--source", "mqtt"],
        popen=record,
        serve=lambda ns: 0,
        probe=lambda _h, _p: False,
    )
    assert rc == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "TimescaleDB" in err
    assert "Mosquitto" in err
    assert "uv run poe up" in err
    rc = devstack.main(
        ["--source", "mock"],
        popen=lambda cmd: FakeProc(cmd, exits=True),  # type: ignore[arg-type,return-value]
        serve=lambda ns: 0,
        probe=lambda _h, _p: True,
    )
    assert rc == 1
    assert "did not become ready" in capsys.readouterr().err
    served: list[str] = []

    def no_popen(cmd: list[str]) -> Any:
        pytest.fail("no publisher for the file source")

    def serve_file(ns: argparse.Namespace) -> int:
        served.append(ns.source)
        return 0

    rc = devstack.main(
        ["--source", "file"], popen=no_popen, serve=serve_file, probe=lambda _h, _p: True
    )
    assert rc == 0
    assert served == ["file"]


def test_smoke_real_mock_publisher_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real `python -m twin_sim mock-server` child is started, reachable while the backend
    would serve, and stopped afterwards (TimescaleDB is faked as reachable)."""
    config = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "config", config)
    port = free_port()
    data = yaml.safe_load((config / "sources.yaml").read_text(encoding="utf-8"))
    data["mock"]["base_url"] = f"http://127.0.0.1:{port}"
    (config / "sources.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("LINETWIN_CONFIG_DIR", str(config))
    monkeypatch.chdir(REPO_ROOT)
    started: list[subprocess.Popen[Any]] = []

    def popen(cmd: list[str]) -> subprocess.Popen[Any]:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(proc)
        return proc

    settings = TwinSettings()

    def probe(host: str, p: int) -> bool:
        return True if p == settings.pg_port else devstack.port_open(host, p)

    def serve(_ns: argparse.Namespace) -> int:
        info = httpx.get(f"http://127.0.0.1:{port}/info", timeout=5).json()
        assert "CNC-01" in str(info)
        return 0

    try:
        assert (
            devstack.main(
                ["--source", "mock", "--speed", "5"], popen=popen, serve=serve, probe=probe
            )
            == 0
        )
        assert started
        assert started[0].poll() is not None  # terminated
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()

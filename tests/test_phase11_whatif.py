"""Phase 11 Part C: what-if projections end to end (ADR 0031).

A simulated line runs behind the mock machine server (20x), the backend (real uvicorn) ingests
it, and ``POST /whatif`` forks the simulator behind the source:

* without inference: projections of the simulator only (baseline and scenario, ghost states, a
  note why there are no predictions), the safety rules and limits apply, 404 / 409 / 422;
* a source without the capability (file replay) answers 409;
* with inference (the quick test bundles): both projections are scored by a separate worker
  pool, the evaluated alarm rule runs over them, and the live inference is not disturbed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from twin_backend.api.app import create_app
from twin_backend.api.server import ServerHandle
from twin_backend.runtime import Runtime, build_runtime
from twin_backend.testing import (
    REPO_ROOT,
    SimExport,
    file_runtime,
    settings_for,
    storage_config,
)
from twin_common.config import load_line_config
from twin_common.config.inference import InferenceConfig
from twin_common.config.loader import read_yaml
from twin_common.config.sources import SourcesConfig
from twin_common.enums import AdapterKind
from twin_sim.config import load_scenario, load_sim_config
from twin_sim.publishers.http import ServerHandle as SimServerHandle
from twin_sim.publishers.mock_server import create_app as create_mock_app
from twin_sim.publishers.session import SimSession, build_simulator

STARTUP_TIMEOUT_S = 240.0
SPEED = 20.0
WAIT_S = 60.0
CONFIG = REPO_ROOT / "config"
FAST_RECONNECT = {"initial_s": 0.1, "max_s": 0.5, "jitter": 0.0}
WARM_S = 75  # simulated seconds before a what-if has a full model window behind it


def quick_inference_config(tracking: Path, cache: Path) -> dict[str, Any]:
    """config/inference.yaml with the quick test bundles (as tests/test_phase11_inference.py)."""
    data = read_yaml(CONFIG / "inference.yaml")
    data |= {
        "enabled": True,
        "model": "gru",
        "models": {"gru": {"registered": "test-gru", "family": "nn"}},
        "anomaly": {"registered": "test-autoencoder", "version": "latest"},
        "tracking_dir": str(tracking),
    }
    data["worker"]["cache_dir"] = str(cache)
    return data


@dataclass
class Stack:
    rt: Runtime
    api: ServerHandle
    session: SimSession
    http: httpx.AsyncClient


def _session() -> SimSession:
    """The full line, no scripted commands or faults: the whole line runs steadily."""
    line = load_line_config(CONFIG / "line.yaml")
    scenario = load_scenario(CONFIG / "scenarios" / "cnc_stop.yaml").model_copy(
        update={"commands": [], "duration_s": 36000.0}
    )
    scenario = scenario.model_copy(
        update={"recovery": scenario.recovery.model_copy(update={"auto": False})}
    )
    sim = build_simulator(line, load_sim_config(CONFIG / "simulator.yaml"), scenario, 7)
    return SimSession(sim, SPEED, 20_000)


@asynccontextmanager
async def mock_stack(
    sources: SourcesConfig, tmp: Path, inference: InferenceConfig | None = None
) -> AsyncIterator[Stack]:
    session = _session()
    sim_server = SimServerHandle(create_mock_app(session))
    await sim_server.start()
    assert sources.mock is not None
    srcs = sources.model_copy(
        update={
            "active": AdapterKind.MOCK,
            "reconnect": sources.reconnect.model_copy(update=FAST_RECONNECT),
            "mock": sources.mock.model_copy(update={"base_url": sim_server.base_url}),
        }
    )
    session.start()
    base = storage_config()
    st = base.model_copy(
        update={"sqlite": base.sqlite.model_copy(update={"path": str(tmp / "whatif.sqlite")})}
    )
    rt = build_runtime(
        settings_for(),
        root=REPO_ROOT,
        sources=srcs,
        storage_cfg=st,
        inference_cfg=inference,
    )
    api = ServerHandle(create_app(rt))
    await api.start()
    http = httpx.AsyncClient(base_url=api.base_url, timeout=120)
    try:
        yield Stack(rt, api, session, http)
    finally:
        await http.aclose()
        await api.stop()
        await sim_server.stop()
        await session.stop()


async def poll(fn: Callable[[], Awaitable[Any]], what: str, timeout: float = WAIT_S) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = await fn()
        if last:
            return last
        await asyncio.sleep(0.1)
    pytest.fail(f"timed out waiting for {what} (last: {last!r})")


async def warm(s: Stack) -> None:
    async def ready() -> bool:
        ms = {m["machineId"]: m for m in (await s.http.get("/machines")).json()["machines"]}
        return s.session.sim.t >= WARM_S and all(m["state"] == "RUNNING" for m in ms.values())

    await poll(ready, "a warm, running line")


def body(ctype: str, machine: str = "CNC-01", horizon: int = 60, **params: float) -> dict[str, Any]:
    return {"machineId": machine, "type": ctype, "params": params, "horizonS": horizon}


async def test_projection_without_inference_and_the_rules_around_it(
    sources_cfg: SourcesConfig, tmp_path: Path
) -> None:
    async with mock_stack(sources_cfg, tmp_path) as s:
        await warm(s)
        live_before = (await s.http.get("/commands")).json()["commands"]
        r = await s.http.post("/whatif", json=body("STOP"))
        assert r.status_code == 200, r.text
        w = r.json()
        assert w["machineId"] == "CNC-01"
        assert w["horizonS"] == 60
        assert w["ghostHorizonsS"] == [10, 30]
        assert "without predictions" in w["modelNote"]
        assert "model" not in w
        # the history ends at the fork point, the projections start after it
        assert all(p["tS"] <= 0 for p in w["history"]["CNC-01"])
        for variant in ("baseline", "scenario"):
            pts = w[variant]["machines"]["CNC-01"]["points"]
            assert [p["tS"] for p in pts] == list(range(1, 61))
        base = w["baseline"]["machines"]["CNC-01"]
        scen = w["scenario"]["machines"]["CNC-01"]
        assert base["ghost"]["30"] == "RUNNING"
        assert scen["ghost"]["10"] == "IDLE"
        assert scen["ghost"]["30"] == "IDLE"
        assert scen["ghost"]["end"] == "IDLE"
        # the stop propagates: the inspection station starves in the projection only
        assert w["scenario"]["machines"]["INSP-01"]["ghost"]["end"] == "STARVED"
        assert w["baseline"]["machines"]["INSP-01"]["ghost"]["end"] == "RUNNING"
        # nothing was sent to the machine and nothing recorded as a command
        assert (await s.http.get("/commands")).json()["commands"] == live_before
        ms = {m["machineId"]: m for m in (await s.http.get("/machines")).json()["machines"]}
        assert ms["CNC-01"]["state"] == "RUNNING"

        # the same safety rules and limits as a command; unknown machine 404; bad body 422
        over = await s.http.post("/whatif", json=body("SET_SPEED", speed=99999.0))
        assert over.status_code == 422
        assert "outside" in over.json()["detail"]
        missing = await s.http.post("/whatif", json=body("SET_SPEED"))
        assert missing.status_code == 422
        assert "speed" in missing.json()["detail"]
        too_long = await s.http.post("/whatif", json=body("STOP", horizon=100000))
        assert too_long.status_code == 422
        assert "maximum" in too_long.json()["detail"]
        assert (await s.http.post("/whatif", json=body("STOP", machine="NOPE"))).status_code == 404
        # a setpoint change is projected too
        ok = await s.http.post("/whatif", json=body("SET_SPEED", speed=3000.0, horizon=30))
        assert ok.status_code == 200, ok.text


async def test_file_source_has_no_whatif(sim_export: SimExport, tmp_path: Path) -> None:
    rt = file_runtime(sim_export, tmp_path, settings=settings_for())
    api = ServerHandle(create_app(rt))
    await api.start()
    try:
        await rt.manager.wait_idle()
        async with httpx.AsyncClient(base_url=api.base_url) as c:
            r = await c.post("/whatif", json=body("STOP"))
            assert r.status_code == 409
            assert "does not support what-if" in r.json()["detail"]
    finally:
        await api.stop()


async def test_projection_scored_by_the_model(
    serve_setup: Any, sources_cfg: SourcesConfig, tmp_path: Path
) -> None:
    cfg = InferenceConfig.model_validate(quick_inference_config(serve_setup[0], tmp_path / "cache"))
    async with mock_stack(sources_cfg, tmp_path, cfg) as s:
        assert s.rt.inference is not None
        assert s.rt.whatif is not None
        await s.rt.inference.start()
        assert await s.rt.inference.wait_ready(STARTUP_TIMEOUT_S), s.rt.inference.status
        await warm(s)
        live_pool = s.rt.inference.pool
        r = await s.http.post("/whatif", json=body("STOP", horizon=60))
        assert r.status_code == 200, r.text
        w = r.json()
        assert "modelNote" not in w
        assert w["model"]
        assert w["primaryHorizonS"] == 30
        scored = [p for p in w["history"]["CNC-01"] if p.get("gate") == "predict"]
        assert scored, "the history fills the model window, so some history points are scored"
        for variant in ("baseline", "scenario"):
            pts = w[variant]["machines"]["CNC-01"]["points"]
            assert all("gate" in p for p in pts)
            for p in pts:
                if p["gate"] == "predict":
                    assert 0.0 <= p["pFault"] <= 1.0
                    assert 0.0 <= p["pWarning"] <= 1.0
                    assert p["topClass"] in ("NORMAL", "WARNING", "FAULT")
                    assert isinstance(p["alarm"], bool)
        # the stopped machine is gated off in the projection: no prediction while stopped
        stopped = [p for p in w["scenario"]["machines"]["CNC-01"]["points"] if p["state"] == "IDLE"]
        assert stopped
        assert all(p["gate"] != "predict" for p in stopped[1:])
        # a second request reuses the workers (the live pool is a different set of processes)
        again = await s.http.post("/whatif", json=body("STOP", horizon=30))
        assert again.status_code == 200
        assert s.rt.whatif.scorer is not None
        assert s.rt.whatif.scorer.pool is not live_pool
        assert s.rt.whatif.scorer.pool.alive
        assert live_pool.alive
        assert s.rt.inference.ready

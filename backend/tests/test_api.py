"""REST API and WebSocket stream on a real uvicorn server (file source + event store).

Covers auth (roles, no-key rules, WebSocket first-message auth), every Part A endpoint,
``/replay?t=`` == ``reconstruct(t)``, ``/history`` == store queries (both backends), CORS,
the 501 stubs and ``/metrics``.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets
from pydantic import SecretStr

from twin_backend.api.app import create_app
from twin_backend.api.auth import Authenticator, StartupRefusedError, is_loopback
from twin_backend.api.server import ServerHandle
from twin_backend.api.ws import CLOSE_UNAUTHORIZED
from twin_backend.api.ws_models import SERVER_ADAPTER
from twin_backend.storage.base import Stream
from twin_backend.storage.export import order_readings, readings_to_csv
from twin_backend.testing import SimExport, file_runtime, settings_for
from twin_common.config.settings import parse_api_keys
from twin_common.enums import Role

OP_KEY = "op-key-0123456789abcdef"
VIEW_KEY = "viewer-key-0123456789abcdef"
KEYS = {
    "api_keys_operator": SecretStr(f"alice:{OP_KEY}"),
    "api_keys_viewer": SecretStr(f"victor:{VIEW_KEY}"),
}
FAR = {"from": "2000-01-01T00:00:00Z", "to": "2100-01-01T00:00:00Z"}


@asynccontextmanager
async def running_api(
    export: SimExport,
    tmp: Path,
    *,
    backend: str = "sqlite",
    dsn: str | None = None,
    keys: bool = True,
    wait_idle: bool = True,
    **runtime_kw: Any,
) -> AsyncIterator[tuple[Any, ServerHandle]]:
    rt = file_runtime(
        export,
        tmp,
        backend,
        dsn,
        settings=settings_for(dsn, **(KEYS if keys else {})),
        **runtime_kw,
    )
    server = ServerHandle(create_app(rt))
    await server.start()
    try:
        if wait_idle:
            await rt.manager.wait_idle()
            await rt.engine.drain()
            await rt.storage.flush()
        yield rt, server
    finally:
        await server.stop()


def client(server: ServerHandle, key: str | None = VIEW_KEY) -> httpx.AsyncClient:
    headers = {"X-API-Key": key} if key else {}
    return httpx.AsyncClient(base_url=server.base_url, headers=headers, timeout=10.0)


# ---------------------------------------------------------------- auth units
def test_loopback_and_startup_rules() -> None:
    assert is_loopback("127.0.0.1")
    assert is_loopback("localhost")
    assert is_loopback("::1")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("192.168.1.5")
    with pytest.raises(StartupRefusedError, match="LINETWIN_API_KEYS_OPERATOR"):
        Authenticator([], bind_host="0.0.0.0")
    keys = parse_api_keys(f"alice:{OP_KEY}", Role.OPERATOR)
    auth = Authenticator(keys, bind_host="0.0.0.0")  # keys configured: any bind is fine
    assert not auth.anonymous_read
    assert auth.resolve(None) is None
    assert auth.resolve("wrong") is None
    p = auth.resolve(OP_KEY)
    assert p is not None
    assert p.name == "alice"
    assert p.has(Role.OPERATOR)
    open_auth = Authenticator([], bind_host="127.0.0.1")
    anon = open_auth.resolve(None)
    assert anon is not None
    assert anon.anonymous
    assert not anon.has(Role.OPERATOR)
    with pytest.raises(StartupRefusedError, match="duplicate"):
        Authenticator(parse_api_keys("a:k1,a:k2", Role.VIEWER), bind_host="127.0.0.1")


def test_parse_api_keys() -> None:
    keys = parse_api_keys(" alice:k1 , k2,, ", Role.VIEWER)
    assert [(k.principal, k.secret.get_secret_value()) for k in keys] == [
        ("alice", "k1"),
        ("viewer-2", "k2"),
    ]
    with pytest.raises(ValueError, match="malformed"):
        parse_api_keys("alice:", Role.OPERATOR)


# ---------------------------------------------------------------- REST
async def test_role_matrix(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (_rt, server):
        async with client(server, None) as anon:
            assert (await anon.get("/machines")).status_code == 401  # keys configured
            assert (await anon.get("/health")).status_code == 200  # liveness needs no key
        async with client(server, "nope") as bad:
            assert (await bad.get("/machines")).status_code == 401
        async with client(server, VIEW_KEY) as viewer:
            assert (await viewer.get("/machines")).status_code == 200
            r = await viewer.post("/sources/switch", json={"kind": "mock"})
            assert r.status_code == 403
            assert "operator key is required" in r.json()["detail"]
            assert (await viewer.post("/alerts/x/ack")).status_code == 403
        async with client(server, OP_KEY) as op:
            assert (await op.post("/alerts/nope/ack")).status_code == 404


async def test_no_keys_reads_open_writes_refused(sim_export: SimExport, tmp_path: Path) -> None:
    async with (
        running_api(sim_export, tmp_path, keys=False) as (_rt, server),
        client(server, None) as anon,
    ):
        assert (await anon.get("/machines")).status_code == 200
        r = await anon.post("/sources/switch", json={"kind": "mock"})
        assert r.status_code == 403
        assert "LINETWIN_API_KEYS_OPERATOR" in r.json()["detail"]
        health = (await anon.get("/health")).json()
        assert health["auth"] == {
            "keysConfigured": False,
            "anonymousRead": True,
            "operatorKeys": False,
        }


async def test_machines_and_state(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (rt, server), client(server) as c:
        body = (await c.get("/machines")).json()
        assert body["lineId"] == "line1"
        assert [m["machineId"] for m in body["machines"]] == rt.engine.machine_ids
        live = rt.engine.snapshot()
        for m in body["machines"]:
            assert m["state"] == live.machines[m["machineId"]].state.value
            assert set(m["dataQuality"]) >= {"status", "latePct", "quarantinePct", "partialPct"}
            assert "cause" in m["rootCause"]
        one = (await c.get("/machines/CNC-01/state")).json()
        assert one["kind"] == "CNC"
        assert one["lastReading"]["machineId"] == "CNC-01"
        assert (await c.get("/machines/NOPE/state")).status_code == 404


async def test_replay_equals_reconstruct_and_history_matches_store(
    sim_export: SimExport, store_backend: str, runtime_dsn: str | None, tmp_path: Path
) -> None:
    async with (
        running_api(sim_export, tmp_path, backend=store_backend, dsn=runtime_dsn) as (rt, server),
        client(server) as c,
    ):
        store = rt.storage.store
        lo, hi = await store.min_max_ts([Stream.READINGS])
        rng = random.Random(4)
        span = (hi - lo).total_seconds()
        times = [lo + timedelta(seconds=rng.uniform(0, span)) for _ in range(8)] + [hi]
        for t in times:
            r = await c.get("/replay", params={"t": t.isoformat()})
            assert r.status_code == 200
            expected = (await rt.storage.reconstruct(t)).model_dump(mode="json")
            assert r.json() == expected
        # at the watermark the replayed state equals the live state
        live = rt.engine.snapshot()
        at_hi = (await c.get("/replay", params={"t": live.watermark().isoformat()})).json()
        assert {k: v for k, v in at_hi.items() if k != "asOf"} == live.body()

        raw = (await c.get("/history", params=FAR | {"resolution": "raw"})).json()
        stored = [s.item.model_dump(mode="json") for s in await store.fetch(Stream.READINGS)]
        assert raw["readings"] == stored
        some = (
            await c.get(
                "/history", params=FAR | {"resolution": "raw", "machines": "CNC-01,INSP-01"}
            )
        ).json()
        assert {r["machineId"] for r in some["readings"]} == {"CNC-01", "INSP-01"}
        t0 = lo.replace(second=0, microsecond=0)
        t1 = hi + timedelta(minutes=10)
        for res in ("1m", "10m"):
            body = (
                await c.get(
                    "/history",
                    params={"from": t0.isoformat(), "to": t1.isoformat(), "resolution": res},
                )
            ).json()
            buckets = await store.history(t0, t1, res)
            assert len(body["buckets"]) == len(buckets)
            for got, exp in zip(body["buckets"], buckets, strict=True):
                assert got["machineId"] == exp.machine_id
                assert got["n"] == exp.n
                assert got["nLate"] == exp.n_late
                assert got["nSuspect"] == exp.n_suspect
                for sensor, stats in exp.sensors.items():
                    assert got["sensors"][sensor.value]["count"] == stats.count
                    assert got["sensors"][sensor.value]["mean"] == pytest.approx(stats.mean)
        bad = await c.get("/history", params={"from": FAR["to"], "to": FAR["from"]})
        assert bad.status_code == 422
        naive = await c.get("/history", params={"from": "2026-01-01T00:00:00", "to": FAR["to"]})
        assert naive.status_code == 422  # timestamps need a zone
        assert (await c.get("/history", params=FAR | {"machines": "NOPE"})).status_code == 404


async def test_export_and_history_limits(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (rt, server), client(server) as c:
        r = await c.get("/export", params=FAR | {"format": "csv"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        rows = await rt.storage.store.fetch(Stream.READINGS)
        assert r.text == readings_to_csv(order_readings(rows, rt.engine.machine_ids))
        j = await c.get("/export", params=FAR | {"format": "jsonl"})
        assert len(j.text.splitlines()) == len(rows)
    small = rt.api_cfg.model_copy(update={"max_history_rows": 5})
    async with (
        running_api(sim_export, tmp_path / "small", api_cfg=small) as (_rt, server),
        client(server) as c,
    ):
        assert (await c.get("/history", params=FAR | {"resolution": "raw"})).status_code == 413
        assert (await c.get("/export", params=FAR)).status_code == 413


async def test_sources_alerts_stubs_metrics_cors(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (rt, server):
        async with client(server) as c:
            src = (await c.get("/sources")).json()
            assert src["active"] == "file"
            assert src["capabilities"] == []
            assert set(src["configured"]) == {"file", "mock", "mqtt", "opcua"}
            assert (await c.get("/alerts")).json() == {"alerts": []}
            hist = (await c.get("/alerts/history", params=FAR)).json()
            assert hist["alerts"] == [
                s.item.model_dump(mode="json") for s in await rt.storage.store.fetch(Stream.ALERTS)
            ]
            assert (await c.get("/predict")).status_code == 501
            assert (await c.post("/whatif")).status_code == 501
            metrics = await c.get("/metrics")
            assert metrics.status_code == 200
            assert "linetwin_engine_transitions_total" in metrics.text
            assert "linetwin_ingest" in metrics.text
            assert "linetwin_ws_clients" in metrics.text
            ok = await c.options(
                "/machines",
                headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
            )
            assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
            evil = await c.options(
                "/machines",
                headers={"Origin": "http://evil.test", "Access-Control-Request-Method": "GET"},
            )
            assert "access-control-allow-origin" not in evil.headers
        async with client(server, OP_KEY) as op:
            # no mock server runs: the mock adapter starts and keeps reconnecting
            r = await op.post("/sources/switch", json={"kind": "mock"})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["active"] == "mock"
            assert body["switches"] == 1
            assert set(body["capabilities"]) >= {"commands"}
            back = await op.post("/sources/switch", json={"kind": "file"})
            assert back.json()["active"] == "file"
            assert (await op.post("/sources/switch", json={"kind": "nope"})).status_code == 422


# ---------------------------------------------------------------- WebSocket
async def _recv(ws: Any, timeout: float = 10.0) -> dict[str, Any]:
    msg: dict[str, Any] = json.loads(await asyncio.wait_for(ws.recv(), timeout))
    SERVER_ADAPTER.validate_python(msg)  # every frame matches the published schema
    return msg


async def test_ws_auth_subscribe_snapshot_and_filter(sim_export: SimExport, tmp_path: Path) -> None:
    async with (
        running_api(sim_export, tmp_path, replay_speed=5.0, loop=True, wait_idle=False) as (
            _rt,
            server,
        ),
        websockets.connect(server.ws_url) as ws,
    ):
        await ws.send(json.dumps({"type": "auth", "apiKey": VIEW_KEY}))
        assert (await _recv(ws))["data"] == {"principal": "victor", "role": "viewer"}
        await ws.send(
            json.dumps({"type": "subscribe", "machines": ["CNC-01"], "channels": ["readings"]})
        )
        sub = await _recv(ws)
        assert sub["type"] == "subscribed"
        assert sub["data"]["machines"] == ["CNC-01"]
        snap = await _recv(ws)
        assert snap["type"] == "snapshot"
        assert list(snap["data"]["machines"]) == ["CNC-01"]
        assert set(snap["data"]) == {"asOf", "machines", "line", "dataQuality", "alerts"}
        for _ in range(3):
            msg = await _recv(ws)
            assert msg["type"] == "reading"
            assert msg["data"]["machineId"] == "CNC-01"
        await ws.send(json.dumps({"type": "subscribe", "channels": ["bogus"]}))
        while (err := await _recv(ws))["type"] != "error":
            pass
        assert err["data"]["channels"] == ["bogus"]
        await ws.send(json.dumps({"type": "ping"}))
        while (await _recv(ws))["type"] != "pong":
            pass


async def _read_until_closed(ws: Any) -> None:
    while True:
        await _recv(ws)


async def _closed_code(url: str, first: str | None) -> int | None:
    async with websockets.connect(url) as ws:
        if first is not None:
            await ws.send(first)
        with pytest.raises(websockets.ConnectionClosed):
            await _read_until_closed(ws)
        return ws.close_code


async def test_ws_refuses_bad_key_missing_auth_and_subscribe_first(
    sim_export: SimExport, tmp_path: Path
) -> None:
    async with running_api(sim_export, tmp_path) as (_rt, server):
        url = server.ws_url
        assert await _closed_code(url, json.dumps({"type": "auth", "apiKey": "nope"})) == (
            CLOSE_UNAUTHORIZED
        )
        assert await _closed_code(url, json.dumps({"type": "subscribe"})) == CLOSE_UNAUTHORIZED
        assert await _closed_code(url, json.dumps({"type": "auth"})) == CLOSE_UNAUTHORIZED
        assert await _closed_code(url, None) == CLOSE_UNAUTHORIZED  # auth timeout, keys set


async def test_ws_anonymous_after_timeout_only_without_keys(
    sim_export: SimExport, tmp_path: Path
) -> None:
    async with (
        running_api(sim_export, tmp_path, keys=False) as (_rt, server),
        websockets.connect(server.ws_url) as ws,
    ):
        msg = await _recv(ws, timeout=10)  # nothing sent: anonymous viewer after the timeout
        assert msg == {"type": "auth", "data": {"principal": "anonymous", "role": "viewer"}}
        await ws.send(json.dumps({"type": "subscribe", "channels": ["line"]}))
        assert (await _recv(ws))["type"] == "subscribed"
        assert (await _recv(ws))["type"] == "snapshot"

"""Phase 6 API additions for the dashboard: /auth/me, /commands/limits, /replay/range, the
/replay line view and alerts, the /sources ingest block, /scenario/options, --cors-origin."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from twin_backend.__main__ import _parser
from twin_backend.api.routes.history import alerts_as_of
from twin_backend.storage.base import Stream
from twin_backend.testing import (
    API_OP_KEY,
    API_VIEW_KEY,
    REPO_ROOT,
    SimExport,
    api_client,
    running_api,
)
from twin_common.config import load_commands_config
from twin_common.config.scenario_options import load_scenario_options
from twin_common.contracts import Alert


async def test_auth_me_per_role(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path / "keys") as (_rt, server):
        async with api_client(server, API_VIEW_KEY) as viewer:
            body = (await viewer.get("/auth/me")).json()
            assert body == {"principal": "victor", "role": "viewer", "anonymous": False}
        async with api_client(server, API_OP_KEY) as op:
            assert (await op.get("/auth/me")).json()["role"] == "operator"
        async with api_client(server, "wrong") as bad:
            assert (await bad.get("/auth/me")).status_code == 401
        async with api_client(server, None) as anon:
            assert (await anon.get("/auth/me")).status_code == 401  # keys are configured
    (tmp_path / "open").mkdir()
    async with (
        running_api(sim_export, tmp_path / "open", keys=False) as (_rt, server),
        api_client(server, None) as anon,
    ):
        body = (await anon.get("/auth/me")).json()
        assert body == {"principal": "anonymous", "role": "viewer", "anonymous": True}


async def test_command_limits_match_config(sim_export: SimExport, tmp_path: Path) -> None:
    cfg = load_commands_config(REPO_ROOT / "config" / "commands.yaml")
    async with running_api(sim_export, tmp_path) as (_rt, server), api_client(server) as c:
        r = await c.get("/commands/limits")
        assert r.status_code == 200
        body = r.json()
    assert body["alwaysAllowed"] == ["STOP"]
    for kind, lim in cfg.limits.items():
        got = body["limits"][kind.value]
        assert got["speedRpm"] == {"min": lim.speed_rpm.min, "max": lim.speed_rpm.max}
        assert got["loadPct"] == {"min": lim.load_pct.min, "max": lim.load_pct.max}
    assert [r["reason"] for r in body["rules"]] == [r.reason for r in cfg.rules]
    assert all("STOP" not in r["types"] for r in body["rules"])


async def test_replay_range_line_and_alerts(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (rt, server), api_client(server) as c:
        store = rt.storage.store
        span = await store.min_max_ts([Stream.READINGS, Stream.STATE_CHANGES])
        assert span is not None
        rng = (await c.get("/replay/range")).json()
        assert datetime.fromisoformat(rng["start"]) == span[0]
        assert datetime.fromisoformat(rng["end"]) == span[1]
        body = (await c.get("/replay", params={"t": span[1].isoformat()})).json()
        assert set(body["line"]["rootCauses"]) == set(rt.engine.machine_ids)
        expected = [a.model_dump(mode="json") for a in await alerts_as_of(store, span[1])]
        assert body["alerts"] == expected
        # before the first event nothing is known and no alert exists
        early = (await c.get("/replay", params={"t": "2000-01-01T00:00:00Z"})).json()
        assert early["alerts"] == []
        assert early["line"]["status"] == "UNKNOWN"


class _Row:
    def __init__(self, item: Alert) -> None:
        self.item = item


class _FakeStore:
    def __init__(self, alerts: list[Alert]) -> None:
        self.alerts = alerts

    async def fetch(self, stream: Stream, t0: Any, t1: Any, **_: Any) -> list[_Row]:
        assert stream is Stream.ALERTS
        assert t0 is None
        return [_Row(a) for a in self.alerts if a.ts < t1]


async def test_alerts_as_of_takes_highest_rev_and_drops_resolved() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    def alert(rev: int, *, active: bool, acked: bool, alert_id: str = "a1") -> Alert:
        return Alert.model_validate(
            {
                "alertId": alert_id,
                "machineId": "CNC-01",
                "ts": (t0 + timedelta(seconds=rev)).isoformat(),
                "severity": "CRITICAL",
                "kind": "machine_fault",
                "message": "m",
                "source": "engine",
                "active": active,
                "acknowledged": acked,
                "ackedBy": "alice" if acked else None,
                "ackedAt": (t0 + timedelta(seconds=rev)).isoformat() if acked else None,
                "rev": rev,
            }
        )

    store: Any = _FakeStore(
        [
            alert(0, active=True, acked=False),
            alert(1, active=True, acked=True),
            alert(2, active=False, acked=True),  # cleared + acknowledged = resolved
            alert(0, active=True, acked=False, alert_id="b"),
        ]
    )
    at1 = await alerts_as_of(store, t0 + timedelta(seconds=1))
    assert [(a.alert_id, a.rev) for a in at1] == [("b", 0), ("a1", 1)]  # by ts
    at2 = await alerts_as_of(store, t0 + timedelta(seconds=2))
    assert [a.alert_id for a in at2] == ["b"]


async def test_sources_ingest_block(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (rt, server), api_client(server) as c:
        body = (await c.get("/sources")).json()
    ingest = body["ingest"]
    assert ingest["published"] > 0
    assert ingest["messagesIn"] >= ingest["published"]
    assert ingest["quarantined"] == sum(ingest["quarantinedByReason"].values())
    assert ingest["latePct"] == pytest.approx(100 * ingest["late"] / ingest["published"])
    assert ingest["partialPct"] == pytest.approx(100 * ingest["partial"] / ingest["published"])
    lat = ingest["latency"]
    assert lat["count"] > 0
    assert lat["meanS"] >= 0
    assert lat.get("p95S") is None or lat["p95S"] > 0
    assert rt.metrics.value("published", source="file") == ingest["published"]


def test_scenario_options_match_the_simulator(tmp_path: Path) -> None:
    from twin_sim.config import GRADUAL_FAULTS, FaultType, load_sim_config

    path = REPO_ROOT / "config" / "simulator.yaml"
    opts = load_scenario_options(path)
    sim = load_sim_config(path, check_feasibility=False)
    assert {f.fault_type for f in opts.faults} == {f.value for f in FaultType}
    assert {f.fault_type for f in opts.faults if f.gradual} == {f.value for f in GRADUAL_FAULTS}
    for f in opts.faults:
        for kind in f.kinds:
            assert sim.faults.applicable(FaultType(f.fault_type), kind)
    assert opts.progression_s == tuple(sim.faults.progression_s)
    assert load_scenario_options(tmp_path / "missing.yaml").faults == ()


async def test_scenario_options_route(sim_export: SimExport, tmp_path: Path) -> None:
    async with running_api(sim_export, tmp_path) as (_rt, server), api_client(server) as c:
        body = (await c.get("/scenario/options")).json()
    kinds = {f["faultType"]: f["kinds"] for f in body["faults"]}
    assert kinds["spindle_imbalance"] == ["CNC"]
    assert body["progressionS"]["min"] < body["progressionS"]["max"]


def test_cors_origin_option_parses() -> None:
    args = _parser().parse_args(
        ["serve", "--cors-origin", "http://127.0.0.1:5999", "--cors-origin", "http://x:1"]
    )
    assert args.cors_origins == ["http://127.0.0.1:5999", "http://x:1"]
    assert _parser().parse_args(["serve"]).cors_origins is None

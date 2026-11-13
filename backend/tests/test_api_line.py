"""GET /line (Phase 5): the line config the 3D twin is laid out from, and the WS frame schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from twin_backend.api.app import create_app
from twin_backend.api.routes.line import line_response
from twin_backend.api.ws_models import CLIENT_ADAPTER, SERVER_ADAPTER, openapi_components
from twin_backend.runtime import build_runtime
from twin_backend.testing import REPO_ROOT
from twin_common.config import ConfigError, MachineLayout, TwinSettings, load_line_config

CONFIG = REPO_ROOT / "config"


def _get_line(config_dir: Path, line_file: str = "line.yaml") -> dict[str, Any]:
    settings = TwinSettings(_env_file=None, config_dir=config_dir, line_file=line_file)
    app = create_app(
        build_runtime(settings, root=REPO_ROOT, with_storage=False),
        host="127.0.0.1",
        manage_runtime=False,
    )
    resp = TestClient(app).get("/line")  # no lifespan: nothing starts, nothing connects
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


def test_full_line() -> None:
    body = _get_line(CONFIG)
    assert body["lineId"] == "line1"
    assert [m["id"] for m in body["machines"]] == ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]
    cnc = body["machines"][1]
    assert cnc["kind"] == "CNC"
    assert cnc["parts"] == ["spindle", "spindle_motor", "spindle_bearing"]
    assert cnc["sensorParts"]["vib"] == "spindle_bearing"
    assert cnc["layout"] is None
    assert body["buffers"][0] == {"id": "B1", "from": "CNV-01", "to": "CNC-01", "capacity": 10}
    assert body["topology"] == {
        "order": ["CNV-01", "CNC-01", "INSP-01", "CNV-02"],
        "upstream": {"CNV-01": None, "CNC-01": "CNV-01", "INSP-01": "CNC-01", "CNV-02": "INSP-01"},
        "downstream": {
            "CNV-01": "CNC-01",
            "CNC-01": "INSP-01",
            "INSP-01": "CNV-02",
            "CNV-02": None,
        },
    }
    assert body["sensors"]["temp"] == {"unit": "degC", "min": -20.0, "max": 150.0}


def test_single_station_line() -> None:
    body = _get_line(CONFIG, "line_single.yaml")
    assert body["lineId"] == "line1_single"
    assert [m["id"] for m in body["machines"]] == ["CNV-01"]
    assert body["buffers"] == []
    assert body["topology"]["upstream"] == {"CNV-01": None}
    assert body["topology"]["downstream"] == {"CNV-01": None}


def test_added_machine_and_layout_override_appear(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for src in CONFIG.glob("*.yaml"):
        (config_dir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    line = yaml.safe_load((CONFIG / "line.yaml").read_text(encoding="utf-8"))
    extra = dict(line["machines"][3], id="CNV-03", name="Extra conveyor")
    extra["layout"] = {"position": [1.0, 0.0, 6.5], "rotation_deg": 90}
    line["machines"].append(extra)
    line["buffers"].append({"id": "B4", "from": "CNV-02", "to": "CNV-03", "capacity": 5})
    (config_dir / "line.yaml").write_text(yaml.safe_dump(line), encoding="utf-8")

    body = _get_line(config_dir)
    added = body["machines"][-1]
    assert added["id"] == "CNV-03"
    assert added["layout"] == {"position": [1.0, 0.0, 6.5], "rotationDeg": 90.0}
    assert body["topology"]["downstream"]["CNV-02"] == "CNV-03"
    assert body["topology"]["upstream"]["CNV-03"] == "CNV-02"


def test_machine_layout_validation() -> None:
    assert MachineLayout(rotation_deg=45).position is None
    assert MachineLayout(position=(1, 2, 3)).rotation_deg is None
    with pytest.raises(ValidationError, match="finite"):
        MachineLayout(position=(float("nan"), 0, 0))
    with pytest.raises(ValidationError, match="finite"):
        MachineLayout(rotation_deg=float("inf"))
    with pytest.raises(ValidationError):
        MachineLayout.model_validate({"position": [1, 2]})
    with pytest.raises(ValidationError):
        MachineLayout.model_validate({"scale": 2})


def test_bad_layout_in_yaml_is_a_config_error(tmp_path: Path) -> None:
    line = yaml.safe_load((CONFIG / "line.yaml").read_text(encoding="utf-8"))
    line["machines"][0]["layout"] = {"position": [0, 0]}
    path = tmp_path / "line.yaml"
    path.write_text(yaml.safe_dump(line), encoding="utf-8")
    with pytest.raises(ConfigError, match="layout"):
        load_line_config(path)


def test_line_response_matches_engine_topology_for_shipped_configs() -> None:
    for name in ("line.yaml", "line_single.yaml"):
        cfg = load_line_config(CONFIG / name)
        resp = line_response(cfg)
        assert resp.topology.order == [m.id for m in cfg.machines]
        assert {m.id: m.parts for m in resp.machines} == {m.id: m.parts for m in cfg.machines}


# ---------------------------------------------------------------- WS frame models
def test_ws_client_messages_parse() -> None:
    assert CLIENT_ADAPTER.validate_python({"type": "auth", "apiKey": "k"}).api_key == "k"
    assert CLIENT_ADAPTER.validate_python({"type": "auth"}).api_key is None
    sub = CLIENT_ADAPTER.validate_python({"type": "subscribe", "machines": None})
    assert sub.channels == [
        "readings",
        "states",
        "line",
        "dataQuality",
        "alerts",
        "commands",
        "predictions",
    ]
    with pytest.raises(ValidationError):
        CLIENT_ADAPTER.validate_python({"type": "subscribe", "channels": ["bogus"]})
    with pytest.raises(ValidationError):
        CLIENT_ADAPTER.validate_python({"type": "hello"})


def test_ws_server_messages_validate_and_reject() -> None:
    SERVER_ADAPTER.validate_python({"type": "pong", "data": {}})
    SERVER_ADAPTER.validate_python({"type": "dropped", "data": {"count": 3}})
    SERVER_ADAPTER.validate_python(
        {"type": "auth", "data": {"principal": "anonymous", "role": "viewer"}}
    )
    with pytest.raises(ValidationError):
        SERVER_ADAPTER.validate_python({"type": "dropped", "data": {"count": "x"}})
    with pytest.raises(ValidationError):
        SERVER_ADAPTER.validate_python({"type": "pong", "data": {"extra": 1}})


def test_openapi_has_ws_unions_with_every_frame_type() -> None:
    comps = openapi_components()
    mapping = comps["WsServerMessage"]["discriminator"]["mapping"]
    assert set(mapping) == {
        "auth",
        "subscribed",
        "snapshot",
        "reading",
        "state",
        "line",
        "dataQuality",
        "alert",
        "command",
        "ack",
        "prediction",
        "dropped",
        "error",
        "pong",
    }
    assert set(comps["WsClientMessage"]["discriminator"]["mapping"]) == {
        "auth",
        "subscribe",
        "ping",
    }
    spec = json.loads((REPO_ROOT / "frontend" / "src" / "api" / "openapi.json").read_text("utf-8"))
    assert "WsServerMessage" in spec["components"]["schemas"]
    assert "/line" in spec["paths"]

"""Drift in the live worker (ADR 0031): the bundle carries a training reference, the worker
reports drift for operating rows only, a shifted load is flagged, a clean series is not, and
a bundle without a reference (built before Phase 11 Part C) simply has no drift monitor."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from twin_common.config import load_line_config
from twin_ml.serve.drift import DriftSettings
from twin_ml.testing import CONFIG_DIR, configure_message, series_rows

LOAD_COL = 6  # a series row is [seq, ts, state, speed, temp, vib, load, current]
SETTINGS = {
    "margin": 1.5,
    "clearRatio": 0.6,
    "sustainS": 10,
    "clearS": 20,
    "adwinDelta": 0.002,
}


@pytest.fixture(scope="module")
def worker(serve_setup: Any) -> Any:
    from twin_ml.serve.worker import load_worker

    tracking, _, _, _ = serve_setup
    return load_worker(
        ["monitor"],
        None,
        ("test-autoencoder", "latest"),
        tracking,
        tracking.parent / "cache_drift",
        8,
    )


def configure(worker: Any, machine: str, kind: str, parts: dict[str, str], drift: Any) -> None:
    msg = configure_message(machine, kind, parts)
    msg["drift"] = drift
    reply = worker.configure(msg)
    assert reply["machines"] == [machine]


def longest_series(split: Any) -> int:
    ids, counts = np.unique(split.arrays["series"], return_counts=True)
    return int(ids[int(np.argmax(counts))])


def machine_of(split: Any, sid: int) -> tuple[str, str, dict[str, str]]:
    line = load_line_config(CONFIG_DIR / "line.yaml")
    machine = str(split.arrays["series_machine"][sid])
    m = next(x for x in line.machines if x.id == machine)
    return machine, m.kind.value, dict(m.sensor_parts)


def feed(worker: Any, machine: str, rows: list[list[Any]]) -> list[dict[str, Any]]:
    out = []
    for i in range(0, len(rows), 50):
        out += worker.step({machine: rows[i : i + 50]})["rows"][machine]
    return out


def test_the_bundle_carries_a_training_reference(worker: Any) -> None:
    from twin_ml.serve.drift import drift_public

    ref = worker.anomaly.drift_reference
    assert ref is not None
    assert ref["channels"] == ["speed", "load"]
    assert set(ref["kinds"]) >= {"CNC"}
    assert drift_public(ref)["available"] is True
    for per_channel in ref["kinds"].values():
        assert set(per_channel) == {"speed", "load"}  # operating-condition inputs only


def test_clean_rows_are_not_drift_and_a_load_shift_is(serve_setup: Any, worker: Any) -> None:
    _, _, ds, _ = serve_setup
    split = ds.split("train")  # the reference's own data: as clean as it gets
    sid = longest_series(split)
    machine, kind, parts = machine_of(split, sid)
    rows = series_rows(split, sid)
    assert len(rows) > 200
    configure(worker, machine, kind, parts, SETTINGS)
    clean = feed(worker, machine, rows)
    drift_entries = [e["drift"] for e in clean if "drift" in e]
    assert drift_entries, "operating rows carry a drift report"
    # non-operating rows carry none
    state_of = {int(r[0]): r[2] for r in rows}
    assert all(("drift" in e) == (state_of[int(e["seq"])] in ("RUNNING", "WARNING")) for e in clean)
    assert not any(d["drifted"] for d in drift_entries)

    # The quick test bundles are tiny, so their calibrated alert level is not meaningful (it can
    # exceed what any window reaches). The decision logic is covered by test_ml_drift.py; here a
    # low margin shows that the shifted rows reach the monitor and are decided on.
    configure(worker, machine, kind, parts, SETTINGS | {"margin": 0.01})  # fresh monitors
    shifted_rows = [[*r[:LOAD_COL], (r[LOAD_COL] or 0.0) + 200.0, *r[LOAD_COL + 1 :]] for r in rows]
    shifted = feed(worker, machine, shifted_rows)
    reports = [e["drift"] for e in shifted if "drift" in e]
    flagged = [d for d in reports if d["drifted"]]
    clean_psi = max(d["psi"]["load"] for d in drift_entries if d["psi"]["load"] is not None)
    assert max(d["psi"]["load"] for d in reports if d["psi"]["load"] is not None) > clean_psi
    assert flagged, "a load far outside the training range is drift"
    assert flagged[0]["psi"]["load"] >= flagged[0]["alert"]["load"]
    assert flagged[0]["since"] is not None


def test_without_drift_settings_or_reference_there_is_no_report(
    serve_setup: Any, worker: Any
) -> None:
    _, _, ds, _ = serve_setup
    split = ds.split("train")
    sid = longest_series(split)
    machine, kind, parts = machine_of(split, sid)
    configure(worker, machine, kind, parts, None)  # drift off in the settings
    assert not any("drift" in e for e in feed(worker, machine, series_rows(split, sid)[:150]))
    saved = worker.anomaly.drift_reference
    try:
        worker.anomaly.drift_reference = None  # a bundle built before Part C
        msg = configure_message(machine, kind, parts)
        msg["drift"] = SETTINGS
        assert worker.configure(msg)["drift"] == []
        assert not any("drift" in e for e in feed(worker, machine, series_rows(split, sid)[:150]))
    finally:
        worker.anomaly.drift_reference = saved


def test_settings_message_round_trip() -> None:
    s = DriftSettings.of(copy.deepcopy(SETTINGS), window_s=60)
    assert (s.sustain_rows, s.clear_rows, s.window_s, s.margin) == (10, 20, 60, 1.5)

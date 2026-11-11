"""Accumulates StepFrames into a labelled Arrow table (one row per machine per sample)."""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pyarrow as pa

from twin_common.enums import MachineState, Quality
from twin_sim.labels import time_to_fault
from twin_sim.line import LineSimulator, StepFrame

SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("machine_id", pa.string()),
        ("seq", pa.int64()),
        ("ts", pa.timestamp("ms", tz="UTC")),
        ("t_s", pa.float64()),
        ("state", pa.string()),
        ("speed", pa.float64()),
        ("temp", pa.float64()),
        ("vib", pa.float64()),
        ("load", pa.float64()),
        ("current", pa.float64()),
        ("run_time", pa.float64()),
        ("quality", pa.string()),
        ("fault_type", pa.string()),
        ("time_to_fault_s", pa.float64()),
        ("ttf_censored", pa.bool_()),
        ("root_cause_machine", pa.string()),
        ("load_range", pa.string()),
        ("sensor_fault", pa.string()),
        ("buffer_in", pa.int32()),
        ("buffer_out", pa.int32()),
    ]
)

_SENSORS = ("speed", "temp", "vib", "load", "current")
_PER_SAMPLE = (
    "seq",
    "state",
    *_SENSORS,
    "run_time",
    "fault_type",
    "root_cause",
    "sensor_fault",
    "buffer_in",
    "buffer_out",
)


class Recorder:
    def __init__(self, sim: LineSimulator) -> None:
        self.sim = sim
        self._t: list[float] = []
        self._cols: list[dict[str, list[object]]] = [{k: [] for k in _PER_SAMPLE} for _ in sim.ids]

    def add(self, frame: StepFrame) -> None:
        self._t.append(frame.t_s)
        for cols, s in zip(self._cols, frame.samples, strict=True):
            for k in _PER_SAMPLE:
                cols[k].append(getattr(s, k))

    def __len__(self) -> int:
        return len(self._t)

    def to_table(self) -> pa.Table:
        sim, plan = self.sim, self.sim.plan
        t = np.asarray(self._t, dtype=np.float64)
        n = t.shape[0]
        start_ms = int(plan.start_ts.astimezone(UTC).timestamp() * 1000)
        ts_ms = start_ms + np.round(t * 1000.0).astype(np.int64)
        cap = sim.cfg.labels.ttf_cap_s
        tables: list[pa.Table] = []
        for mid, cols in zip(sim.ids, self._cols, strict=True):
            is_fault = np.asarray([s is MachineState.FAULT for s in cols["state"]], dtype=bool)
            ttf, censored = time_to_fault(t, is_fault, cap)
            data: dict[str, object] = {
                "run_id": pa.array([plan.run_id] * n, pa.string()),
                "machine_id": pa.array([mid] * n, pa.string()),
                "seq": pa.array(cols["seq"], pa.int64()),
                "ts": pa.array(ts_ms, pa.timestamp("ms", tz="UTC")),
                "t_s": pa.array(t),
                "state": pa.array([str(s) for s in cols["state"]], pa.string()),
                **{k: pa.array(cols[k], pa.float64()) for k in _SENSORS},
                "run_time": pa.array(cols["run_time"], pa.float64()),
                "quality": pa.array([Quality.OK.value] * n, pa.string()),
                "fault_type": pa.array(cols["fault_type"], pa.string()),
                "time_to_fault_s": pa.array(ttf),
                "ttf_censored": pa.array(censored),
                "root_cause_machine": pa.array(cols["root_cause"], pa.string()),
                "load_range": pa.array([plan.load_range] * n, pa.string()),
                "sensor_fault": pa.array(cols["sensor_fault"], pa.string()),
                "buffer_in": pa.array(
                    [None if b < 0 else b for b in cols["buffer_in"]],  # type: ignore[operator]
                    pa.int32(),
                ),
                "buffer_out": pa.array(
                    [None if b < 0 else b for b in cols["buffer_out"]],  # type: ignore[operator]
                    pa.int32(),
                ),
            }
            tables.append(pa.table(data, schema=SCHEMA))
        return pa.concat_tables(tables)

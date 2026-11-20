"""Scoring a projected trajectory with the deployed model (ADR 0031).

A separate worker pool (same bundles, same gate, no attributions) is started on the first
request and reused, so a what-if never touches the buffers, alarms or alerts of the live
inference. Each variant is replayed from empty buffers over ``history + projection`` in
order, with the evaluated alarm rule (``MachineAlarms``), exactly as live. One request at a
time (a lock): the pool is single-purpose.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twin_backend.inference.alarms import MachineAlarms, sustain_windows
from twin_backend.inference.workers import WorkerPool
from twin_common.config.inference import InferenceConfig
from twin_common.config.line import LineConfig
from twin_common.contracts import Reading
from twin_common.enums import HealthClass
from twin_common.logging import get_logger

log = get_logger(__name__)

CLASSES = (HealthClass.NORMAL, HealthClass.WARNING, HealthClass.FAULT)


@dataclass(frozen=True, slots=True)
class Scored:
    """The model's view of one reading."""

    gate: str
    p_warning: float | None = None
    p_fault: float | None = None
    top_class: HealthClass | None = None
    alarm: bool | None = None
    unusual: bool | None = None
    ttf_s: float | None = None


def row(r: Reading) -> list[Any]:
    return [r.seq, r.ts.timestamp(), r.state.value, r.speed, r.temp, r.vib, r.load, r.current]


class ScoringError(RuntimeError):
    """The what-if workers cannot start or answer."""


class WhatIfScorer:
    def __init__(
        self, cfg: InferenceConfig, line: LineConfig, root: Path, rows_per_step: int
    ) -> None:
        self.cfg = cfg
        self.line = line
        self.rows_per_step = rows_per_step
        self.pool = WorkerPool(cfg, line, root)
        self._lock = asyncio.Lock()
        self._ready = False
        self.model_version: str | None = None
        self._k = 1

    async def _ensure(self) -> None:
        if self._ready and self.pool.alive:
            return
        try:
            info = await self.pool.start()
        except Exception as exc:
            raise ScoringError(f"{type(exc).__name__}: {exc}") from exc
        sup = info.bundles.get("supervised", {})
        sustain = float(sup.get("sustainS", 0))
        horizons = [int(h) for h in sup.get("horizons", [])]
        if sustain <= 0 or self.cfg.primary_horizon_s not in horizons:
            await self.pool.close()
            raise ScoringError("the bundle has no sustain time or lacks the primary horizon")
        self._k = sustain_windows(sustain)
        self.model_version = info.model_version
        self._ready = True

    async def score(
        self, history: list[Reading], variants: dict[str, list[Reading]]
    ) -> dict[str, dict[tuple[str, int], Scored]]:
        """For each variant: ``(machine, seq) -> Scored`` for ``history + variant`` readings."""
        async with self._lock:
            await self._ensure()
            out: dict[str, dict[tuple[str, int], Scored]] = {}
            try:
                for name, future in variants.items():
                    await self.pool.reconfigure(explain=False)  # empty buffers
                    out[name] = await self._replay(history + future)
            except Exception as exc:  # worker lost mid-run: start again next time
                self._ready = False
                await self.pool.close()
                raise ScoringError(f"{type(exc).__name__}: {exc}") from exc
            return out

    async def _replay(self, readings: list[Reading]) -> dict[tuple[str, int], Scored]:
        by_machine: dict[str, list[Reading]] = {}
        for r in readings:
            by_machine.setdefault(r.machine_id, []).append(r)
        alarms = {m: MachineAlarms(self._k) for m in by_machine}
        primary = str(self.cfg.primary_horizon_s)
        scored: dict[tuple[str, int], Scored] = {}
        n = max((len(v) for v in by_machine.values()), default=0)
        for i in range(0, n, self.rows_per_step):
            rows = {
                m: [row(r) for r in rs[i : i + self.rows_per_step]]
                for m, rs in by_machine.items()
                if rs[i : i + self.rows_per_step]
            }
            res = await self.pool.step(rows)
            for machine, entries in res["rows"].items():
                for e in entries:
                    out = e.get("out")
                    seq = int(e["seq"])
                    if e["status"] != "predict" or not out:
                        alarms[machine].reset()
                        scored[(machine, seq)] = Scored(gate=str(e["status"]))
                        continue
                    state = alarms[machine].update(out["probs"][primary], out.get("anomalyFlag"))
                    cal = [float(x) for x in out["cal"][primary]]
                    total = sum(cal)
                    p = [v / total for v in cal]
                    scored[(machine, seq)] = Scored(
                        gate="predict",
                        p_warning=p[1],
                        p_fault=p[2],
                        top_class=CLASSES[max(range(3), key=lambda j: p[j])],
                        alarm=state.alarm,
                        unusual=state.unusual,
                        ttf_s=out.get("ttf"),
                    )
        return scored

    async def close(self) -> None:
        await self.pool.close()
        self._ready = False

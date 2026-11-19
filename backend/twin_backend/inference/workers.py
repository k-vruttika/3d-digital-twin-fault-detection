"""The inference worker processes of the backend (ADR 0030).

``InferenceConfig.worker_plans`` decides the processes (one torch process for a sequence
model + autoencoder; XGBoost and torch apart). The pool starts them, says hello (bundle info,
loaded libraries), configures their buffers and gate, sends every batch of readings to all of
them, and merges their answers per reading. The backend process itself never imports an ML
library: the workers are ``python -m twin_ml.serve worker`` child processes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twin_common.config.inference import InferenceConfig, WorkerPlan
from twin_common.config.line import LineConfig
from twin_common.jsonl_process import JsonLinesProcess, WorkerError
from twin_common.logging import get_logger

log = get_logger(__name__)

ML_LIBRARY_PAIR = ("torch", "xgboost")  # never in one process (ADR 0020)


class WorkerMixError(RuntimeError):
    """A worker reports torch and XGBoost loaded together."""


@dataclass
class PoolInfo:
    """What the workers said in their hello (bundles, labels, loaded libraries)."""

    bundles: dict[str, dict[str, Any]] = field(default_factory=dict)  # role -> public info
    processes: dict[str, dict[str, Any]] = field(default_factory=dict)  # worker -> libraries
    window: int = 0

    @property
    def model_version(self) -> str:
        labels = [self.bundles[r]["label"] for r in ("supervised", "anomaly") if r in self.bundles]
        return " + ".join(labels)


class WorkerPool:
    def __init__(
        self,
        cfg: InferenceConfig,
        line: LineConfig,
        root: Path,
        python: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.line = line
        self.root = root
        self.plans: list[WorkerPlan] = cfg.worker_plans()
        self.procs = [
            JsonLinesProcess(p.command(cfg, root, python), name=f"inference-{p.name}", cwd=root)
            for p in self.plans
        ]
        self.info = PoolInfo()

    def configure_message(self) -> dict[str, Any]:
        return {
            "op": "configure",
            "machines": {
                m.id: {"kind": m.kind.value, "sensorParts": dict(m.sensor_parts)}
                for m in self.line.machines
            },
            "gate": {
                "operating": [s.value for s in self.cfg.gate.operating],
                "waiting": [s.value for s in self.cfg.gate.waiting],
                "maxGapRows": self.cfg.gate.max_gap_rows,
            },
            "explainAsync": self.cfg.explain_async,
            "capacity": self.cfg.buffer_capacity_rows,
        }

    def _start_sync(self) -> PoolInfo:
        info = PoolInfo()
        for proc in self.procs:
            proc.start()
        windows = set()
        for plan, proc in zip(self.plans, self.procs, strict=True):
            hello = proc.request({"op": "hello"}, self.cfg.worker.start_timeout_s)
            libs = set(hello["process"]["libraries"])
            if set(ML_LIBRARY_PAIR) <= libs:
                raise WorkerMixError(f"worker {plan.name} loaded both torch and xgboost")
            info.processes[plan.name] = {**hello["process"], "pid": proc.pid}
            info.bundles |= hello["bundles"]
            windows.add(int(hello["window"]))
        if len(windows) != 1:
            raise WorkerError(f"workers disagree on the window length: {sorted(windows)}")
        info.window = windows.pop()
        conf = self.configure_message()
        for proc in self.procs:
            proc.request(conf, self.cfg.worker.request_timeout_s)
        return info

    async def start(self) -> PoolInfo:
        """Start (or restart) every worker; raises when one fails to load or configure."""
        try:
            self.info = await asyncio.to_thread(self._start_sync)
        except BaseException:
            await self.close()
            raise
        log.info(
            "inference.workers_ready",
            model_version=self.info.model_version,
            workers={n: p["libraries"] for n, p in self.info.processes.items()},
        )
        return self.info

    async def step(self, rows: dict[str, list[list[Any]]]) -> dict[str, Any]:
        """Send one batch to every worker; per machine, one merged entry per new reading."""
        msg = {"op": "step", "rows": rows}
        timeout = self.cfg.worker.request_timeout_s
        replies = await asyncio.gather(*(p.arequest(msg, timeout) for p in self.procs))
        merged: dict[str, list[dict[str, Any]]] = {}
        timings: dict[str, float] = {}
        for reply in replies:
            for stage, ms in reply["timings_ms"].items():
                # workers run in parallel: the batch takes the slowest total; stages add up
                if stage == "total":
                    timings[stage] = max(timings.get(stage, 0.0), ms)
                else:
                    timings[stage] = timings.get(stage, 0.0) + ms
            for machine, entries in reply["rows"].items():
                have = merged.setdefault(machine, [])
                if not have:
                    have.extend(dict(e) for e in entries)
                    continue
                by_seq = {e["seq"]: e for e in have}
                for e in entries:
                    mine = by_seq.get(e["seq"])
                    if mine is None:
                        have.append(dict(e))
                        continue
                    if mine["status"] != e["status"]:  # same readings, same rule: never differs
                        raise WorkerError(
                            f"{machine} seq {e['seq']}: workers disagree on the gate "
                            f"({mine['status']} vs {e['status']})"
                        )
                    if "out" in e:
                        mine["out"] = {**mine.get("out", {}), **e["out"]}
        windows = max((r["windows"] for r in replies), default=0)
        explanations = [e for r in replies for e in r.get("explanations", [])]
        return {
            "rows": merged,
            "timings_ms": timings,
            "windows": windows,
            "explanations": explanations,
        }

    @property
    def alive(self) -> bool:
        return all(p.alive for p in self.procs)

    def exit_codes(self) -> dict[str, int | None]:
        return {p.name: p.returncode for p in self.procs}

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        for p in self.procs:
            p.close()

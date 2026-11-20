"""Live inference service (Phase 11, ADR 0030).

Subscribes to the readings on the event bus, sends them in batches to the inference workers
(``workers.WorkerPool``), and turns every predicted window into ``Prediction`` events, one per
horizon, published on ``Topic.PREDICTIONS``: the storage writer stores them (so replay shows
predictions as they were) and the WebSocket hub sends them.

* **Gate.** The workers report a gate status for every reading: ``predict`` (all window rows
  RUNNING / WARNING), ``warming_up``, ``waiting`` (STARVED / BLOCKED) or ``stopped``. A machine
  whose twin state is STALE / OFFLINE, or which sent nothing for ``prediction_stale_s``, counts
  as stopped. No prediction is made while the gate is closed, and every alarm clears.
* **Alarms.** ``alarms.MachineAlarms`` applies the evaluated rule to the *uncalibrated*
  probabilities; ``Prediction.probs`` holds the calibrated ones (shown as confidence only).
* **Alerts.** ``predicted_warning`` / ``predicted_fault`` / ``unusual_behaviour`` conditions go
  to ``TwinEngine.update_conditions`` (ISA-18.2 lifecycle, deduplicated, cleared when the
  condition ends).
* **Attributions** (supervised ``explain()`` only, never from the autoencoder) come from the
  workers beside the prediction path: the newest explanation of a machine is attached to its
  primary-horizon predictions while the uncalibrated argmax stays WARNING / FAULT, with the
  window it explains (``attributionsTs``); it is dropped when the argmax is NORMAL.
* **Reliability** is DEGRADED while the machine's data quality is DEGRADED, or while the
  monitor worker reports drift of its operating-condition inputs (PSI against the training
  distribution, ADWIN dating the change; ADR 0031). Drift also raises ``model_drift``.

The service never imports an ML library; minimal mode does not build it at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from twin_backend.bus.base import EventBus, Subscription, Topic
from twin_backend.clock import Clock
from twin_backend.engine.twin import AlertCondition, TwinEngine
from twin_backend.inference.alarms import OFF, AlarmState, MachineAlarms, sustain_windows
from twin_backend.inference.workers import PoolInfo, WorkerPool
from twin_common.config.engine import AlertKind
from twin_common.config.inference import InferenceConfig
from twin_common.config.line import LineConfig
from twin_common.contracts import Attribution, Prediction, Reading
from twin_common.enums import DataQuality, HealthClass, MachineState, Reliability
from twin_common.jsonl_process import WorkerError
from twin_common.logging import get_logger

log = get_logger(__name__)

PREDICT = "predict"
STOPPED = "stopped"
UNAVAILABLE = "unavailable"
LINK_DOWN = (MachineState.STALE, MachineState.OFFLINE)
CLASSES = (HealthClass.NORMAL, HealthClass.WARNING, HealthClass.FAULT)
ALERT_KINDS = (
    AlertKind.PREDICTED_WARNING,
    AlertKind.PREDICTED_FAULT,
    AlertKind.UNUSUAL_BEHAVIOUR,
    AlertKind.MODEL_DRIFT,
)


class InferenceMetrics:
    def __init__(self, registry: CollectorRegistry | None) -> None:
        kw: dict[str, Any] = {"registry": registry}
        self.step_seconds = Histogram(
            "linetwin_inference_step_seconds",
            "Round trip of one batch to the inference workers (IPC included)",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            **kw,
        )
        self.stage_seconds = Histogram(
            "linetwin_inference_stage_seconds",
            "Worker time per stage and batch",
            ["stage"],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
            **kw,
        )
        self.predictions = Counter(
            "linetwin_inference_predictions", "Prediction events published", ["machine"], **kw
        )
        self.dropped = Counter(
            "linetwin_inference_readings_dropped",
            "Readings not sent to a worker (workers not ready)",
            **kw,
        )
        self.restarts = Counter("linetwin_inference_worker_restarts", "Worker restarts", **kw)
        self.explain_age = Histogram(
            "linetwin_inference_explain_age_seconds",
            "From an explanation request to its result (beside the prediction path)",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            **kw,
        )
        self.ready = Gauge("linetwin_inference_ready", "1 while the workers are ready", **kw)


@dataclass
class MachineGate:
    status: str = STOPPED
    run_len: int = 0
    last_reading_wall: float | None = None
    last_prediction_ts: datetime | None = None
    alarms: AlarmState = OFF
    sent: dict[AlertKind, bool] = field(default_factory=dict)  # last condition sent per kind
    explanation: tuple[datetime, list[Attribution]] | None = None  # newest, while alarmed
    drifted: bool = False  # operating-condition drift (monitor worker; ADR 0031)
    drift: dict[str, Any] | None = None  # the worker's last drift report


@dataclass
class InferenceStatus:
    state: str  # starting | ready | restarting | failed | stopped
    reason: str | None = None
    since: datetime | None = None


class InferenceService:
    def __init__(
        self,
        cfg: InferenceConfig,
        line: LineConfig,
        bus: EventBus,
        engine: TwinEngine,
        clock: Clock,
        root: Path,
        registry: CollectorRegistry | None = None,
        pool: WorkerPool | None = None,
    ) -> None:
        self.cfg = cfg
        self.line = line
        self.bus = bus
        self.engine = engine
        self.clock = clock
        self.root = root
        self.metrics = InferenceMetrics(registry)
        self.pool = pool if pool is not None else WorkerPool(cfg, line, root)
        self.machine_ids = [m.id for m in line.machines]
        self.gates = {m: MachineGate() for m in self.machine_ids}
        self.alarms: dict[str, MachineAlarms] = {}
        self.status = InferenceStatus("stopped")
        self.info = PoolInfo()
        self.sustain_s: float | None = None
        self.horizons: list[int] = []
        self._sub: Subscription | None = None
        self._task: asyncio.Task[None] | None = None
        self._ts: dict[str, dict[int, datetime]] = {m: {} for m in self.machine_ids}
        self._restart_attempt = 0
        self._settled = asyncio.Event()  # set when the first start attempt ended

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._task is not None:
            return
        self._sub = self.bus.subscribe(Topic.READINGS, maxsize=self.cfg.reading_queue_size)
        self._task = asyncio.create_task(self._run(), name="inference")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._sub is not None:
            self._sub.close()
            self._sub = None
        await self.pool.close()
        self.metrics.ready.set(0)
        self._set_status("stopped")

    def _set_status(self, state: str, reason: str | None = None) -> None:
        self.status = InferenceStatus(state, reason, self.clock.now())

    @property
    def ready(self) -> bool:
        return self.status.state == "ready"

    async def wait_ready(self, timeout_s: float) -> bool:
        """Wait for the first worker start to finish; True when the workers are ready."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._settled.wait(), timeout_s)
        return self.ready

    # ------------------------------------------------------------------ worker lifecycle
    async def _start_pool(self, state: str) -> bool:
        self._set_status(state)
        try:
            self.info = await self.pool.start()
        except Exception as exc:
            self._set_status("failed", f"{type(exc).__name__}: {exc}")
            log.error("inference.start_failed", error=self.status.reason)
            self.metrics.ready.set(0)
            return False
        sup = self.info.bundles.get("supervised", {})
        self.sustain_s = float(sup.get("sustainS", 0)) or None
        self.horizons = [int(h) for h in sup.get("horizons", [])]
        if self.sustain_s is None or self.cfg.primary_horizon_s not in self.horizons:
            self._set_status(
                "failed",
                f"bundle horizons {self.horizons} lack the primary horizon "
                f"{self.cfg.primary_horizon_s} s, or it has no sustain time",
            )
            await self.pool.close()
            return False
        k = sustain_windows(self.sustain_s)
        self.alarms = {m: MachineAlarms(k) for m in self.machine_ids}
        self._restart_attempt = 0
        self._set_status("ready")
        self.metrics.ready.set(1)
        return True

    async def _backoff(self) -> None:
        delays = self.cfg.worker.restart_backoff_s
        delay = delays[min(self._restart_attempt, len(delays) - 1)]
        self._restart_attempt += 1
        await self.clock.sleep(delay)

    # ------------------------------------------------------------------ main loop
    async def _run(self) -> None:
        ok = await self._start_pool("starting")
        self._settled.set()
        if not ok:
            await self._close_all_gates()
            while not await self._retry():
                pass
        while True:
            await self.clock.sleep(self.cfg.batch_interval_s)
            rows = self._collect()
            if not self.ready:
                continue
            if not self.pool.alive:
                await self._worker_lost(f"worker exited: {self.pool.exit_codes()}")
                continue
            if rows:
                try:
                    t0 = time.perf_counter()
                    res = await self.pool.step(rows)
                    self.metrics.step_seconds.observe(time.perf_counter() - t0)
                except WorkerError as exc:
                    await self._worker_lost(str(exc))
                    continue
                for stage, ms in res["timings_ms"].items():
                    self.metrics.stage_seconds.labels(stage=stage).observe(ms / 1000.0)
                self._explanations(res.get("explanations", []))
                await self._apply(res["rows"])
            await self._check_stopped()

    async def _retry(self) -> bool:
        await self._backoff()
        self.metrics.restarts.inc()
        return await self._start_pool("restarting")

    async def _worker_lost(self, reason: str) -> None:
        log.error("inference.worker_lost", reason=reason)
        self.metrics.ready.set(0)
        self._set_status("restarting", reason)
        await self.pool.close()
        await self._close_all_gates()
        while not await self._retry():
            pass

    def _collect(self) -> dict[str, list[list[Any]]]:
        rows: dict[str, list[list[Any]]] = {}
        if self._sub is None:
            return rows
        now = time.monotonic()
        while (msg := self._sub.get_nowait()) is not None:
            if not isinstance(msg, Reading) or msg.machine_id not in self.gates:
                continue
            self.gates[msg.machine_id].last_reading_wall = now
            if not self.ready:
                self.metrics.dropped.inc()
                continue
            self._ts[msg.machine_id][msg.seq] = msg.ts
            rows.setdefault(msg.machine_id, []).append(
                [
                    msg.seq,
                    msg.ts.timestamp(),
                    msg.state.value,
                    msg.speed,
                    msg.temp,
                    msg.vib,
                    msg.load,
                    msg.current,
                ]
            )
        return rows

    # ------------------------------------------------------------------ results
    def _explanations(self, items: list[dict[str, Any]]) -> None:
        for e in items:
            gate = self.gates.get(str(e["machine"]))
            if gate is None or e.get("error") or not e["attributions"]:
                if e.get("error"):
                    log.warning("inference.explain_failed", error=e["error"])
                continue
            ts = datetime.fromtimestamp(float(e["ts"]), UTC)
            if gate.explanation is None or ts >= gate.explanation[0]:
                gate.explanation = (
                    ts,
                    [Attribution.model_validate(a) for a in e["attributions"]],
                )
            self.metrics.explain_age.observe(float(e["ageMs"]) / 1000.0)

    async def _apply(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        conditions: list[AlertCondition] = []
        predictions: list[Prediction] = []
        for machine, entries in rows.items():
            gate = self.gates[machine]
            seen = self._ts[machine]
            for e in entries:
                ts = seen.pop(int(e["seq"]), None)
                if ts is None:
                    continue
                gate.status, gate.run_len = str(e["status"]), int(e["runLen"])
                if "drift" in e:
                    conditions += self._drift(machine, gate, e["drift"], ts)
                out = e.get("out")
                if gate.status != PREDICT or not out:
                    gate.explanation = None
                    self.alarms[machine].reset()
                    gate.alarms = OFF
                    conditions += self._conditions(machine, OFF, ts, None)
                    continue
                primary = out["probs"][str(self.cfg.primary_horizon_s)]
                state = self.alarms[machine].update(primary, out.get("anomalyFlag"))
                if not out.get("alarmClass"):
                    gate.explanation = None  # no highlight while the model predicts NORMAL
                gate.alarms = state
                gate.last_prediction_ts = ts
                preds = self._predictions(machine, ts, out, state)
                predictions += preds
                conditions += self._conditions(machine, state, ts, preds)
            # readings the workers did not report (ignored as duplicates) are forgotten
            if len(seen) > self.cfg.buffer_capacity_rows:
                seen.clear()
        for p in predictions:
            await self.bus.publish(Topic.PREDICTIONS, p)
            self.metrics.predictions.labels(machine=p.machine_id).inc()
        if conditions:
            self.engine.update_conditions(conditions)

    def _drift(
        self, machine: str, gate: MachineGate, report: dict[str, Any], ts: datetime
    ) -> list[AlertCondition]:
        """Take the monitor's drift report of one operating row; the condition if it changed."""
        gate.drift = report
        drifted = bool(report["drifted"])
        if drifted == gate.drifted:
            return []
        gate.drifted = drifted
        return self._conditions(machine, gate.alarms, ts, None)

    def _reliability(self, machine: str) -> Reliability:
        dq = self.engine.data_quality(machine)
        drifted = self.gates[machine].drifted
        return (
            Reliability.DEGRADED if dq.status is DataQuality.DEGRADED or drifted else Reliability.OK
        )

    def _predictions(
        self, machine: str, ts: datetime, out: dict[str, Any], state: AlarmState
    ) -> list[Prediction]:
        reliability = self._reliability(machine)
        version = self.info.model_version
        primary = self.cfg.primary_horizon_s
        expl = self.gates[machine].explanation
        if expl is not None and ts - expl[0] > self.stale_after():
            expl = None  # too old to describe this window
        result = []
        for h in self.horizons:
            cal = [float(x) for x in out["cal"][str(h)]]
            total = sum(cal)
            probs = {c: v / total for c, v in zip(CLASSES, cal, strict=True)}
            iv = out.get("ttfInterval")
            result.append(
                Prediction(
                    machine_id=machine,
                    ts=ts,
                    horizon_s=h,
                    probs=probs,
                    calibrated=True,
                    conformal_set=[HealthClass(c) for c in out["set"][str(h)]],
                    ttf_s=out.get("ttf"),
                    ttf_interval=None if iv is None else (float(iv[0]), float(iv[1])),
                    anomaly_score=float(out.get("anomalyScore", 0.0)),
                    attributions=list(expl[1]) if expl is not None and h == primary else [],
                    attributions_ts=expl[0] if expl is not None and h == primary else None,
                    reliability=reliability,
                    model_version=version,
                    alarm=state.alarm if h == primary else None,
                    anomaly_flag=out.get("anomalyFlag"),
                    anomaly_threshold=out.get("anomalyThreshold"),
                )
            )
        return result

    def _conditions(
        self,
        machine: str,
        state: AlarmState,
        ts: datetime,
        preds: list[Prediction] | None,
    ) -> list[AlertCondition]:
        """Changed alert conditions only (the manager deduplicates anyway)."""
        gate = self.gates[machine]
        wanted = {
            AlertKind.PREDICTED_WARNING: state.predicted_warning,
            AlertKind.PREDICTED_FAULT: state.predicted_fault,
            AlertKind.UNUSUAL_BEHAVIOUR: state.unusual,
            AlertKind.MODEL_DRIFT: gate.drifted,
        }
        out = []
        for kind, active in wanted.items():
            if gate.sent.get(kind, False) == active:
                continue
            gate.sent[kind] = active
            msg = self._message(machine, kind, preds) if active else f"{kind.value} ended"
            out.append(AlertCondition(machine, kind, active, ts, msg))
        return out

    def _message(self, machine: str, kind: AlertKind, preds: list[Prediction] | None) -> str:
        if kind is AlertKind.MODEL_DRIFT:
            return self._drift_message(machine)
        if kind is AlertKind.UNUSUAL_BEHAVIOUR:
            return (
                f"{machine}: unusual behaviour, cause unknown (autoencoder score above its "
                "threshold; the supervised model predicts no fault, so no part is named)"
            )
        primary = next((p for p in preds or [] if p.horizon_s == self.cfg.primary_horizon_s), None)
        cls = "FAULT" if kind is AlertKind.PREDICTED_FAULT else "WARNING"
        text = f"{machine}: predicted {cls} within {self.cfg.primary_horizon_s} s"
        if primary is not None:
            if primary.ttf_s is not None:
                text += f"; time to fault about {primary.ttf_s:.0f} s"
                if primary.ttf_interval is not None:
                    lo, hi = primary.ttf_interval
                    text += f" (interval {lo:.0f}-{hi:.0f} s)"
            text += f"; calibrated P({cls}) {primary.probs[HealthClass(cls)]:.2f}"
            if primary.attributions:
                top = primary.attributions[0]
                text += f"; top signal {top.sensor.value} ({top.part})"
        return text

    def _drift_message(self, machine: str) -> str:
        report = self.gates[machine].drift or {}
        psis = report.get("psi", {})
        alert = report.get("alert", {})
        parts = ", ".join(
            f"{ch} PSI {p:.2f} vs alert level {alert[ch]:.2f}"
            for ch, p in psis.items()
            if p is not None and ch in alert
        )
        since = report.get("since")
        when = (
            f" since {datetime.fromtimestamp(float(since), UTC):%H:%M:%S} UTC"
            if since is not None
            else ""
        )
        return (
            f"{machine}: operating conditions differ from the training data ({parts}){when}; "
            "predictions are marked DEGRADED and their confidence may not hold"
        )

    async def _check_stopped(self) -> None:
        """Machines whose link is down or which sent nothing recently: gate closed."""
        now = time.monotonic()
        conditions: list[AlertCondition] = []
        for m, gate in self.gates.items():
            st = self.engine.machine_state(m)
            silent = (
                gate.last_reading_wall is None
                or now - gate.last_reading_wall > self.cfg.prediction_stale_s
            )
            closed = st in LINK_DOWN or silent
            if closed and (gate.status != STOPPED or any(gate.sent.values())):
                gate.status, gate.run_len = STOPPED, 0
                gate.drifted, gate.drift = False, None
                if m in self.alarms:
                    self.alarms[m].reset()
                gate.alarms = OFF
                ts = self.engine.machine_watermark(m)
                if ts is not None:
                    conditions += self._conditions(m, OFF, ts, None)
        if conditions:
            self.engine.update_conditions(conditions)

    async def _close_all_gates(self) -> None:
        conditions: list[AlertCondition] = []
        for m, gate in self.gates.items():
            gate.status, gate.run_len, gate.alarms = UNAVAILABLE, 0, OFF
            gate.drifted, gate.drift = False, None
            ts = self.engine.machine_watermark(m)
            if ts is not None:
                conditions += self._conditions(m, OFF, ts, None)
        if conditions:
            self.engine.update_conditions(conditions)

    # ------------------------------------------------------------------ views
    def machine_status(self, machine: str) -> dict[str, Any]:
        gate = self.gates[machine]
        st = self.engine.machine_state(machine)
        status = gate.status
        if not self.ready:
            status = UNAVAILABLE
        elif st in LINK_DOWN:
            status = STOPPED
        return {
            "status": status,
            "runLen": gate.run_len,
            "window": self.info.window,
            "lastPredictionTs": gate.last_prediction_ts,
            "alarm": gate.alarms.alarm,
            "predictedFault": gate.alarms.predicted_fault,
            "unusual": gate.alarms.unusual,
        }

    def stale_after(self) -> timedelta:
        return timedelta(seconds=self.cfg.prediction_stale_s)

"""LineSimulator: four coupled machines, buffers, faults, commands, snapshot/restore.

All mutable state lives in plain dataclasses (``MachineDyn``, ``Event``, ...) collected by
``snapshot()`` into a JSON-serialisable ``SimSnapshot``; static data is rebuilt from the
configs and the ``RunPlan``. Same plan + same commands => bit-identical output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from twin_common.config.line import LineConfig
from twin_common.contracts import Command, CommandAck, Reading
from twin_common.enums import AckStatus, CommandType, MachineState, SensorName
from twin_sim.config import (
    NO_FAULT_LABEL,
    OVERLOAD_LABEL,
    FaultType,
    KindPhysics,
    SensorFaultType,
    SimConfig,
)
from twin_sim.physics import (
    NO_EFFECTS,
    current_true,
    fault_effects,
    fault_load_exponent,
    heat_w,
    load_factor,
    speed_rpm,
    steady_temp,
    thermal_step,
    vib_true,
)
from twin_sim.rng import get_state, make_streams, set_state
from twin_sim.scenario import RunPlan
from twin_sim.thresholds import Thresholds

SNAPSHOT_VERSION = 1
SOURCE_ID = "SOURCE"
SINK_ID = "SINK"
_SOURCE_QUEUE_CAP = 1
_N_PROCESS_NORMAL = 3  # load OU, progress, speed
_N_PROCESS_UNIFORM = 2  # burst occurrence, burst amplitude
_N_MEASURE = 6  # temp, vib mult, vib add, current mult, current add, load


@dataclass(slots=True)
class MachineDyn:
    """Mutable state of one machine."""

    machine_id: str
    temp: float
    load_mu: float
    band_lo: float
    band_hi: float
    load_eff: float
    speed_set: float
    seq: int = 0
    run_time: float = 0.0
    commanded_run: bool = True
    fault_latched: bool = False
    trip_signal: str | None = None
    maintenance_until: float | None = None
    warning: bool = False
    starved: bool = False
    blocked: bool = False
    load_dev: float = 0.0
    fault: FaultType | None = None
    fault_w: float = 0.0
    fault_progression_s: float = 0.0
    fault_log_index: int | None = None
    jam_t0: float | None = None
    jam_speed0: float = 0.0
    has_part: bool = False
    progress_s: float = 0.0
    holding: bool = False
    wait_in_s: float = 0.0
    wait_out_s: float = 0.0
    speed: float = 0.0
    vib: float = 0.0
    current: float = 0.0
    state: MachineState = MachineState.RUNNING
    root_cause: str = ""
    sf_type: SensorFaultType | None = None
    sf_sensor: SensorName | None = None
    sf_onset: float = 0.0
    sf_end: float = 0.0
    sf_drift_rate: float = 0.0
    sf_stuck_value: float = 0.0
    last_obs: dict[str, float] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Motor is commanded on and nothing prevents it from running."""
        return self.commanded_run and not self.fault_latched and self.maintenance_until is None


@dataclass(slots=True)
class Event:
    t: float
    order: int
    kind: str  # "fault" | "sensor_fault" | "command"
    machine: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FaultLogEntry:
    machine: str
    type: str
    planned_onset_s: float
    start_s: float | None = None
    fault_s: float | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MachineSample:
    """One machine at one output instant. Sensor values are observed (NaN on dropout)."""

    machine_id: str
    seq: int
    state: MachineState
    speed: float
    temp: float
    vib: float
    load: float
    current: float
    run_time: float
    fault_type: str
    root_cause: str
    sensor_fault: str
    buffer_in: int
    buffer_out: int
    dropout: bool
    true_vib: float
    true_temp: float
    true_current: float


@dataclass(frozen=True, slots=True)
class StepFrame:
    t_s: float
    samples: list[MachineSample]


class SimSnapshot(BaseModel):
    """Complete serialisable simulator state (``model_dump_json`` / ``model_validate_json``)."""

    model_config = ConfigDict(extra="forbid")

    version: int = SNAPSHOT_VERSION
    plan: RunPlan
    t: float
    machines: list[MachineDyn]
    buffers: list[int]
    source_queue: int
    next_arrival_s: float
    produced: int
    events: list[Event]
    event_counter: int
    idempotency: dict[str, CommandAck]
    fault_log: list[FaultLogEntry]
    rng: dict[str, dict[str, Any]]


class LineSimulator:
    def __init__(self, line: LineConfig, sim: SimConfig, plan: RunPlan) -> None:
        self.line = line
        self.cfg = sim
        self.plan = plan
        self.ids = [m.id for m in line.machines]
        self._index = {mid: i for i, mid in enumerate(self.ids)}
        self.kp: list[KindPhysics] = [sim.kinds[m.kind] for m in line.machines]
        self.thr: list[Thresholds] = [plan.machine_plan(mid).thresholds for mid in self.ids]
        self.capacity = [b.capacity for b in line.buffers]
        self.buf_in: list[int | None] = [None] * len(self.ids)
        self.buf_out: list[int | None] = [None] * len(self.ids)
        self.upstream: list[int | None] = [None] * len(self.ids)
        self.downstream: list[int | None] = [None] * len(self.ids)
        for bi, b in enumerate(line.buffers):
            src, dst = self._index[b.from_machine], self._index[b.to_machine]
            self.buf_out[src], self.buf_in[dst] = bi, bi
            self.downstream[src], self.upstream[dst] = dst, src
        self._start_ts: datetime = plan.start_ts
        self._rng = make_streams(plan.base_seed, plan.run_index)
        self._init_state()

    # ------------------------------------------------------------------ construction

    def _init_state(self) -> None:
        plan, op = self.plan, self.cfg.operation
        self.t = 0.0
        self.machines: list[MachineDyn] = []
        lo, hi = plan.load_band
        for i, mid in enumerate(self.ids):
            kp = self.kp[i]
            mu = plan.machine_plan(mid).load_mu
            speed = speed_rpm(kp, kp.speed_set_rpm, mu, NO_EFFECTS)
            temp = steady_temp(
                kp, heat_w(kp, mu, speed / kp.speed_set_rpm, NO_EFFECTS), 0.0, plan.ambient_c
            )
            self.machines.append(
                MachineDyn(
                    machine_id=mid,
                    temp=temp,
                    load_mu=mu,
                    band_lo=lo,
                    band_hi=hi,
                    load_eff=mu,
                    speed_set=kp.speed_set_rpm,
                    speed=speed,
                )
            )
        self.buffers = [min(op.initial_buffer_level, c) for c in self.capacity]
        self.source_queue = _SOURCE_QUEUE_CAP
        self.next_arrival_s = plan.source.arrival_interval_s
        self.produced = 0
        self.events: list[Event] = []
        self.event_counter = 0
        self.idempotency: dict[str, CommandAck] = {}
        self.fault_log: list[FaultLogEntry] = []
        for fe in plan.fault_events:
            self._push_event(
                fe.onset_s,
                "fault",
                fe.machine,
                {
                    "type": fe.type.value,
                    "progression_s": fe.progression_s,
                    "log": len(self.fault_log),
                },
            )
            self.fault_log.append(
                FaultLogEntry(machine=fe.machine, type=fe.type.value, planned_onset_s=fe.onset_s)
            )
        for se in plan.sensor_fault_events:
            self._push_event(
                se.onset_s,
                "sensor_fault",
                se.machine,
                {
                    "type": se.type.value,
                    "sensor": se.sensor.value if se.sensor else None,
                    "duration_s": se.duration_s,
                    "drift_rate": se.drift_rate_per_s,
                },
            )
        for c in plan.commands:
            self._push_event(
                c.at_s, "command", c.machine, {"type": c.type.value, "params": dict(c.params)}
            )
        self._update_states()

    def _push_event(self, t: float, kind: str, machine: str, payload: dict[str, Any]) -> None:
        self.events.append(Event(t, self.event_counter, kind, machine, payload))
        self.event_counter += 1
        self.events.sort(key=lambda e: (e.t, e.order))

    # ------------------------------------------------------------------ scenario injection

    def inject_fault(
        self, machine_id: str, fault: FaultType, progression_s: float | None = None
    ) -> tuple[AckStatus, str | None]:
        """Start ``fault`` on a machine now (simulation only, ADR 0015).

        The fault goes through the event queue like a planned one (so snapshot/restore stays
        exact) and starts at the next substep. Gradual faults default to the fastest configured
        progression time.
        """
        rejected = AckStatus.REJECTED
        if machine_id not in self._index:
            return rejected, f"unknown machine '{machine_id}'"
        i = self._index[machine_id]
        m, kind = self.machines[i], self.line.machines[i].kind
        if not self.cfg.faults.applicable(fault, kind):
            return rejected, f"fault '{fault.value}' does not apply to a {kind.value} machine"
        if m.fault is not None or m.fault_latched:
            return rejected, "another fault is active on this machine"
        if m.maintenance_until is not None:
            return rejected, "machine is in maintenance"
        if fault is FaultType.BELT_JAM:
            if not m.active:
                return rejected, "a jam needs the machine running"
            progression: float | None = None
        else:
            progression = (
                progression_s if progression_s is not None else self.cfg.faults.progression_s[0]
            )
        self._push_event(
            self.t,
            "fault",
            machine_id,
            {"type": fault.value, "progression_s": progression, "log": len(self.fault_log)},
        )
        self.fault_log.append(
            FaultLogEntry(machine=machine_id, type=fault.value, planned_onset_s=self.t)
        )
        return AckStatus.EXECUTED, None

    def schedule_plan(self, plan: RunPlan) -> int:
        """Schedule a plan's fault, sensor-fault and command events relative to now (ADR 0015).

        Only the events are taken; loads, thresholds and the source keep the current run's
        values. Returns the number of events scheduled.
        """
        t0, n = self.t, 0
        for fe in plan.fault_events:
            self._push_event(
                t0 + fe.onset_s,
                "fault",
                fe.machine,
                {
                    "type": fe.type.value,
                    "progression_s": fe.progression_s,
                    "log": len(self.fault_log),
                },
            )
            self.fault_log.append(
                FaultLogEntry(
                    machine=fe.machine, type=fe.type.value, planned_onset_s=t0 + fe.onset_s
                )
            )
            n += 1
        for se in plan.sensor_fault_events:
            self._push_event(
                t0 + se.onset_s,
                "sensor_fault",
                se.machine,
                {
                    "type": se.type.value,
                    "sensor": se.sensor.value if se.sensor else None,
                    "duration_s": se.duration_s,
                    "drift_rate": se.drift_rate_per_s,
                },
            )
            n += 1
        for c in plan.commands:
            self._push_event(
                t0 + c.at_s, "command", c.machine, {"type": c.type.value, "params": dict(c.params)}
            )
            n += 1
        return n

    # ------------------------------------------------------------------ public API

    def step(self, dt: float = 1.0) -> StepFrame:
        """Advance by ``dt`` seconds (split into substeps) and return one output sample."""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        max_sub = self.cfg.operation.max_substep_s
        n = max(1, math.ceil(dt / max_sub - 1e-9))
        sub = dt / n
        for _ in range(n):
            self._substep(sub)
        return self._observe()

    def fast_forward(self, seconds: float, record: bool = False) -> list[StepFrame]:
        """Run ``seconds`` of 1 s steps as fast as possible (identical to repeated step(1))."""
        frames: list[StepFrame] = []
        for _ in range(round(seconds)):
            frame = self.step(1.0)
            if record:
                frames.append(frame)
        return frames

    def run(self, duration_s: float | None = None) -> list[StepFrame]:
        return self.fast_forward(self.plan.duration_s if duration_s is None else duration_s, True)

    def timestamp(self, t_s: float) -> datetime:
        return self._start_ts + timedelta(seconds=t_s)

    def readings(self, frame: StepFrame) -> list[Reading]:
        """Wire-format Readings for a frame; machines in sensor dropout emit nothing."""
        ts = self.timestamp(frame.t_s)
        return [
            Reading(
                machine_id=s.machine_id,
                seq=s.seq,
                ts=ts,
                state=s.state,
                speed=s.speed,
                temp=s.temp,
                vib=s.vib,
                load=s.load,
                current=s.current,
                run_time=s.run_time,
            )
            for s in frame.samples
            if not s.dropout
        ]

    def apply_command(self, cmd: Command) -> CommandAck:
        cached = self.idempotency.get(cmd.idempotency_key)
        if cached is not None:
            return cached
        status, reason = self._execute(cmd.machine_id, cmd.type, dict(cmd.params))
        ack = CommandAck(
            command_id=cmd.command_id, status=status, reason=reason, ts=self.timestamp(self.t)
        )
        self.idempotency[cmd.idempotency_key] = ack
        return ack

    def snapshot(self) -> SimSnapshot:
        from copy import deepcopy

        return SimSnapshot(
            plan=self.plan,
            t=self.t,
            machines=deepcopy(self.machines),
            buffers=list(self.buffers),
            source_queue=self.source_queue,
            next_arrival_s=self.next_arrival_s,
            produced=self.produced,
            events=deepcopy(self.events),
            event_counter=self.event_counter,
            idempotency=dict(self.idempotency),
            fault_log=deepcopy(self.fault_log),
            rng={k: get_state(g) for k, g in self._rng.items()},
        )

    def restore(self, snap: SimSnapshot) -> None:
        from copy import deepcopy

        if snap.version != SNAPSHOT_VERSION:
            raise ValueError(f"unsupported snapshot version {snap.version}")
        if snap.plan != self.plan:
            raise ValueError("snapshot belongs to a different run plan; use from_snapshot()")
        self.t = snap.t
        self.machines = deepcopy(snap.machines)
        self.buffers = list(snap.buffers)
        self.source_queue = snap.source_queue
        self.next_arrival_s = snap.next_arrival_s
        self.produced = snap.produced
        self.events = deepcopy(snap.events)
        self.event_counter = snap.event_counter
        self.idempotency = dict(snap.idempotency)
        self.fault_log = deepcopy(snap.fault_log)
        for k, g in self._rng.items():
            set_state(g, snap.rng[k])

    @classmethod
    def from_snapshot(cls, line: LineConfig, sim: SimConfig, snap: SimSnapshot) -> LineSimulator:
        obj = cls(line, sim, snap.plan)
        obj.restore(snap)
        return obj

    def machine(self, machine_id: str) -> MachineDyn:
        return self.machines[self._index[machine_id]]

    # ------------------------------------------------------------------ commands

    def _execute(
        self, machine_id: str, ctype: CommandType, params: dict[str, Any]
    ) -> tuple[AckStatus, str | None]:
        if machine_id not in self._index:
            return AckStatus.REJECTED, f"unknown machine '{machine_id}'"
        i = self._index[machine_id]
        m, kp = self.machines[i], self.kp[i]
        rejected = AckStatus.REJECTED
        if ctype is CommandType.START:
            if m.maintenance_until is not None:
                return rejected, "machine is in maintenance"
            if m.fault_latched:
                return rejected, "machine is faulted; RESET first"
            m.commanded_run = True
        elif ctype is CommandType.STOP:
            if not m.commanded_run:
                return rejected, "machine is already stopped"
            m.commanded_run = False
        elif ctype is CommandType.SET_SPEED:
            speed = float(params["speed"])
            if not 0.0 < speed <= kp.speed_max_rpm:
                return rejected, f"speed must be in (0, {kp.speed_max_rpm}] RPM"
            m.speed_set = speed
        elif ctype is CommandType.SET_LOAD:
            load = float(params["load"])
            max_load = self.cfg.operation.max_load_pct
            if not 0.0 <= load <= max_load:
                return rejected, f"load must be in [0, {max_load}] %"
            delta = load - m.load_mu
            m.load_mu = load
            m.band_lo = min(max(m.band_lo + delta, 0.0), max_load)
            m.band_hi = min(max(m.band_hi + delta, 0.0), max_load)
        elif ctype is CommandType.SCHEDULE_MAINTENANCE:
            if m.maintenance_until is not None:
                return rejected, "maintenance already in progress"
            duration = float(params.get("durationS", self.cfg.operation.maintenance_duration_s))
            if duration <= 0:
                return rejected, "durationS must be > 0"
            m.maintenance_until = self.t + duration
            m.fault_latched = False
            m.trip_signal = None
            m.fault = None
            m.fault_w = 0.0
            m.fault_log_index = None
            m.jam_t0 = None
            m.warning = False
        elif ctype is CommandType.RESET:
            if not m.fault_latched:
                return rejected, "machine is not faulted"
            m.fault_latched = False
            m.trip_signal = None
            m.commanded_run = False
            if m.fault is FaultType.BELT_JAM:
                m.fault = None
                m.fault_log_index = None
                m.jam_t0 = None
        self._update_states()
        return AckStatus.EXECUTED, None

    # ------------------------------------------------------------------ simulation core

    def _process_events(self) -> None:
        while self.events and self.events[0].t <= self.t + 1e-9:
            ev = self.events.pop(0)
            if ev.kind == "fault":
                self._start_fault(ev)
            elif ev.kind == "sensor_fault":
                self._start_sensor_fault(ev)
            elif ev.kind == "command":
                self._execute(ev.machine, CommandType(ev.payload["type"]), ev.payload["params"])
        for m in self.machines:
            if m.maintenance_until is not None and self.t >= m.maintenance_until - 1e-9:
                m.maintenance_until = None
                m.commanded_run = True
            if m.sf_type is not None and self.t >= m.sf_end - 1e-9:
                m.sf_type = None
                m.sf_sensor = None

    def _start_fault(self, ev: Event) -> None:
        i = self._index[ev.machine]
        m = self.machines[i]
        log = self.fault_log[ev.payload["log"]]
        fault = FaultType(ev.payload["type"])
        if m.fault is not None or m.fault_latched:
            log.skipped_reason = "another fault active"
            return
        if m.maintenance_until is not None:
            log.skipped_reason = "machine in maintenance"
            return
        if fault is FaultType.BELT_JAM:
            if not m.active:
                log.skipped_reason = "machine not running"
                return
            m.fault = fault
            m.fault_log_index = ev.payload["log"]
            m.jam_t0 = self.t
            m.jam_speed0 = m.speed
            log.start_s = self.t
            self._trip(i, "jam")
            return
        m.fault = fault
        m.fault_w = 0.0
        m.fault_progression_s = float(ev.payload["progression_s"])
        m.fault_log_index = ev.payload["log"]
        log.start_s = self.t

    def _start_sensor_fault(self, ev: Event) -> None:
        m = self.machines[self._index[ev.machine]]
        if m.sf_type is not None:
            return
        p = ev.payload
        m.sf_type = SensorFaultType(p["type"])
        m.sf_sensor = SensorName(p["sensor"]) if p["sensor"] else None
        m.sf_onset = self.t
        m.sf_end = self.t + float(p["duration_s"])
        m.sf_drift_rate = float(p["drift_rate"])
        if m.sf_sensor is not None:
            m.sf_stuck_value = m.last_obs.get(m.sf_sensor.value, 0.0)

    def _trip(self, i: int, signal: str) -> None:
        m = self.machines[i]
        m.fault_latched = True
        m.trip_signal = signal
        m.warning = False
        if m.fault_log_index is not None:
            self.fault_log[m.fault_log_index].fault_s = self.t
        rec = self.plan.recovery
        if not rec.auto:
            return
        ops = self._rng["ops"]
        mid = m.machine_id
        if m.fault is FaultType.BELT_JAM:
            delay = float(ops.uniform(*rec.jam_clear_s))
            self._push_event(self.t + delay, "command", mid, {"type": "RESET", "params": {}})
            self._push_event(self.t + delay, "command", mid, {"type": "START", "params": {}})
        else:
            delay = float(ops.uniform(*rec.repair_delay_s))
            self._push_event(
                self.t + delay,
                "command",
                mid,
                {
                    "type": "SCHEDULE_MAINTENANCE",
                    "params": {"durationS": rec.maintenance_duration_s},
                },
            )

    def _substep(self, dt: float) -> None:
        self._process_events()
        t_new = self.t + dt
        self._source(t_new)
        self._flow(dt)
        proc = self._rng["process"]
        normals = proc.standard_normal(_N_PROCESS_NORMAL * len(self.machines)).tolist()
        uniforms = proc.random(_N_PROCESS_UNIFORM * len(self.machines)).tolist()
        self.t = t_new
        for i, m in enumerate(self.machines):
            self._physics(
                i,
                m,
                dt,
                normals[_N_PROCESS_NORMAL * i : _N_PROCESS_NORMAL * (i + 1)],
                uniforms[_N_PROCESS_UNIFORM * i : _N_PROCESS_UNIFORM * (i + 1)],
            )
        self._update_states()

    def _source(self, t_new: float) -> None:
        src = self.plan.source
        ops = self._rng["ops"]
        while t_new >= self.next_arrival_s:
            self.source_queue = min(self.source_queue + 1, _SOURCE_QUEUE_CAP)
            jitter = float(ops.uniform(-src.jitter_frac, src.jitter_frac))
            self.next_arrival_s += src.arrival_interval_s * (1.0 + jitter)

    def _try_push(self, i: int) -> None:
        b = self.buf_out[i]
        if b is None:
            self.produced += 1
            self.machines[i].holding = False
        elif self.buffers[b] < self.capacity[b]:
            self.buffers[b] += 1
            self.machines[i].holding = False

    def _try_pull(self, i: int) -> bool:
        b = self.buf_in[i]
        if b is None:
            if self.source_queue > 0:
                self.source_queue -= 1
                return True
            return False
        if self.buffers[b] > 0:
            self.buffers[b] -= 1
            return True
        return False

    def _flow(self, dt: float) -> None:
        op = self.cfg.operation
        for i in reversed(range(len(self.machines))):
            m = self.machines[i]
            if not m.active:
                m.wait_in_s = m.wait_out_s = 0.0
                m.starved = m.blocked = False
                continue
            kp = self.kp[i]
            if m.holding:
                self._try_push(i)
            if not m.has_part and not m.holding and self._try_pull(i):
                m.has_part, m.progress_s = True, 0.0
            if m.has_part:
                m.progress_s += dt * m.speed_set / kp.speed_set_rpm
                if m.progress_s >= kp.cycle_time_s:
                    leftover = m.progress_s - kp.cycle_time_s
                    m.has_part, m.holding, m.progress_s = False, True, 0.0
                    self._try_push(i)
                    if not m.holding and self._try_pull(i):
                        m.has_part, m.progress_s = True, leftover
            m.wait_in_s = 0.0 if (m.has_part or m.holding) else m.wait_in_s + dt
            m.wait_out_s = m.wait_out_s + dt if m.holding else 0.0
            m.starved = m.wait_in_s >= op.starved_after_s
            m.blocked = m.wait_out_s >= op.blocked_after_s

    def _physics(
        self, i: int, m: MachineDyn, dt: float, normals: list[float], uniforms: list[float]
    ) -> None:
        kp, fp, op = self.kp[i], self.cfg.faults, self.cfg.operation
        lp = self.plan.load_process
        active = m.active

        # Load: OU demand around the operating point, then a first-order lag.
        a = math.exp(-dt / lp.tau_s)
        sigma = lp.sigma_frac * (m.band_hi - m.band_lo)
        m.load_dev = m.load_dev * a + sigma * math.sqrt(1.0 - a * a) * normals[0]
        demand = min(max(m.load_mu + m.load_dev, m.band_lo), m.band_hi)
        target = (op.idle_load_pct if m.starved or m.blocked else demand) if active else 0.0
        m.load_eff += (target - m.load_eff) * (1.0 - math.exp(-dt / op.load_lag_tau_s))

        # Gradual fault progress (frozen once tripped, uncapped otherwise).
        gradual = m.fault is not None and m.fault is not FaultType.BELT_JAM
        if gradual and active and m.fault is not None:
            s = fp.progress_noise_sigma
            g = load_factor(m.load_eff, fault_load_exponent(m.fault, fp), fp)
            m.fault_w += g * dt / m.fault_progression_s * math.exp(s * normals[1] - 0.5 * s * s)
        eff = fault_effects(m.fault if gradual else None, m.fault_w, kp, fp)

        # Speed.
        if active:
            det = speed_rpm(kp, m.speed_set, m.load_eff, eff)
            m.speed = max(det * (1.0 + kp.speed_noise_frac * eff.speed_jitter * normals[2]), 0.0)
        elif m.jam_t0 is not None:
            jam = fp.belt_jam
            m.speed = m.jam_speed0 * math.exp(-(self.t - m.jam_t0) / jam.speed_tau_s)
        else:
            m.speed = 0.0
        ratio = m.speed / kp.speed_set_rpm

        # Temperature, vibration, current (true values).
        heat = heat_w(kp, m.load_eff, ratio, eff) if active else 0.0
        m.temp = thermal_step(m.temp, heat, kp, eff.cooling_loss, self.plan.ambient_c, dt)
        m.vib = vib_true(kp, m.load_eff, ratio, eff) if m.speed > 0.0 else 0.0
        if active:
            m.current = current_true(kp, m.load_eff, eff)
        elif m.jam_t0 is not None and self.t - m.jam_t0 <= fp.belt_jam.trip_after_s + 1e-9:
            m.current = fp.belt_jam.stall_current_x_rated * kp.current_rated_a
        else:
            m.current = 0.0
        if m.speed > 0.0:
            m.run_time += dt

        # Stick-slip bursts (lubrication loss) are stored for the observation layer.
        m.last_obs["_burst"] = 0.0
        if eff.burst_prob > 0.0 and m.speed > 0.0 and uniforms[0] < eff.burst_prob:
            lub = fp.lubrication_loss
            lo, hi = lub.burst_amp_range
            amp = lo + (hi - lo) * uniforms[1]
            m.last_obs["_burst"] = amp * lub.burst_amp_x_fault * kp.thresholds.vib_fault

        # Protection and warning logic on the deterministic signals.
        thr = self.thr[i]
        if active:
            signal = thr.fault_signal(m.vib, m.temp, m.current)
            if signal is not None:
                self._trip(i, signal)
                return
        if not m.fault_latched and m.maintenance_until is None:
            m.warning = thr.warning_state(m.warning, m.vib, m.temp, m.current, op.hysteresis_frac)

    def _update_states(self) -> None:
        for m in self.machines:
            if m.maintenance_until is not None:
                m.state = MachineState.MAINTENANCE
            elif m.fault_latched:
                m.state = MachineState.FAULT
            elif not m.commanded_run:
                m.state = MachineState.IDLE
            elif m.warning:
                m.state = MachineState.WARNING
            elif m.starved:
                m.state = MachineState.STARVED
            elif m.blocked:
                m.state = MachineState.BLOCKED
            else:
                m.state = MachineState.RUNNING
        for i, m in enumerate(self.machines):
            m.root_cause = self._root_cause(i, 0)

    def _root_cause(self, i: int, depth: int) -> str:
        m = self.machines[i]
        st = m.state
        if st is MachineState.RUNNING:
            return ""
        if st is MachineState.STARVED or st is MachineState.BLOCKED:
            nxt = self.upstream[i] if st is MachineState.STARVED else self.downstream[i]
            if nxt is None:
                return SOURCE_ID if st is MachineState.STARVED else SINK_ID
            if depth >= len(self.machines):
                return self.ids[nxt]
            if self.machines[nxt].state is MachineState.RUNNING:
                return self.ids[nxt]
            return self._root_cause(nxt, depth + 1)
        return m.machine_id

    # ------------------------------------------------------------------ observation layer

    def _observe(self) -> StepFrame:
        noise = self.cfg.measurement
        meas = self._rng["measure"].standard_normal(_N_MEASURE * len(self.machines)).tolist()
        samples: list[MachineSample] = []
        for i, m in enumerate(self.machines):
            e = meas[_N_MEASURE * i : _N_MEASURE * (i + 1)]
            obs = {
                "speed": m.speed,
                "temp": m.temp + noise.temp_sigma_c * e[0],
                "vib": max(m.vib * (1.0 + noise.vib_mult * e[1]) + noise.vib_add_mm_s * e[2], 0.0)
                + m.last_obs.get("_burst", 0.0),
                "current": max(
                    m.current * (1.0 + noise.current_mult * e[3]) + noise.current_add_a * e[4], 0.0
                ),
                "load": max(m.load_eff + noise.load_sigma_pct * e[5], 0.0),
            }
            dropout = False
            sf_label = NO_FAULT_LABEL
            if m.sf_type is not None:
                if m.sf_type is SensorFaultType.DROPOUT:
                    dropout = True
                    sf_label = SensorFaultType.DROPOUT.value
                elif m.sf_sensor is not None:
                    key = m.sf_sensor.value
                    sf_label = f"{m.sf_type.value}:{key}"
                    if m.sf_type is SensorFaultType.STUCK:
                        obs[key] = m.sf_stuck_value
                    else:
                        drifted = obs[key] + m.sf_drift_rate * (self.t - m.sf_onset)
                        obs[key] = drifted if key == "temp" else max(drifted, 0.0)
            burst = m.last_obs.get("_burst", 0.0)
            m.last_obs = dict(obs)
            m.last_obs["_burst"] = burst
            m.seq += 1
            if dropout:
                obs = dict.fromkeys(obs, math.nan)
            fault_label = (
                m.fault.value
                if m.fault is not None
                else (OVERLOAD_LABEL if m.fault_latched else NO_FAULT_LABEL)
            )
            bi, bo = self.buf_in[i], self.buf_out[i]
            samples.append(
                MachineSample(
                    machine_id=m.machine_id,
                    seq=m.seq,
                    state=m.state,
                    speed=obs["speed"],
                    temp=obs["temp"],
                    vib=obs["vib"],
                    load=obs["load"],
                    current=obs["current"],
                    run_time=m.run_time,
                    fault_type=fault_label,
                    root_cause=m.root_cause,
                    sensor_fault=sf_label,
                    buffer_in=-1 if bi is None else self.buffers[bi],
                    buffer_out=-1 if bo is None else self.buffers[bo],
                    dropout=dropout,
                    true_vib=m.vib,
                    true_temp=m.temp,
                    true_current=m.current,
                )
            )
        return StepFrame(self.t, samples)


def thresholds_table(sim: LineSimulator) -> dict[str, dict[str, float]]:
    """Per-machine absolute thresholds of this run (for manifests and plots)."""
    from dataclasses import asdict

    return {mid: asdict(thr) for mid, thr in zip(sim.ids, sim.thr, strict=True)}


__all__ = [
    "SINK_ID",
    "SOURCE_ID",
    "LineSimulator",
    "MachineSample",
    "SimSnapshot",
    "StepFrame",
]

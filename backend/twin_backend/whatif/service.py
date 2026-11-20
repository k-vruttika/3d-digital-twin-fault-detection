"""What-if service (Phase 11 Part C, ADR 0031): ``POST /whatif``.

Flow: safety rules on the change (the same rules as a real command) -> the active source forks
its simulation (``Capability.WHATIF``: snapshot, apply the change to a copy, fast-forward the
copy and an unchanged copy) -> with live inference configured, the projected readings are
scored by the deployed model in a separate worker pool -> two projections (baseline and
scenario) with sensor values, projected states ("ghost" states), predictions and alarms.

Nothing here touches the live line: no command is sent, nothing is published or stored.
Operators apply a change for real through ``POST /commands`` (the dashboard asks first).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from twin_backend.adapters.base import WhatIfUnsupportedError
from twin_backend.adapters.manager import SourceManager
from twin_backend.clock import Clock
from twin_backend.commands.safety import SafetyRules
from twin_backend.engine.twin import TwinEngine
from twin_backend.whatif.models import (
    MachineProjection,
    ProjectionPoint,
    Variant,
    WhatIfPayload,
    WhatIfResponse,
)
from twin_backend.whatif.scoring import Scored, ScoringError, WhatIfScorer
from twin_common.config.commands import CommandsConfig
from twin_common.config.inference import InferenceConfig
from twin_common.config.line import LineConfig
from twin_common.config.whatif import WhatIfConfig
from twin_common.contracts import Command, Reading
from twin_common.enums import AckStatus, Capability, MachineState
from twin_common.logging import get_logger
from twin_common.whatif import WhatIfSimRequest

log = get_logger(__name__)

WHATIF_ISSUER = "what-if"
FAULT_STATES = (MachineState.FAULT,)
WARNING_STATES = (MachineState.WARNING,)
OFF_NOTE = "Live inference is off: the projection shows the simulator only, without predictions."


class WhatIfRejectedError(ValueError):
    """The change is not allowed (safety rules) or the simulator rejects it: HTTP 422."""


class WhatIfService:
    def __init__(
        self,
        cfg: WhatIfConfig,
        commands_cfg: CommandsConfig,
        line: LineConfig,
        manager: SourceManager,
        engine: TwinEngine,
        clock: Clock,
        root: Path,
        inference_cfg: InferenceConfig | None = None,
        scorer: WhatIfScorer | None = None,
    ) -> None:
        self.cfg = cfg
        self.line = line
        self.manager = manager
        self.engine = engine
        self.clock = clock
        self.kinds = {m.id: m.kind for m in line.machines}
        self.safety = SafetyRules(commands_cfg)
        self.inference_cfg = inference_cfg
        self.scorer = scorer
        if scorer is None and inference_cfg is not None and inference_cfg.enabled:
            self.scorer = WhatIfScorer(inference_cfg, line, root, cfg.rows_per_step)

    # ------------------------------------------------------------------ request
    def check(self, payload: WhatIfPayload) -> tuple[Command, int]:
        """Validate the change like a command; returns it and the projection length."""
        horizon = payload.horizon_s or self.cfg.horizon_default_s
        if horizon > self.cfg.horizon_max_s:
            raise WhatIfRejectedError(
                f"horizonS {horizon} exceeds the maximum of {self.cfg.horizon_max_s} s"
            )
        try:
            cmd = Command(
                command_id=uuid.uuid4().hex,
                idempotency_key="what-if",
                machine_id=payload.machine_id,
                type=payload.type,
                params=payload.params,
                issued_by=WHATIF_ISSUER,
                ts=self.clock.now(),
            )
        except ValidationError as exc:
            msgs = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())
            raise WhatIfRejectedError(msgs) from exc
        verdict = self.safety.check(
            cmd, self.engine.machine_state(cmd.machine_id), self.kinds[cmd.machine_id]
        )
        if not verdict.allowed:
            raise WhatIfRejectedError(verdict.reason or "not allowed")
        return cmd, horizon

    async def run(self, payload: WhatIfPayload) -> WhatIfResponse:
        """Raises ``WhatIfUnsupportedError`` (409), ``WhatIfRejectedError`` (422) or
        ``twin_backend.adapters.base.WhatIfError`` (502: the source could not fork)."""
        if not self.manager.supports(Capability.WHATIF):
            kind = self.manager.active_kind
            raise WhatIfUnsupportedError(kind.value if kind is not None else "no")
        cmd, horizon = self.check(payload)
        started = time.perf_counter()
        sim = await self.manager.whatif(
            WhatIfSimRequest(
                machine_id=cmd.machine_id,
                type=cmd.type,
                params=cmd.params,
                horizon_s=horizon,
                history_s=self.cfg.history_s,
            ),
            self.cfg.request_timeout_s,
        )
        if sim.action_status is AckStatus.REJECTED:
            raise WhatIfRejectedError(sim.action_reason or "the simulator rejects the change")
        scores: dict[str, dict[tuple[str, int], Scored]] = {}
        version: str | None = None
        note: str | None = None
        if self.scorer is None:
            note = OFF_NOTE
        else:
            try:
                scores = await self.scorer.score(
                    sim.history, {"baseline": sim.baseline, "scenario": sim.scenario}
                )
                version = self.scorer.model_version
            except ScoringError as exc:
                log.warning("whatif.scoring_failed", error=str(exc))
                note = f"The model could not score the projection ({exc}); simulator only."
        history: dict[str, list[ProjectionPoint]] = {}
        for r in sim.history:
            history.setdefault(r.machine_id, []).append(
                point(r, sim.as_of, scores.get("baseline", {}).get((r.machine_id, r.seq)))
            )
        return WhatIfResponse(
            as_of=sim.as_of,
            machine_id=cmd.machine_id,
            type=cmd.type,
            params=dict(cmd.params),
            horizon_s=horizon,
            ghost_horizons_s=self.cfg.ghost_horizons_s,
            primary_horizon_s=self.inference_cfg.primary_horizon_s
            if self.scorer is not None and self.inference_cfg is not None and version is not None
            else None,
            model=version,
            model_note=note,
            history=history,
            baseline=self._variant(sim.baseline, sim.as_of, scores.get("baseline", {}), horizon),
            scenario=self._variant(sim.scenario, sim.as_of, scores.get("scenario", {}), horizon),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )

    def _variant(
        self,
        readings: list[Reading],
        as_of: datetime,
        scored: dict[tuple[str, int], Scored],
        horizon: int,
    ) -> Variant:
        by_machine: dict[str, list[ProjectionPoint]] = {m: [] for m in self.kinds}
        for r in readings:
            if r.machine_id in by_machine:
                by_machine[r.machine_id].append(point(r, as_of, scored.get((r.machine_id, r.seq))))
        return Variant(machines={m: self._machine(pts, horizon) for m, pts in by_machine.items()})

    def _machine(self, pts: list[ProjectionPoint], horizon: int) -> MachineProjection:
        ghost: dict[str, MachineState] = {}
        for h in self.cfg.ghost_horizons_s:
            at = state_at(pts, h)
            if at is not None:
                ghost[str(h)] = at
        end = state_at(pts, horizon)
        if end is not None:
            ghost["end"] = end
        pf = [p.p_fault for p in pts if p.p_fault is not None]
        return MachineProjection(
            points=pts,
            ghost=ghost,
            first_warning_s=first(pts, lambda p: p.state in WARNING_STATES),
            first_fault_s=first(pts, lambda p: p.state in FAULT_STATES),
            first_alarm_s=first(pts, lambda p: p.alarm is True),
            max_p_fault=max(pf) if pf else None,
        )

    async def close(self) -> None:
        if self.scorer is not None:
            await self.scorer.close()


def point(r: Reading, as_of: datetime, s: Scored | None) -> ProjectionPoint:
    return ProjectionPoint(
        t_s=round((r.ts - as_of).total_seconds()),
        ts=r.ts,
        state=r.state,
        speed=r.speed,
        temp=r.temp,
        vib=r.vib,
        load=r.load,
        current=r.current,
        gate=s.gate if s is not None else None,
        p_warning=s.p_warning if s is not None else None,
        p_fault=s.p_fault if s is not None else None,
        top_class=s.top_class if s is not None else None,
        alarm=s.alarm if s is not None else None,
        unusual=s.unusual if s is not None else None,
        ttf_s=s.ttf_s if s is not None else None,
    )


def state_at(pts: list[ProjectionPoint], t_s: int) -> MachineState | None:
    """The simulated state at ``t_s`` (the last reading at or before it)."""
    best: ProjectionPoint | None = None
    for p in pts:
        if p.t_s <= t_s:
            best = p
    return best.state if best is not None else None


def first(pts: list[ProjectionPoint], pred: Callable[[ProjectionPoint], bool]) -> int | None:
    for p in pts:
        if pred(p):
            return p.t_s
    return None

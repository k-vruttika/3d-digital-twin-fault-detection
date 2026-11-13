"""Command safety rules from ``config/commands.yaml`` (ADR 0014).

**Fail-safe principle:** STOP is always allowed, in every state (including OFFLINE, STALE,
FAULT and an unknown state). Stopping a machine must never be blocked by the twin's view of
it; the machine itself decides whether the stop is still meaningful. The evaluator exempts STOP
in code, and the config schema rejects any rule that names STOP.

Every other command is checked against the configured rules (forbidden / required states) and
the per-machine-kind parameter limits. Without any state for the machine yet (no data since
startup) only STOP is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from twin_common.config.commands import FAIL_SAFE_COMMANDS, CommandsConfig
from twin_common.contracts import Command
from twin_common.enums import CommandType, MachineKind, MachineState


@dataclass(frozen=True, slots=True)
class Verdict:
    allowed: bool
    reason: str | None = None


ALLOWED = Verdict(True)


class SafetyRules:
    def __init__(self, cfg: CommandsConfig) -> None:
        self.cfg = cfg

    def check(self, cmd: Command, state: MachineState | None, kind: MachineKind) -> Verdict:
        if cmd.type in FAIL_SAFE_COMMANDS:
            return ALLOWED
        if state is None:
            return Verdict(
                False, f"the state of {cmd.machine_id} is not known yet; only STOP is allowed"
            )
        for rule in self.cfg.rules:
            if cmd.type not in rule.types:
                continue
            if rule.forbidden_states is not None and state in rule.forbidden_states:
                return Verdict(False, f"{rule.reason} (machine is {state.value})")
            if rule.required_states is not None and state not in rule.required_states:
                return Verdict(False, f"{rule.reason} (machine is {state.value})")
        return self._limits(cmd, kind)

    def _limits(self, cmd: Command, kind: MachineKind) -> Verdict:
        limits = self.cfg.limits[kind]
        checks = {
            CommandType.SET_SPEED: ("speed", limits.speed_rpm, "RPM"),
            CommandType.SET_LOAD: ("load", limits.load_pct, "%"),
        }
        if cmd.type not in checks:
            return ALLOWED
        name, rng, unit = checks[cmd.type]
        value = float(cmd.params[name])  # presence and type checked by the Command contract
        if not rng.min <= value <= rng.max:
            return Verdict(
                False,
                f"{name} {value:g} {unit} is outside [{rng.min:g}, {rng.max:g}] {unit} for a "
                f"{kind.value} machine",
            )
        return ALLOWED

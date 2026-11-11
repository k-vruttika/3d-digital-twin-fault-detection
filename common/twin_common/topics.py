"""MQTT topic helpers: ``{prefix}/{machineId}/telemetry|state|cmd|cmd/ack``."""

from __future__ import annotations

DEFAULT_TOPIC_PREFIX = "plant/line1"

_FORBIDDEN = ("/", "+", "#")


def _check_machine_id(machine_id: str) -> str:
    if not machine_id or any(ch in machine_id for ch in _FORBIDDEN):
        raise ValueError(f"invalid machine id for an MQTT topic: {machine_id!r}")
    return machine_id


def _base(prefix: str, machine_id: str) -> str:
    return f"{prefix.rstrip('/')}/{_check_machine_id(machine_id)}"


def telemetry_topic(machine_id: str, prefix: str = DEFAULT_TOPIC_PREFIX) -> str:
    return f"{_base(prefix, machine_id)}/telemetry"


def state_topic(machine_id: str, prefix: str = DEFAULT_TOPIC_PREFIX) -> str:
    return f"{_base(prefix, machine_id)}/state"


def cmd_topic(machine_id: str, prefix: str = DEFAULT_TOPIC_PREFIX) -> str:
    return f"{_base(prefix, machine_id)}/cmd"


def cmd_ack_topic(machine_id: str, prefix: str = DEFAULT_TOPIC_PREFIX) -> str:
    return f"{_base(prefix, machine_id)}/cmd/ack"

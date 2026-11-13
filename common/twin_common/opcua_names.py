"""OPC UA naming shared by the simulator's server and the backend adapter.

No asyncua import here: these are plain names, so twin_common stays dependency-free.
"""

from __future__ import annotations

from twin_common.enums import CommandType

READING_EVENT_TYPE = "ReadingEventType"

# Reading attribute -> OPC UA browse name (variables and ReadingEventType fields).
FIELD_BROWSE_NAMES: dict[str, str] = {
    "machine_id": "MachineId",
    "seq": "Seq",
    "ts": "Ts",
    "state": "State",
    "speed": "Speed",
    "temp": "Temperature",
    "vib": "Vibration",
    "load": "Load",
    "current": "Current",
    "run_time": "RunTime",
}

# Command type -> (method browse name, name of the numeric parameter or None).
COMMAND_METHODS: dict[CommandType, tuple[str, str | None]] = {
    CommandType.START: ("Start", None),
    CommandType.STOP: ("Stop", None),
    CommandType.SET_SPEED: ("SetSpeed", "speed"),
    CommandType.SET_LOAD: ("SetLoad", "load"),
    CommandType.SCHEDULE_MAINTENANCE: ("ScheduleMaintenance", None),
    CommandType.RESET: ("Reset", None),
}

# String arguments every command method takes before its optional numeric value.
COMMAND_ARGS: tuple[str, ...] = ("commandId", "idempotencyKey", "issuedBy")

# Line-level method (simulation only, ADR 0015): one String argument with the ScenarioRequest
# JSON; returns the ScenarioAck JSON.
SCENARIO_METHOD = "InjectScenario"

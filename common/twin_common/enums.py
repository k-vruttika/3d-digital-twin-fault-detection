"""Enumerations shared by every LineTwin package (CLAUDE.md section 6)."""

from __future__ import annotations

from enum import StrEnum


class MachineState(StrEnum):
    """Operational state of a single machine."""

    RUNNING = "RUNNING"
    IDLE = "IDLE"
    WARNING = "WARNING"
    FAULT = "FAULT"
    STARVED = "STARVED"
    BLOCKED = "BLOCKED"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"
    STALE = "STALE"


class Quality(StrEnum):
    """Data-quality flag attached to each reading by the ingest pipeline."""

    OK = "OK"
    FILLED = "FILLED"
    LATE = "LATE"
    SUSPECT = "SUSPECT"


class HealthClass(StrEnum):
    """ML target classes for machine health."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    FAULT = "FAULT"


class SensorName(StrEnum):
    """Sensor channels carried by a Reading (field names in Python and in YAML config)."""

    SPEED = "speed"
    TEMP = "temp"
    VIB = "vib"
    LOAD = "load"
    CURRENT = "current"


class CommandType(StrEnum):
    START = "START"
    STOP = "STOP"
    SET_SPEED = "SET_SPEED"
    SET_LOAD = "SET_LOAD"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    RESET = "RESET"


class AckStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    TIMEOUT = "TIMEOUT"


class Reliability(StrEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"


class MachineKind(StrEnum):
    CONVEYOR = "CONVEYOR"
    CNC = "CNC"
    INSPECTION = "INSPECTION"


class AdapterKind(StrEnum):
    FILE = "file"
    MOCK = "mock"
    MQTT = "mqtt"
    OPCUA = "opcua"


class Capability(StrEnum):
    """Optional features a source adapter supports beyond streaming readings."""

    COMMANDS = "commands"
    WHATIF = "whatif"
    HISTORY = "history"


class AdapterStatus(StrEnum):
    """Connection status of a source adapter."""

    STARTING = "STARTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    EOF = "EOF"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class AlertSeverity(StrEnum):
    """How urgent an alert is."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class QuarantineReason(StrEnum):
    """Why the ingest pipeline rejected a message."""

    DECODE_ERROR = "DECODE_ERROR"
    MAPPING_ERROR = "MAPPING_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    NON_FINITE = "NON_FINITE"  # runTime not finite (sensor NaN = missing, not an error)
    NO_SENSOR_DATA = "NO_SENSOR_DATA"  # every sensor value of the machine is missing
    UNKNOWN_MACHINE = "UNKNOWN_MACHINE"
    OUT_OF_RANGE = "OUT_OF_RANGE"


# CLAUDE.md section 6: ML target mapping. OFFLINE and STALE describe the *data link*, not
# machine health, so they deliberately have no label (see to_health_class).
_HEALTH_MAP: dict[MachineState, HealthClass] = {
    MachineState.RUNNING: HealthClass.NORMAL,
    MachineState.IDLE: HealthClass.NORMAL,
    MachineState.STARVED: HealthClass.NORMAL,
    MachineState.BLOCKED: HealthClass.NORMAL,
    MachineState.MAINTENANCE: HealthClass.NORMAL,
    MachineState.WARNING: HealthClass.WARNING,
    MachineState.FAULT: HealthClass.FAULT,
}


class UnlabelableStateError(ValueError):
    """Raised when a MachineState has no health label (OFFLINE, STALE)."""


def to_health_class(state: MachineState) -> HealthClass:
    """Map a machine state to its ML health target.

    Raises:
        UnlabelableStateError: for OFFLINE and STALE, which must be excluded from training.
    """
    try:
        return _HEALTH_MAP[state]
    except KeyError:
        raise UnlabelableStateError(
            f"state {state.value} has no health label; exclude it from training data"
        ) from None


def is_labelable(state: MachineState) -> bool:
    """True if the state maps to a health class."""
    return state in _HEALTH_MAP

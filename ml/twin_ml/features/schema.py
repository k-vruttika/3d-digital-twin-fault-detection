"""Channel schema of a built dataset: sensors, engineered channels, kinds and sample period.

The simulator datasets use :data:`SIM_SCHEMA` (five sensors, ``vib_norm``, the three machine
kinds, one row per second). The real-data datasets (Phase 10, ADR 0028) carry their own schema in
``meta.json``; a dataset without one is a simulator dataset. Model input sizes, summary feature
names, explanation maps and time conversions all read the schema of the split they work on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twin_common.enums import MachineKind

SIM_SENSORS: tuple[str, ...] = ("speed", "temp", "vib", "load", "current")
VIB_NORM = "vib_norm"


@dataclass(frozen=True)
class ChannelSchema:
    """What one row of a dataset holds and how far apart rows are.

    * ``sensors``: raw sensor columns, in order (the model inputs);
    * ``vib_norm``: whether the engineered ``vib_norm`` channel is appended (needs ``speed`` and
      ``vib``);
    * ``kinds``: machine kind names, in one-hot order;
    * ``step_s``: seconds between consecutive rows (regular grid).
    """

    sensors: tuple[str, ...]
    kinds: tuple[str, ...]
    vib_norm: bool = False
    step_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.sensors:
            raise ValueError("a schema needs at least one sensor")
        if len(set(self.sensors)) != len(self.sensors):
            raise ValueError("duplicate sensor names")
        if not self.kinds or len(set(self.kinds)) != len(self.kinds):
            raise ValueError("kinds must be non-empty and unique")
        if self.vib_norm and not {"speed", "vib"} <= set(self.sensors):
            raise ValueError("vib_norm needs both speed and vib sensors")
        if self.step_s <= 0:
            raise ValueError("step_s must be > 0")

    # ------------------------------------------------------------------ channels
    @property
    def value_channels(self) -> tuple[str, ...]:
        return (*self.sensors, VIB_NORM) if self.vib_norm else self.sensors

    @property
    def mask_channels(self) -> tuple[str, ...]:
        return tuple(f"{s}_missing" for s in self.sensors)

    @property
    def kind_channels(self) -> tuple[str, ...]:
        return tuple(f"kind_{k}" for k in self.kinds)

    @property
    def sequence_channels(self) -> tuple[str, ...]:
        return (*self.value_channels, *self.mask_channels, *self.kind_channels)

    def value_sensor(self, channel: str) -> str:
        """The sensor a value channel is computed from (``vib_norm`` -> ``vib``)."""
        if channel in self.sensors:
            return channel
        if self.vib_norm and channel == VIB_NORM:
            return "vib"
        raise KeyError(f"unknown value channel {channel!r}")

    def kind_index(self, kind: str) -> int:
        return self.kinds.index(str(kind))

    # ------------------------------------------------------------------ time
    def steps(self, seconds: float) -> int:
        """Whole rows in ``seconds`` (must be a multiple of the step)."""
        n = seconds / self.step_s
        if abs(n - round(n)) > 1e-9:
            raise ValueError(f"{seconds} s is not a multiple of the step {self.step_s} s")
        return round(n)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return {
            "sensors": list(self.sensors),
            "kinds": list(self.kinds),
            "vib_norm": self.vib_norm,
            "step_s": self.step_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChannelSchema:
        return cls(
            sensors=tuple(str(s) for s in data["sensors"]),
            kinds=tuple(str(k) for k in data["kinds"]),
            vib_norm=bool(data.get("vib_norm", False)),
            step_s=float(data.get("step_s", 1.0)),
        )

    @classmethod
    def of_meta(cls, meta: dict[str, Any]) -> ChannelSchema:
        """The schema recorded in a dataset's ``meta.json``; simulator builds have none."""
        raw = meta.get("schema")
        return SIM_SCHEMA if raw is None else cls.from_dict(raw)


SIM_SCHEMA = ChannelSchema(
    sensors=SIM_SENSORS,
    kinds=tuple(k.value for k in MachineKind),
    vib_norm=True,
    step_s=1.0,
)

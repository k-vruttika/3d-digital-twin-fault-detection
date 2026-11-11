"""Schema for ``config/line.yaml``: machines, sensors, buffers, part mappings, load ranges."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from twin_common.enums import MachineKind, SensorName

LOAD_PERCENT_MIN = 0.0
LOAD_PERCENT_MAX = 100.0


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True)


class SensorSpec(ConfigModel):
    """Catalog entry: unit and physically plausible range of a sensor channel."""

    unit: str = Field(min_length=1)
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> SensorSpec:
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self


class MachineConfig(ConfigModel):
    id: str = Field(min_length=1)
    kind: MachineKind
    name: str = ""
    sensors: list[str] = Field(min_length=1)
    parts: list[str] = Field(min_length=1)
    sensor_parts: dict[str, str] = Field(description="sensor -> part, for 3D highlighting")

    @model_validator(mode="after")
    def _consistent(self) -> MachineConfig:
        errors: list[str] = []
        for label, items in (("sensors", self.sensors), ("parts", self.parts)):
            dups = sorted(k for k, n in Counter(items).items() if n > 1)
            if dups:
                errors.append(f"{label}: duplicates {dups}")
        sensors, parts = set(self.sensors), set(self.parts)
        for sensor, part in self.sensor_parts.items():
            if sensor not in sensors:
                errors.append(
                    f"sensor_parts: '{sensor}' is not one of this machine's sensors "
                    f"{sorted(sensors)}"
                )
            if part not in parts:
                errors.append(
                    f"sensor_parts: part '{part}' (for sensor '{sensor}') is not one of "
                    f"this machine's parts {sorted(parts)}"
                )
        unmapped = sorted(sensors - set(self.sensor_parts))
        if unmapped:
            errors.append(f"sensor_parts: sensors without a part mapping {unmapped}")
        if errors:
            raise ValueError(f"machine '{self.id}': " + "; ".join(errors))
        return self


class BufferConfig(ConfigModel):
    id: str = Field(min_length=1)
    from_machine: str = Field(alias="from", min_length=1)
    to_machine: str = Field(alias="to", min_length=1)
    capacity: int = Field(ge=1, description="Maximum number of parts held")


class LoadRange(ConfigModel):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> LoadRange:
        if not LOAD_PERCENT_MIN <= self.min < self.max <= LOAD_PERCENT_MAX:
            raise ValueError(
                f"load range must satisfy {LOAD_PERCENT_MIN} <= min < max <= "
                f"{LOAD_PERCENT_MAX} (got {self.min}-{self.max})"
            )
        return self


class LoadRanges(ConfigModel):
    """Operating-condition bands; must be contiguous: low.max == medium.min, etc."""

    low: LoadRange
    medium: LoadRange
    high: LoadRange

    @model_validator(mode="after")
    def _contiguous(self) -> LoadRanges:
        pairs = (
            ("low", self.low, "medium", self.medium),
            ("medium", self.medium, "high", self.high),
        )
        for a_name, a, b_name, b in pairs:
            if a.max > b.min:
                raise ValueError(
                    f"load ranges overlap: {a_name} ends at {a.max} but {b_name} starts at {b.min}"
                )
            if a.max < b.min:
                raise ValueError(
                    f"gap between load ranges: {a_name} ends at {a.max} but {b_name} "
                    f"starts at {b.min}"
                )
        return self


class LineConfig(ConfigModel):
    line_id: str = Field(min_length=1)
    sensors: dict[str, SensorSpec] = Field(min_length=1)
    machines: list[MachineConfig] = Field(min_length=1, description="In material-flow order")
    buffers: list[BufferConfig] = Field(default_factory=list)
    load_ranges: LoadRanges

    @model_validator(mode="after")
    def _cross_refs(self) -> LineConfig:
        errors: list[str] = []
        valid_sensor_names = {s.value for s in SensorName}
        for name in self.sensors:
            if name not in valid_sensor_names:
                errors.append(
                    f"sensors: '{name}' is not a Reading sensor field "
                    f"(allowed: {', '.join(sorted(valid_sensor_names))})"
                )

        machine_ids = [m.id for m in self.machines]
        buffer_ids = [b.id for b in self.buffers]
        for label, ids in (("machine", machine_ids), ("buffer", buffer_ids)):
            dups = sorted(k for k, n in Counter(ids).items() if n > 1)
            if dups:
                errors.append(f"duplicate {label} id(s): {dups}")

        known = sorted(self.sensors)
        for i, m in enumerate(self.machines):
            for s in m.sensors:
                if s not in self.sensors:
                    errors.append(
                        f"machines[{i}] ({m.id}).sensors: unknown sensor '{s}' "
                        f"(known: {', '.join(known)})"
                    )

        existing = set(machine_ids)
        seen_links: set[tuple[str, str]] = set()
        for i, b in enumerate(self.buffers):
            for end, mid in (("from", b.from_machine), ("to", b.to_machine)):
                if mid not in existing:
                    errors.append(
                        f"buffers[{i}] ({b.id}).{end}: unknown machine '{mid}' "
                        f"(known: {', '.join(machine_ids)})"
                    )
            if b.from_machine == b.to_machine:
                errors.append(f"buffers[{i}] ({b.id}): links machine '{b.from_machine}' to itself")
            link = (b.from_machine, b.to_machine)
            if link in seen_links:
                errors.append(f"buffers[{i}] ({b.id}): duplicate link {link[0]} -> {link[1]}")
            seen_links.add(link)

        if errors:
            raise ValueError("\n".join(errors))
        return self

    def machine(self, machine_id: str) -> MachineConfig:
        for m in self.machines:
            if m.id == machine_id:
                return m
        raise KeyError(machine_id)

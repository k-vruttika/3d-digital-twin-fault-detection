"""Injectable fault types for the dashboard's scenario panel (simulation only, ADR 0015).

Read from the simulator's ``config/simulator.yaml`` ``faults`` section without importing the
simulator, so the backend stays independent of ``twin_sim``. Only the names, the machine kinds
each fault applies to, whether it progresses gradually and whether it is held out are read; the
physics parameters stay the simulator's business. A test checks this reading against
``twin_sim.config`` so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twin_common.config.loader import ConfigError, read_yaml
from twin_common.enums import MachineKind

# Gradual faults have a load-dependent progression (``load_exponent``); a jam is immediate.
_GRADUAL_KEY = "load_exponent"


@dataclass(frozen=True, slots=True)
class FaultOption:
    fault_type: str
    kinds: tuple[MachineKind, ...]
    gradual: bool
    held_out: bool


@dataclass(frozen=True, slots=True)
class ScenarioOptions:
    faults: tuple[FaultOption, ...]
    progression_s: tuple[float, float] | None


EMPTY = ScenarioOptions(faults=(), progression_s=None)


def load_scenario_options(path: str | Path) -> ScenarioOptions:
    """The fault catalog in ``path``; empty when the file does not exist (no simulator)."""
    p = Path(path)
    if not p.is_file():
        return EMPTY
    data = read_yaml(p)
    faults = data.get("faults")
    if not isinstance(faults, dict):
        raise ConfigError(f"{p}: 'faults' must be a mapping")
    options: list[FaultOption] = []
    for name, entry in faults.items():
        if not isinstance(entry, dict) or "kinds" not in entry:
            continue  # shared settings such as progression_s
        options.append(_option(p, str(name), entry))
    prog = faults.get("progression_s")
    progression: tuple[float, float] | None = None
    if isinstance(prog, list) and len(prog) == 2:
        progression = (float(prog[0]), float(prog[1]))
    return ScenarioOptions(faults=tuple(options), progression_s=progression)


def _option(path: Path, name: str, entry: dict[str, Any]) -> FaultOption:
    try:
        kinds = tuple(MachineKind(k) for k in entry["kinds"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: faults.{name}.kinds: {exc}") from exc
    return FaultOption(
        fault_type=name,
        kinds=kinds,
        gradual=_GRADUAL_KEY in entry,
        held_out=bool(entry.get("held_out", False)),
    )

"""Diagnostic plots: one run per fault type, a sensor-fault run and a line-coupling run.

Generated into ``experiments/results/sim_examples/``. Needs matplotlib (dev dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twin_common.config.line import LineConfig
from twin_common.enums import CommandType, MachineState, SensorName
from twin_sim.config import (
    FaultSchedule,
    FaultType,
    RecoveryParams,
    ScenarioConfig,
    ScheduledFault,
    ScheduledSensorFault,
    ScriptedCommand,
    SensorFaultSchedule,
    SensorFaultType,
    SimConfig,
    load_scenario,
)
from twin_sim.line import LineSimulator, StepFrame
from twin_sim.scenario import make_run_plan

# Reference palette (dataviz skill): one series hue, status colours for states.
SERIES = "#2a78d6"
ALT = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
STATE_COLORS: dict[MachineState, str] = {
    MachineState.RUNNING: "#0ca30c",
    MachineState.WARNING: "#fab219",
    MachineState.FAULT: "#d03b3b",
    MachineState.STARVED: "#86b6ef",
    MachineState.BLOCKED: "#ec835a",
    MachineState.MAINTENANCE: "#5e5d59",
    MachineState.IDLE: "#c3c2b7",
}
WARN_COLOR, FAULT_COLOR = "#c98500", "#d03b3b"


@dataclass(frozen=True)
class Example:
    name: str
    machine: str
    scenario: ScenarioConfig


def _examples() -> list[Example]:
    out: list[Example] = []
    for fault, mid in (
        (FaultType.BEARING_WEAR, "CNC-01"),
        (FaultType.OVERHEATING, "CNV-01"),
        (FaultType.SPINDLE_IMBALANCE, "CNC-01"),
        (FaultType.BELT_JAM, "CNV-01"),
        (FaultType.LUBRICATION_LOSS, "CNC-01"),
    ):
        out.append(
            Example(
                fault.value,
                mid,
                ScenarioConfig(
                    name=f"example_{fault.value}",
                    duration_s=2400,
                    load_ranges=["medium"],
                    faults=FaultSchedule(
                        scheduled=[ScheduledFault(type=fault, machine=mid, onset_s=300)]
                    ),
                ),
            )
        )
    out.append(
        Example(
            "sensor_faults",
            "INSP-01",
            ScenarioConfig(
                name="example_sensor_faults",
                duration_s=2400,
                load_ranges=["medium"],
                sensor_faults=SensorFaultSchedule(
                    scheduled=[
                        ScheduledSensorFault(
                            type=SensorFaultType.STUCK,
                            machine="INSP-01",
                            sensor=SensorName.TEMP,
                            onset_s=300,
                            duration_s=400,
                        ),
                        ScheduledSensorFault(
                            type=SensorFaultType.DRIFT,
                            machine="INSP-01",
                            sensor=SensorName.VIB,
                            onset_s=900,
                            duration_s=600,
                        ),
                        ScheduledSensorFault(
                            type=SensorFaultType.DROPOUT,
                            machine="INSP-01",
                            onset_s=1700,
                            duration_s=200,
                        ),
                    ]
                ),
            ),
        )
    )
    return out


def _run(
    line: LineConfig, sim_cfg: SimConfig, scenario: ScenarioConfig, seed: int
) -> tuple[LineSimulator, list[StepFrame]]:
    sim = LineSimulator(line, sim_cfg, make_run_plan(scenario, sim_cfg, line, seed, 0))
    return sim, sim.run()


def _state_band(ax: Any, t: np.ndarray, states: list[MachineState]) -> None:
    start = 0
    for k in range(1, len(states) + 1):
        if k == len(states) or states[k] is not states[start]:
            ax.axvspan(t[start] - 0.5, t[k - 1] + 0.5, color=STATE_COLORS[states[start]], lw=0)
            start = k
    ax.set_yticks([])


def _style(ax: Any) -> None:
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def _plot_machine(ex: Example, sim: LineSimulator, frames: list[StepFrame], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    i = sim.ids.index(ex.machine)
    t = np.array([f.t_s for f in frames])
    samples = [f.samples[i] for f in frames]
    thr = sim.thr[i]
    panels: list[tuple[str, str, tuple[float, float] | None]] = [
        ("speed", "speed (RPM)", None),
        ("temp", "temp (°C)", (thr.temp_warning, thr.temp_fault)),
        ("vib", "vib (mm/s RMS)", (thr.vib_warning, thr.vib_fault)),
        ("current", "current (A)", (thr.current_warning, thr.current_fault)),
        ("load", "load (%)", None),
    ]
    fig, axes = plt.subplots(
        len(panels) + 1,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1] * len(panels) + [0.35]},
    )
    for ax, (key, label, lines) in zip(axes, panels, strict=False):
        ax.plot(t, [getattr(s, key) for s in samples], color=SERIES, lw=1.0)
        if lines is not None:
            ax.axhline(lines[0], color=WARN_COLOR, lw=1.2, ls="--")
            ax.axhline(lines[1], color=FAULT_COLOR, lw=1.2, ls="--")
        ax.set_ylabel(label, fontsize=9, color=INK)
        _style(ax)
    states = [s.state for s in samples]
    _state_band(axes[-1], t, states)
    axes[-1].set_ylabel("state", fontsize=9)
    axes[-1].set_xlabel("simulated time (s)", fontsize=9)
    used = [s for s in STATE_COLORS if s in set(states)]
    handles: list[Any] = [Patch(color=STATE_COLORS[s], label=s.value) for s in used]
    handles += [
        Line2D([], [], color=WARN_COLOR, ls="--", label="WARNING threshold"),
        Line2D([], [], color=FAULT_COLOR, ls="--", label="FAULT threshold"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8, frameon=False
    )
    fig.suptitle(
        f"{ex.name} on {ex.machine} (onset 300 s; load band {sim.plan.load_range}, "
        f"ambient {sim.plan.ambient_c:.1f} °C)",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_coupling(sim: LineSimulator, frames: list[StepFrame], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = np.array([f.t_s for f in frames])
    n = len(sim.ids)
    fig, axes = plt.subplots(n + len(sim.capacity), 1, figsize=(11, 8), sharex=True)
    seen: set[MachineState] = set()
    for i, mid in enumerate(sim.ids):
        states = [f.samples[i].state for f in frames]
        seen.update(states)
        _state_band(axes[i], t, states)
        axes[i].set_ylabel(mid, fontsize=9, rotation=0, ha="right", va="center")
    for b, buf in enumerate(sim.line.buffers):
        ax = axes[n + b]
        up = sim.ids.index(buf.from_machine)
        ax.step(t, [f.samples[up].buffer_out for f in frames], color=SERIES, lw=1.2, where="post")
        ax.set_ylim(-0.5, buf.capacity + 0.5)
        ax.set_ylabel(f"{buf.id} parts", fontsize=9)
        _style(ax)
    axes[-1].set_xlabel("simulated time (s)", fontsize=9)
    handles = [Patch(color=STATE_COLORS[s], label=s.value) for s in STATE_COLORS if s in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("Line coupling: operator stops CNC-01 at 600 s, restarts at 1200 s", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


SPEED_STEP_S = 200.0
SPEED_STEPS_RPM = (10000.0, 7500.0)
SPEED_PLOT_DURATION_S = 1600.0
SPEED_PLOT_PROGRESSION_S = 1500.0


def _speed_step_scenario(fault: FaultType) -> ScenarioConfig:
    """CNC-01 alternating between the ends of its speed range, one gradual fault at 300 s."""
    n = int(SPEED_PLOT_DURATION_S // SPEED_STEP_S)
    commands = [
        ScriptedCommand(
            at_s=k * SPEED_STEP_S,
            machine="CNC-01",
            type=CommandType.SET_SPEED,
            params={"speed": SPEED_STEPS_RPM[k % 2]},
        )
        for k in range(1, n)
    ]
    return ScenarioConfig(
        name=f"speed_steps_{fault.value}",
        duration_s=SPEED_PLOT_DURATION_S,
        load_ranges=["medium"],
        recovery=RecoveryParams(auto=False),
        commands=commands,
        faults=FaultSchedule(
            scheduled=[
                ScheduledFault(
                    type=fault,
                    machine="CNC-01",
                    onset_s=300,
                    progression_s=SPEED_PLOT_PROGRESSION_S,
                )
            ]
        ),
    )


def _plot_speed_steps(line: LineConfig, sim_cfg: SimConfig, seed: int, path: Path) -> None:
    """Imbalance vibration (∝ speed²) vs bearing-wear vibration (∝ speed) under speed steps."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = {
        f: _run(line, sim_cfg, _speed_step_scenario(f), seed)
        for f in (FaultType.SPINDLE_IMBALANCE, FaultType.BEARING_WEAR)
    }
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    sim0, frames0 = runs[FaultType.SPINDLE_IMBALANCE]
    i = sim0.ids.index("CNC-01")
    nominal = sim0.kp[i].speed_set_rpm
    t = np.array([f.t_s for f in frames0])
    axes[0].plot(t, [f.samples[i].speed for f in frames0], color=SERIES, lw=1.0)
    axes[0].set_ylabel("speed (RPM)", fontsize=9)
    # Vibration normalised by the speed ratio: bearing wear (∝ speed) becomes smooth,
    # imbalance (∝ speed²) keeps a step at every speed change.
    norm_ax = axes[3]
    for fault, color in ((FaultType.SPINDLE_IMBALANCE, SERIES), (FaultType.BEARING_WEAR, ALT)):
        _, frames = runs[fault]
        pts = [
            (f.t_s, f.samples[i].vib / (f.samples[i].speed / nominal))
            for f in frames
            if f.samples[i].speed > 0.5 * nominal
        ]
        norm_ax.plot(
            [p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.0, label=fault.value
        )
    norm_ax.set_ylabel("vib ÷ (speed/nominal)\n(mm/s)", fontsize=9)
    norm_ax.legend(fontsize=8, frameon=False, loc="upper left")
    for ax, (fault, (s, frames)) in zip(axes[1:3], runs.items(), strict=True):
        thr = s.thr[i]
        vib = [f.samples[i].vib for f in frames]
        ax.plot(t[: len(vib)], vib, color=SERIES, lw=1.0)
        ax.axhline(thr.vib_warning, color=WARN_COLOR, lw=1.2, ls="--")
        ax.axhline(thr.vib_fault, color=FAULT_COLOR, lw=1.2, ls="--")
        k_fault = next(
            (k for k, f in enumerate(frames) if f.samples[i].state is MachineState.FAULT), None
        )
        if k_fault is not None:
            ax.axvline(frames[k_fault].t_s, color=MUTED, lw=1.0, ls=":")
        ax.set_ylabel(f"vib (mm/s)\n{fault.value}", fontsize=9)
    for ax in axes:
        _style(ax)
    axes[-1].set_xlabel("simulated time (s); fault onset at 300 s", fontsize=9)
    handles = [
        Line2D([], [], color=WARN_COLOR, ls="--", label="WARNING threshold"),
        Line2D([], [], color=FAULT_COLOR, ls="--", label="FAULT threshold"),
        Line2D([], [], color=MUTED, ls=":", label="FAULT (machine trips)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "CNC-01 speed steps 7500/10000 RPM: imbalance vibration scales with speed², "
        "bearing-wear vibration with speed",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_examples(
    *, line: LineConfig, sim: SimConfig, out_dir: Path, seed: int, scenario_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ex in _examples():
        s, frames = _run(line, sim, ex.scenario, seed)
        path = out_dir / f"{ex.name}.png"
        _plot_machine(ex, s, frames, path)
        paths.append(path)
    s, frames = _run(line, sim, load_scenario(scenario_dir / "cnc_stop.yaml"), seed)
    path = out_dir / "line_coupling_cnc_stop.png"
    _plot_coupling(s, frames, path)
    paths.append(path)
    path = out_dir / "imbalance_vs_bearing_speed_steps.png"
    _plot_speed_steps(line, sim, seed, path)
    paths.append(path)
    return paths

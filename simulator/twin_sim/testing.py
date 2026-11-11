"""Pytest fixtures for simulator tests (registered as a plugin by the root conftest.py).

Not used by production code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from twin_common.config import load_line_config
from twin_common.config.line import LineConfig
from twin_sim.config import ScenarioConfig, SimConfig, load_sim_config
from twin_sim.line import LineSimulator
from twin_sim.scenario import MachinePlan, make_run_plan
from twin_sim.thresholds import scaled_thresholds

REPO_ROOT = Path(__file__).resolve().parents[2]  # simulator/twin_sim/testing.py -> repo
CONFIG_DIR = REPO_ROOT / "config"

SimFactory = Callable[..., LineSimulator]


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture(scope="session")
def line_cfg() -> LineConfig:
    return load_line_config(CONFIG_DIR / "line.yaml")


@pytest.fixture(scope="session")
def single_line_cfg() -> LineConfig:
    """Single-station mode: CNV-01 only, no buffers (config/line_single.yaml)."""
    return load_line_config(CONFIG_DIR / "line_single.yaml")


@pytest.fixture(scope="session")
def sim_cfg() -> SimConfig:
    return load_sim_config(CONFIG_DIR / "simulator.yaml")


@pytest.fixture(scope="session")
def make_sim(line_cfg: LineConfig, sim_cfg: SimConfig) -> SimFactory:
    """Build a simulator; ``thr_scale`` forces every threshold to base * scale."""

    def factory(
        scenario: ScenarioConfig,
        seed: int = 1,
        run_index: int = 0,
        thr_scale: float | None = None,
    ) -> LineSimulator:
        plan = make_run_plan(scenario, sim_cfg, line_cfg, seed, run_index)
        if thr_scale is not None:
            machines = [
                MachinePlan(
                    machine_id=m.machine_id,
                    load_mu=m.load_mu,
                    thresholds=scaled_thresholds(
                        sim_cfg.kinds[line_cfg.machine(m.machine_id).kind], thr_scale
                    ),
                )
                for m in plan.machines
            ]
            plan = plan.model_copy(update={"machines": machines})
        return LineSimulator(line_cfg, sim_cfg, plan)

    return factory

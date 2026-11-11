from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from twin_common.config.line import LineConfig
from twin_common.config.loader import ConfigError
from twin_sim.config import ScenarioConfig, SimConfig, check_scenario_compat, load_sim_config
from twin_sim.feasibility import check_fault_feasibility


def _with_faults(sim_cfg: SimConfig, **faults: object) -> SimConfig:
    data = sim_cfg.model_dump(mode="json")
    for key, value in faults.items():
        if isinstance(value, dict):
            data["faults"][key].update(value)
        else:
            data["faults"][key] = value
    return SimConfig.model_validate(data)


def test_default_config_is_feasible(sim_cfg: SimConfig) -> None:
    assert check_fault_feasibility(sim_cfg) == []


def test_unreachable_overheating_is_rejected(sim_cfg: SimConfig) -> None:
    # The pre-revision model: cooling loss only, no fault heat.
    bad = _with_faults(
        sim_cfg,
        overheating={"fault_heat_dt_k": 0.0, "cooling_loss_max": 0.75, "cooling_loss_rate": 0.75},
    )
    problems = check_fault_feasibility(bad)
    assert any("overheating" in p and "best case" in p for p in problems)


def test_too_fast_progression_violates_lead_time(sim_cfg: SimConfig) -> None:
    bad = _with_faults(sim_cfg, progression_s=[200.0, 400.0])
    problems = check_fault_feasibility(bad)
    assert any("minimum lead time" in p for p in problems)


def test_healthy_warning_margin_is_checked(sim_cfg: SimConfig) -> None:
    data = sim_cfg.model_dump(mode="json")
    data["kinds"]["CNC"]["thresholds"]["temp_warning"] = 50.0
    problems = check_fault_feasibility(SimConfig.model_validate(data))
    assert any("CNC: healthy machine reaches WARNING" in p for p in problems)


def test_loader_raises_config_error(sim_cfg: SimConfig, tmp_path: Path) -> None:
    data = sim_cfg.model_dump(mode="json")
    data["faults"]["progression_s"] = [200.0, 400.0]
    path = tmp_path / "simulator.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="infeasible"):
        load_sim_config(path)
    assert load_sim_config(path, check_feasibility=False).faults.progression_s == (200.0, 400.0)


def test_invalid_yaml_values_are_readable(sim_cfg: SimConfig, tmp_path: Path) -> None:
    data = sim_cfg.model_dump(mode="json")
    data["kinds"]["CNC"]["thresholds"]["vib_warning"] = 9.0  # above vib_fault
    path = tmp_path / "simulator.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"kinds\.CNC\.thresholds.*vib: warning"):
        load_sim_config(path)


def test_scenario_compat(sim_cfg: SimConfig, line_cfg: LineConfig) -> None:
    ok = ScenarioConfig(name="t")
    check_scenario_compat(ok, sim_cfg, line_cfg)
    too_wide = ScenarioConfig(name="t", threshold_perturbation_pct=20.0)
    with pytest.raises(ConfigError, match="threshold_perturbation_pct"):
        check_scenario_compat(too_wide, sim_cfg, line_cfg)
    bad_machine = ScenarioConfig.model_validate(
        {"name": "t", "commands": [{"at_s": 1, "machine": "NOPE", "type": "STOP"}]}
    )
    with pytest.raises(ConfigError, match="unknown machine 'NOPE'"):
        check_scenario_compat(bad_machine, sim_cfg, line_cfg)


def test_shipped_scenarios_load(config_dir: Path, sim_cfg: SimConfig, line_cfg: LineConfig) -> None:
    from twin_sim.config import load_scenario

    files = sorted((config_dir / "scenarios").glob("*.yaml"))
    assert {f.stem for f in files} >= {"default", "quick", "e5_lubrication_loss", "cnc_stop"}
    for f in files:
        check_scenario_compat(load_scenario(f), sim_cfg, line_cfg)
    default = load_scenario(config_dir / "scenarios" / "default.yaml")
    assert [s.machine for s in default.speed_changes] == ["CNC-01"]


def _with_cnc_speeds(sim_cfg: SimConfig, lo: float, hi: float) -> SimConfig:
    data = sim_cfg.model_dump(mode="json")
    data["kinds"]["CNC"]["speed_operating_rpm"] = [lo, hi]
    return SimConfig.model_validate(data)


def test_validator_covers_operating_speed_range(sim_cfg: SimConfig) -> None:
    # At 12000 RPM bearing vibration (proportional to speed) trips too soon after onset.
    problems = check_fault_feasibility(_with_cnc_speeds(sim_cfg, 7500.0, 12000.0))
    assert any("bearing_wear on CNC at 12000 RPM" in p and "lead time" in p for p in problems)
    assert not any("7500 RPM" in p for p in problems)


def test_operating_range_must_contain_setpoint(sim_cfg: SimConfig) -> None:
    with pytest.raises(ValueError, match="speed_operating_rpm"):
        _with_cnc_speeds(sim_cfg, 8500.0, 10000.0)


def test_scenario_speed_changes_checked(sim_cfg: SimConfig, line_cfg: LineConfig) -> None:
    def scenario(lo: float, hi: float) -> ScenarioConfig:
        return ScenarioConfig.model_validate(
            {
                "name": "t",
                "speed_changes": [
                    {"machine": "CNC-01", "interval_s": [300, 900], "speed_rpm": [lo, hi]}
                ],
            }
        )

    check_scenario_compat(scenario(7500.0, 10000.0), sim_cfg, line_cfg)
    with pytest.raises(ConfigError, match="outside the validated operating range"):
        check_scenario_compat(scenario(7500.0, 11000.0), sim_cfg, line_cfg)
    # A wider validated range would allow 7000 RPM, but the CNC would then be slower than
    # the arrival rate (6 s x 8000/7000 = 6.86 s > 6.5 s).
    wide = _with_cnc_speeds(sim_cfg, 7000.0, 10000.0)
    with pytest.raises(ConfigError, match=r"cycle time 6\.86 s is not below"):
        check_scenario_compat(scenario(7000.0, 9000.0), wide, line_cfg)

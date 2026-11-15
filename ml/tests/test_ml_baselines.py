"""Baselines on hand-built examples."""

from __future__ import annotations

import numpy as np
import pytest

from twin_common.enums import MachineKind
from twin_common.enums import MachineState as S
from twin_ml.baselines.persistence import PersistenceBaseline
from twin_ml.baselines.threshold import RULE_SENSORS, classify, nominal_limits
from twin_ml.data.labels import FAULT, NORMAL, WARNING
from twin_ml.features.inputs import KINDS
from twin_ml.testing import make_split
from twin_sim.config import SimConfig


def test_persistence_predicts_the_current_health_class() -> None:
    states = [S.RUNNING] * 3 + [S.WARNING] * 2 + [S.FAULT] * 2 + [S.STARVED] * 3
    split = make_split(states, window=2, horizons=(1, 2))
    preds = PersistenceBaseline().predict(split)
    expected = split.at_end("health").tolist()
    for h in split.horizons:
        assert preds.label(h).tolist() == expected
    assert expected == [NORMAL, NORMAL, WARNING, WARNING, FAULT, FAULT, NORMAL]
    assert preds.ttf is None
    assert preds.fault_type is None


def test_nominal_limits_come_from_the_unperturbed_config(sim_cfg: SimConfig) -> None:
    limits = nominal_limits(sim_cfg)
    cnc = sim_cfg.kinds[MachineKind.CNC]
    rated = cnc.current_noload_a + cnc.current_load_a
    assert limits[MachineKind.CNC].warning.tolist() == pytest.approx(
        [
            cnc.thresholds.vib_warning,
            cnc.thresholds.temp_warning,
            cnc.thresholds.current_warning_x * rated,
        ]
    )
    assert limits[MachineKind.CNC].fault[0] == cnc.thresholds.vib_fault


@pytest.mark.parametrize("kind", list(MachineKind))
def test_threshold_rule_at_and_just_below_each_threshold(
    sim_cfg: SimConfig, kind: MachineKind
) -> None:
    limits = nominal_limits(sim_cfg)
    lim = limits[kind]
    code = KINDS.index(kind)
    healthy = lim.warning * 0.5
    rows, expected = [healthy], [NORMAL]
    for i, _ in enumerate(RULE_SENSORS):
        for value, label in (
            (np.nextafter(lim.warning[i], 0), NORMAL),
            (lim.warning[i], WARNING),
            (np.nextafter(lim.fault[i], 0), WARNING),
            (lim.fault[i], FAULT),
        ):
            row = healthy.copy()
            row[i] = value
            rows.append(row)
            expected.append(label)
    got = classify(np.array(rows), np.full(len(rows), code), limits)
    assert got.tolist() == expected

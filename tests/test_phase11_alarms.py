"""Phase 11: the live alarm rule equals the evaluated rule (ADR 0022 / 0030).

The backend implements the rule without importing twin_ml; these tests compare it with the
experiment code (``twin_ml.evaluate.early_warning``) and check that the live settings are the
evaluated ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from twin_backend.inference.alarms import MachineAlarms, Sustain, argmax, sustain_windows
from twin_common.config.inference import load_inference_config
from twin_common.config.loader import read_yaml
from twin_ml.config import load_eval_config
from twin_ml.data.labels import FAULT, WARNING
from twin_ml.evaluate.early_warning import sustain_windows as ml_sustain_windows
from twin_ml.evaluate.early_warning import sustained
from twin_ml.testing import CONFIG_DIR


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_sustain_equals_the_evaluated_sustained(k: int) -> None:
    rng = np.random.default_rng(k)
    for _ in range(20):
        x = rng.random(300) < rng.uniform(0.2, 0.9)
        s = Sustain(k)
        live = np.array([s.update(bool(v)) for v in x])
        np.testing.assert_array_equal(live, sustained(x, k))


def test_alarm_rule_on_probabilities_equals_experiment_rule() -> None:
    """Uncalibrated argmax >= WARNING, sustained k windows (evaluator.early_warning)."""
    rng = np.random.default_rng(7)
    k = sustain_windows(3)
    probs = rng.dirichlet([2.0, 1.0, 1.0], size=2000)
    probs[::17] = [0.4, 0.4, 0.2]  # ties go to the first class, as numpy.argmax
    labels = np.argmax(probs, axis=1)
    alarms = MachineAlarms(k)
    states = [alarms.update(p.tolist(), None) for p in probs]
    np.testing.assert_array_equal([s.alarm for s in states], sustained(labels >= WARNING, k))
    np.testing.assert_array_equal([s.fault for s in states], sustained(labels == FAULT, k))
    assert all(argmax(p.tolist()) == int(np.argmax(p)) for p in probs)
    # the union of the two alert conditions is exactly the evaluated alarm
    assert all((s.predicted_warning or s.predicted_fault) == s.alarm for s in states)
    assert not any(s.predicted_warning and s.predicted_fault for s in states)


def test_live_settings_are_the_evaluated_ones() -> None:
    eval_cfg = load_eval_config(CONFIG_DIR / "models" / "eval.yaml")
    inf = load_inference_config(CONFIG_DIR / "inference.yaml")
    e5 = read_yaml(CONFIG_DIR / "models" / "e5.yaml")
    # sustain time and window step: k windows of 1 s, as at the dense test stride
    assert sustain_windows(eval_cfg.early_warning.sustain_s) == ml_sustain_windows(
        eval_cfg.early_warning.sustain_s, 1.0
    )
    assert inf.primary_horizon_s == e5["primary"]["horizon"]
    assert [s.value for s in inf.gate.operating] == [s.value for s in eval_cfg.subsets.operating]
    assert inf.ig_steps == e5["explain"]["ig_steps"]


def test_unusual_only_without_a_supervised_alarm() -> None:
    a = MachineAlarms(3)
    normal, warning = [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]
    states = [a.update(normal, True) for _ in range(3)]
    assert states[-1].unusual
    states = [a.update(warning, True) for _ in range(3)]
    assert states[-1].alarm
    assert not states[-1].unusual
    a.reset()
    assert not a.state.anomaly
    assert not a.state.alarm

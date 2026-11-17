"""Phase 9 Part C exploratory follow-ups: config, drift-segment windows, window agreement, and
that the follow-up hashes leave every earlier explanation hash unchanged (no torch, no xgboost)."""

from __future__ import annotations

import numpy as np

from twin_common.enums import MachineState
from twin_ml.config import load_e5_config, load_e5_followup_config, load_eval_config
from twin_ml.experiments import explain_eval
from twin_ml.features.inputs import SENSORS
from twin_ml.testing import CONFIG_DIR, make_split

FOLLOWUP = CONFIG_DIR / "models" / "e5_followup.yaml"


def test_followup_config_loads_and_stays_out_of_e5_yaml() -> None:
    f = load_e5_followup_config(FOLLOWUP)
    e5 = load_e5_config(CONFIG_DIR / "models" / "e5.yaml")
    assert f.ae_shares.detector in e5.anomaly.models
    assert f.ae_shares.source_study in e5.studies
    assert f.ig_sensitivity.ig_steps != e5.explain.ig_steps  # a different step count
    assert e5.explain.ig_steps == 64  # the pre-registered value is untouched
    assert set(f.ig_sensitivity.models) <= set(e5.explain.primary_models)


def test_thin_stride_and_cap() -> None:
    t = np.arange(100, dtype=np.float64)
    cand = np.arange(100, dtype=np.int64)[::-1]  # any order in, end-time order out
    sel = explain_eval.thin(cand, t, 10, 100)
    assert sel.tolist() == list(range(0, 100, 10))
    capped = explain_eval.thin(cand, t, 10, 4)
    assert capped.size == 4
    assert capped[0] == 0
    assert capped[-1] == 90


def test_segment_windows_flagged_only_with_faulty_sensor() -> None:
    n = 120
    split = make_split([MachineState.RUNNING] * n, window=3, horizons=(2,))
    sf_rows = np.zeros(n, dtype=bool)
    sf_rows[10:40] = True  # segment 0: temp, flagged
    sf_rows[60:80] = True  # segment 1: vib, never flagged
    sensors = np.full(n, -1, dtype=np.int8)
    sensors[10:40] = SENSORS.index("temp")
    sensors[60:80] = SENSORS.index("vib")
    ends = split.arrays["ends"]
    flag = (ends >= 20) & (ends < 40)
    segs, total = explain_eval.segment_windows(split, flag, sf_rows, sensors, 5, 30)
    assert total == 2
    assert len(segs) == 1
    s = segs[0]
    assert s.episode == 0
    assert s.fault_type == SENSORS.index("temp")
    assert s.run == 0
    assert np.all((ends[s.windows] >= 20) & (ends[s.windows] <= 39))
    assert np.all(np.diff(split.at_end("t_s")[s.windows]) >= 5)

    # identity expected map: unit agreement = the top mean share is the faulty sensor
    k = s.windows.size
    w = np.zeros((k, len(SENSORS)))
    w[:, SENSORS.index("temp")] = 0.8
    w[:, SENSORS.index("current")] = 0.2
    w[0] = 0.0
    w[0, SENSORS.index("current")] = 1.0  # one window disagrees on its own
    ident = {i: i for i in range(len(SENSORS))}
    unit, _ = explain_eval.agreement_sums(segs, {0: w}, ident, 1, len(SENSORS))
    win = explain_eval.window_agreement_sums(segs, {0: w}, ident, 1, len(SENSORS))
    t = SENSORS.index("temp")
    assert unit[0, t].tolist() == [1.0, 1.0]
    assert win[0, t].tolist() == [float(k), float(k - 1)]


def test_followup_hashes_keep_the_explain_hash() -> None:
    from twin_ml.config import load_model_config
    from twin_ml.experiments.e5_worker import IG_SENSITIVITY_STUDY, explain_config_sha

    e5 = load_e5_config(CONFIG_DIR / "models" / "e5.yaml")
    ev = load_eval_config(CONFIG_DIR / "models" / "eval.yaml")
    cfg = load_model_config(CONFIG_DIR / "models" / "lstm.yaml")
    base = explain_config_sha("lstm", cfg, e5, ev, "k", "src", "c")
    assert base == explain_config_sha("lstm", cfg, e5, ev, "k", "src", "c", IG_SENSITIVITY_STUDY)
    sens = explain_config_sha("lstm", cfg, e5, ev, "k", "src", "c", IG_SENSITIVITY_STUDY, 256)
    assert sens != base
    assert sens != explain_config_sha(
        "lstm", cfg, e5, ev, "k", "src", "c", IG_SENSITIVITY_STUDY, 128
    )

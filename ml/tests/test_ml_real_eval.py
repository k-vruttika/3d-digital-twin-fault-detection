"""Phase 10 rule baselines and event-based evaluation on small hand-built splits."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from twin_common.enums import MachineState
from twin_ml.baselines import rules
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import STATES
from twin_ml.evaluate import run_sums
from twin_ml.experiments import real_eval
from twin_ml.features.schema import ChannelSchema

FAULT_TYPES = ["leak", "none"]
RUN = STATES.index(MachineState.RUNNING)
WARN = STATES.index(MachineState.WARNING)
FAULT = STATES.index(MachineState.FAULT)


def make_split(
    series_rows: list[int],
    machines: list[str],
    *,
    window: int = 2,
    step_s: float = 60.0,
    raw: np.ndarray | None = None,
    state: np.ndarray | None = None,
    fault_type: np.ndarray | None = None,
    sensors: tuple[str, ...] = ("a", "b"),
    t_offsets: list[float] | None = None,
) -> SplitData:
    """Rows of each series on a regular grid; every row ends a window once it has history."""
    n = sum(series_rows)
    series = np.concatenate([np.full(k, i, dtype=np.int32) for i, k in enumerate(series_rows)])
    offs = t_offsets or [0.0] * len(series_rows)
    t = np.concatenate([offs[i] + np.arange(k) * step_s for i, k in enumerate(series_rows)])
    starts = np.concatenate([[0], np.cumsum(series_rows)[:-1]])
    ends = np.concatenate(
        [np.arange(s + window - 1, s + k) for s, k in zip(starts, series_rows, strict=True)]
    ).astype(np.int64)
    r = np.zeros((n, len(sensors)), dtype=np.float32) if raw is None else raw.astype(np.float32)
    mask = ~np.isfinite(r)
    arrays: dict[str, Any] = {
        "raw": r,
        "values": np.nan_to_num(r),
        "mask": mask,
        "kind": np.zeros(n, dtype=np.int8),
        "series": series,
        "t_s": t,
        "state": np.full(n, RUN, dtype=np.int8) if state is None else state.astype(np.int8),
        "health": np.zeros(n, dtype=np.int8),
        "fault_type": (
            np.full(n, FAULT_TYPES.index("none"), dtype=np.int8)
            if fault_type is None
            else fault_type.astype(np.int8)
        ),
        "ends": ends,
        "weight": np.ones(ends.shape[0], dtype=np.float32),
        "series_run": np.array(machines, dtype=str),
        "series_machine": np.array(machines, dtype=str),
    }
    schema = ChannelSchema(sensors=sensors, kinds=("K",), step_s=step_s)
    return SplitData("test", window, (1,), 1, arrays, schema)


# ---------------------------------------------------------------------------- events


def test_warned_lead_and_during() -> None:
    split = make_split([20], ["M1"])
    index = real_eval.SplitIndex.of(split, FAULT_TYPES)
    t_end = split.at_end("t_s")
    ev = real_eval.RealEvent("1", "M1", 15 * 60.0, 17 * 60.0, "leak")
    sus = np.zeros(split.n_windows, dtype=bool)
    sus[(t_end >= 11 * 60) & (t_end <= 12 * 60)] = True  # inside a 5-row lookback [10, 15)
    sus[t_end == 16 * 60] = True  # during the failure
    (res,) = real_eval.event_results(index, sus, [ev], 5, 0.0, 0.5)
    assert res.evaluable
    assert res.warned
    assert not res.clipped
    assert res.lead_s == pytest.approx(4 * 60.0)
    assert res.detected_during
    assert res.delay_s == pytest.approx(60.0)
    assert res.coverage == pytest.approx(1.0)
    # an alarm before the lookback does not count
    sus2 = np.zeros_like(sus)
    sus2[t_end == 9 * 60] = True
    (res2,) = real_eval.event_results(index, sus2, [ev], 5, 0.0, 0.5)
    assert not res2.warned
    assert res2.lead_s is None


def test_not_evaluable_and_clipped() -> None:
    raw = np.zeros((30, 2))
    raw[10:14] = np.nan  # 4 of the 5 lookback rows of event 1 have no data
    split = make_split([30], ["M1"], raw=raw)
    index = real_eval.SplitIndex.of(split, FAULT_TYPES)
    e1 = real_eval.RealEvent("1", "M1", 15 * 60.0, 15 * 60.0, "leak")
    e2 = real_eval.RealEvent("2", "M1", 18 * 60.0, 18 * 60.0, "leak")  # 3 rows after e1
    e3 = real_eval.RealEvent("3", "M1", 2 * 60.0, 2 * 60.0, "leak")  # lookback before split
    sus = np.ones(split.n_windows, dtype=bool)
    r1, r2, r3 = real_eval.event_results(index, sus, [e1, e2, e3], 5, 0.0, 0.5)
    assert r1.coverage == pytest.approx(0.2)
    assert not r1.evaluable
    assert r2.clipped
    assert r2.evaluable is False
    assert r2.lead_s == pytest.approx(2 * 60.0)  # the first window after e1 ends
    assert not r3.inside_split
    assert not r3.evaluable


def test_false_alarms_on_negatives_and_series_edges() -> None:
    state = np.full(20, RUN)
    state[8:10] = WARN
    split = make_split([10, 10], ["M1", "M1"], state=state, t_offsets=[0.0, 3600.0])
    index = real_eval.SplitIndex.of(split, FAULT_TYPES)
    sus = np.zeros(split.n_windows, dtype=bool)
    sus[2:5] = True  # one onset in series 0 (negative windows)
    sus[7:9] = True  # alarm on WARNING windows of series 0: not negative
    sus[9:12] = True  # continues across the series boundary: a new edge in series 1
    fa = real_eval.false_alarm_sums(index, sus, split)
    assert fa.shape == (1, len(real_eval.FA_FIELDS))
    neg = index.negative
    assert fa[0, 0] == 2  # onsets at window 2 and at the start of series 1
    assert fa[0, 3] == neg.sum()
    assert fa[0, 1] == neg.sum() * 60.0
    assert fa[0, 2] == (sus & neg).sum()


def test_metric_fn_and_chance_formula() -> None:
    assert real_eval.chance_probability(1.0, 0.0, 86_400.0) == pytest.approx(1 - math.exp(-1))
    assert real_eval.chance_probability(0.0, 0.25, 3600.0) == pytest.approx(0.25)
    fn = real_eval.metric_fn(6 * 3600.0)
    ev = np.array([4, 4, 2, 2 * 3600.0, 0, 4, 3, 3 * 600.0])
    fa = np.array([10, 20 * 86_400.0, 50, 1000])
    m = fn({"ev": ev, "fa": fa})
    assert m["detection_rate"] == 0.5
    assert m["warned_count"] == 2
    assert m["lead_mean_h"] == pytest.approx(1.0)
    assert m["lead_min"] == pytest.approx(60.0)
    assert m["false_alarms_per_day"] == pytest.approx(0.5)
    assert m["alarm_frac"] == pytest.approx(0.05)
    p = 1 - 0.95 * math.exp(-0.5 * 0.25)
    assert m["chance_p"] == pytest.approx(p)
    assert m["chance_expected"] == pytest.approx(4 * p)
    assert m["during_rate"] == 0.75
    assert m["during_delay_min"] == pytest.approx(10.0)


def test_machine_bootstrap_uses_the_metric_fn() -> None:
    ev = np.array([[2, 2, 1, 3600.0, 0, 0, 0, 0], [2, 2, 2, 7200.0, 0, 0, 0, 0]])
    fa = np.array([[1, 86_400.0, 1, 10], [3, 86_400.0, 2, 10]])
    sums = {"ev": ev, "fa": fa}
    fn = real_eval.metric_fn(3600.0)
    boot = run_sums.bootstrap(sums, 200, 1, 0.95, metric_fn=fn)
    assert boot["detection_rate"]["point"] == pytest.approx(0.75)
    assert 0.5 <= boot["detection_rate"]["lo"] <= boot["detection_rate"]["hi"] <= 1.0
    diff = run_sums.paired(sums, sums, 100, 1, 0.95, ["detection_rate"], metric_fn=fn)
    assert diff["detection_rate"]["point"] == 0
    assert not diff["detection_rate"]["excludes_zero"]


def test_group_lookback_and_months() -> None:
    assert real_eval.group_lookback_s("clean|p99.9|L360", 60.0) == 21_600.0
    meta = {"t0": "2020-04-30 23:00:00"}
    months = real_eval.month_of(meta, np.array([0.0, 7200.0]))
    assert months.tolist() == ["2020-04", "2020-05"]


# ---------------------------------------------------------------------------- rules


def test_window_means_skip_missing_and_threshold_rule() -> None:
    raw = np.array([[1, 10], [3, np.nan], [5, 10], [100, 10], [7, 10]], dtype=float)
    split = make_split([5], ["M1"], raw=raw)
    mean, n = rules.window_means(split)
    assert mean[:, 0].tolist() == [2, 4, 52.5, 53.5]
    assert mean[0, 1] == 10
    assert n[0, 1] == 1
    rule = rules.ThresholdRule(0.0, 100.0).fit(split, np.array([0, 1]))
    assert rule.low[0, 0] == 2
    assert rule.high[0, 0] == 4
    assert rule.alarm(split).tolist() == [False, False, True, True]


def test_zscore_clips_at_series_start_and_needs_history() -> None:
    raw = np.concatenate([np.ones((6, 1)), [[1.0], [9.0]], np.ones((4, 1))])
    raw[:4, 0] = [1.0, 2.0, 1.0, 2.0]
    split = make_split([8, 4], ["M1", "M1"], raw=raw, sensors=("a",))
    z = rules.zscores(split, trailing=4, min_std=1e-6)
    ends = split.arrays["ends"]
    # series 1 (rows 8-11): trailing rows never come from series 0; the windows ending at 9
    # and 10 have fewer than one window (2 rows) of history -> NaN; the one ending at 11 has 2
    assert np.isnan(z[(ends == 9) | (ends == 10)]).all()
    assert z[ends == 11, 0].item() == 0.0
    # first window (rows 0-1) has no trailing rows at all
    assert np.isnan(z[0, 0])
    alarm = rules.zscore_alarm(split, 4, 3.0, 1e-6)
    assert alarm[ends == 7].item()  # window (1, 9) against trailing (2, 1, 2, ... ) jumps
    assert not alarm[ends == 3].item()


def test_persistence_and_error_rule() -> None:
    state = np.full(6, RUN)
    state[4] = FAULT
    raw = np.zeros((6, 3))
    raw[2, 2] = 1  # error2 at row 2
    split = make_split([6], ["M1"], raw=raw, state=state, sensors=("v", "error1", "error2"))
    assert rules.persistence_alarm(split).tolist() == [False, False, False, True, False]
    assert rules.error_alarm(split, "any", ["error1", "error2"]).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert not rules.error_alarm(split, "error1", ["error1", "error2"]).any()
    with pytest.raises(ValueError, match="error-count"):
        rules.error_alarm(make_split([6], ["M1"]), "any", ["error1"])


def test_perturbed_split_is_seeded_and_keeps_labels() -> None:
    from twin_ml.config import PerturbationsConfig
    from twin_ml.data.perturb import parse_condition
    from twin_ml.features.normalize import Normalizer

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(40, 2))
    raw[5, 0] = np.nan
    split = make_split([20, 20], ["M1", "M2"], raw=raw)
    schema = split.schema
    norm = Normalizer.fit([(raw[:20].astype(np.float32), "K")], 0.1, schema=schema)
    cfg = PerturbationsConfig.model_validate(
        {
            "noise": {"levels_pct": [10]},
            "mcar": {"levels_pct": [20]},
            "dropout": {"levels_pct": [5], "duration_s": [1, 2]},
            "stuck": {"levels_pct": [5], "duration_s": [1, 2]},
            "normal_states": ["RUNNING"],
        }
    )
    sigma = np.ones((1, 2), dtype=np.float32)
    cond = parse_condition("noise_10", cfg)
    a = real_eval.perturbed_split(split, norm, cond, cfg, sigma, 808)
    b = real_eval.perturbed_split(split, norm, cond, cfg, sigma, 808)
    assert np.array_equal(a.arrays["raw"], b.arrays["raw"], equal_nan=True)
    assert np.isnan(a.arrays["raw"][5, 0])
    assert not np.allclose(a.arrays["raw"][:5], split.arrays["raw"][:5])
    for key in ("state", "ends", "t_s", "fault_type"):
        assert np.array_equal(a.arrays[key], split.arrays[key])
    m = real_eval.perturbed_split(split, norm, parse_condition("mcar_20", cfg), cfg, sigma, 808)
    assert m.arrays["mask"].mean() > split.arrays["mask"].mean()

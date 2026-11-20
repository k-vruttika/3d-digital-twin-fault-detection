"""Drift detection on the operating-condition inputs (ADR 0031): PSI with a calibrated alert
level, ADWIN dating, the sustain / clear rule, the reference, missing values, kinds without a
reference."""

from __future__ import annotations

from typing import Any

import numpy as np

from twin_ml.serve.drift import (
    DriftSettings,
    MachineDrift,
    build_reference,
    calibrate_thresholds,
    drift_public,
    psi,
    psi_counts,
    window_psis,
)

SETTINGS = DriftSettings(
    window_s=120,
    margin=1.3,  # chosen on validation windows in the real build; here on the synthetic ones
    clear_ratio=0.85,
    sustain_s=30,
    clear_s=60,
    adwin_delta=0.002,
)
SENSORS = ("speed", "temp", "vib", "load", "current")
KINDS = ("CONVEYOR", "CNC", "INSPECTION")
BANDS = (45.0, 62.0)  # a run sits in one (overlapping) load band, as in the simulator


def run_rows(n: int, rng: np.random.Generator, band: float, shift: float = 0.0) -> np.ndarray:
    raw = np.zeros((n, len(SENSORS)))
    raw[:, 0] = rng.normal(8000, 150, n)
    raw[:, 1] = rng.normal(60, 3, n)
    raw[:, 2] = rng.gamma(4, 0.3, n)
    raw[:, 3] = rng.normal(band, 7, n) + shift
    raw[:, 4] = rng.normal(10, 1, n)
    return raw


def training_rows(runs: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = [run_rows(1500, rng, BANDS[i % 2]) for i in range(runs)]
    raw = np.concatenate(parts)
    kind = np.concatenate(
        [np.full(1500, i % 2, dtype=np.int64) for i in range(runs)]
    )  # no INSPECTION
    operating = rng.random(raw.shape[0]) < 0.95
    return raw, kind, np.asarray(operating)


def reference() -> dict[str, Any]:
    raw, kind, operating = training_rows()
    ref = build_reference(raw, kind, operating, SENSORS, KINDS, ["speed", "load"], 10, 0.001)
    rng = np.random.default_rng(99)  # validation runs: unseen, one band each
    val = [(run_rows(1200, rng, BANDS[i % 2]), "CNC") for i in range(20)]
    val += [(run_rows(1200, rng, BANDS[i % 2]), "CONVEYOR") for i in range(20)]
    return calibrate_thresholds(ref, val, SENSORS, SETTINGS.window_s, 10, 0.999)


def stream(
    md: MachineDrift, n: int, seed: int, band: float = 45.0, shift: float = 0.0, t0: float = 0.0
) -> list[dict[str, Any]]:
    raw = run_rows(n, np.random.default_rng(seed), band, shift)
    return [md.update({"speed": r[0], "load": r[3]}, t0 + i) for i, r in enumerate(raw)]


def test_reference_has_decile_bins_and_a_calibrated_alert_level_per_kind_and_channel() -> None:
    ref = reference()
    assert sorted(ref) == ["CNC", "CONVEYOR"]  # no training rows for INSPECTION: no reference
    load = ref["CNC"]["load"]
    assert len(load["ref"]) == len(load["edges"]) + 1 == 12  # 10 quantile bins + 2 overflow bins
    assert load["ref"][0] < 0.005  # almost no training mass outside the support
    assert load["ref"][-1] < 0.005
    assert abs(sum(load["ref"]) - 1.0) < 1e-9
    assert load["n"] > 5000
    assert set(ref["CNC"]) == {"speed", "load"}  # health signals are not monitored for drift
    # a run sits in one band, so a clean window differs from the pooled training distribution:
    # the calibrated level is far above the textbook 0.25 (which would flag every clean run)
    assert load["psi_threshold"] > 0.25
    assert ref["CNC"]["speed"]["psi_threshold"] < load["psi_threshold"]
    assert load["calibration_windows"] > 100
    public = drift_public(
        {
            "channels": ["speed", "load"],
            "bins": 10,
            "window_s": 120,
            "alert_quantile": 0.999,
            "kinds": ref,
        }
    )
    assert public["kinds"] == ["CNC", "CONVEYOR"]
    assert drift_public(None) == {"available": False}


def test_psi_is_zero_for_equal_and_large_for_a_shifted_distribution() -> None:
    p = np.full(10, 0.1)
    assert psi(p, p) == 0.0
    shifted = np.array([0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.2, 0.25, 0.25])
    assert psi(shifted, p) > 0.25
    assert psi(np.array([0.0, 1.0]), np.array([0.5, 0.5])) > 0  # empty bins are floored


def test_window_psis_equal_a_direct_computation() -> None:
    ref = reference()["CNC"]["load"]
    x = run_rows(900, np.random.default_rng(3), 45.0)[:, 3]
    got = window_psis(x, ref, 300, 100)
    edges, q = np.asarray(ref["edges"]), np.asarray(ref["ref"])
    want = [
        psi_counts(
            np.bincount(np.searchsorted(edges, x[e - 300 : e], side="right"), minlength=q.size), q
        )
        for e in (300, 400, 500, 600, 700, 800, 900)
    ]
    np.testing.assert_allclose(got, want)


def test_clean_runs_in_either_band_never_raise_drift() -> None:
    ref = reference()
    for band, seed in ((45.0, 1), (62.0, 2)):
        md = MachineDrift("CNC", ref, SETTINGS)
        reports = stream(md, 3000, seed=seed, band=band)
        assert not any(r["drifted"] for r in reports), band
        assert reports[SETTINGS.window_s - 2]["ready"] is False  # full windows only
        assert reports[SETTINGS.window_s - 1]["ready"] is True


def test_a_load_far_outside_the_training_range_is_sustained_dated_by_adwin_and_clears() -> None:
    md = MachineDrift("CNC", reference(), SETTINGS)
    stream(md, 600, seed=2)  # clean, window full
    shift_at = 600.0
    shifted = stream(md, 400, seed=3, shift=70.0, t0=shift_at)  # load about 115 %: never trained
    first = next(i for i, r in enumerate(shifted) if r["drifted"])
    assert first >= SETTINGS.sustain_rows - 1  # sustained, not on the first high PSI
    report = shifted[first]
    assert report["psi"]["load"] >= report["alert"]["load"]
    assert report["psi"]["speed"] < report["alert"]["speed"]  # only the shifted channel moved
    # ADWIN dates the change: at or just after the shift, not minutes later
    assert report["since"] is not None
    assert shift_at <= report["since"] <= shift_at + 60
    # back to normal: PSI falls below the clear level, then drift clears after clear_s
    back = stream(md, 900, seed=4, t0=1000.0)
    assert back[0]["drifted"]  # still drifted at once (hysteresis)
    assert not back[-1]["drifted"]
    assert next(i for i, r in enumerate(back) if not r["drifted"]) >= SETTINGS.clear_rows


def test_missing_values_are_skipped_not_counted_as_zero() -> None:
    md = MachineDrift("CNC", reference(), SETTINGS)
    raw = run_rows(1500, np.random.default_rng(5), 45.0)
    r: dict[str, Any] = {}
    for i, row in enumerate(raw):
        r = md.update({"speed": float("nan") if i % 3 == 0 else row[0], "load": row[3]}, float(i))
    assert not r["drifted"]
    assert len(md.channels["speed"].bins) <= SETTINGS.window_s


def test_a_kind_without_a_reference_has_no_monitor() -> None:
    assert not MachineDrift("INSPECTION", reference(), SETTINGS).available

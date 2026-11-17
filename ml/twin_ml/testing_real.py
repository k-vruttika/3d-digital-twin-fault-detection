"""Synthetic stand-ins for the Phase 10 public datasets (tests only; CI has no ``data/``).

:func:`real_tree` writes, under a temporary root, small CSV files in the exact formats of
MetroPT-3 and Azure PdM, facts files pointing at them, dataset configs with short time splits,
``real.yaml`` (pre-registered settings unchanged, only the dataset config paths) and
``eval.yaml`` / ``robustness.yaml`` copies. Every config path is relative to that root, as in
the repository. The values are random (seeded); they only exercise the code.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from twin_ml.testing import CONFIG_DIR

METRO_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

# MetroPT-3 stand-in: 3 days at ~10 s; a 30-min gap on day 2 (new series), a 3-min gap on
# day 1 (kept as missing rows), one failure on day 3.
METRO_START = datetime(2020, 2, 1)
METRO_FAILURE = ("2020-02-03 12:00:00", "2020-02-03 14:00:00")
METRO_SPLIT = {
    "train": ["2020-02-01 00:00:00", "2020-02-02 00:00:00"],
    "val": ["2020-02-02 00:00:00", "2020-02-03 00:00:00"],
    "test": ["2020-02-03 00:00:00", "2020-02-04 00:00:00"],
}
# Azure stand-in: 4 machines, hourly from 2015-01-01 06:00 for 60 days.
AZURE_START = datetime(2015, 1, 1, 6)
AZURE_DAYS = 60
AZURE_SPLIT = {
    "train": ["2015-01-01 00:00:00", "2015-02-01 00:00:00"],
    "val": ["2015-02-01 00:00:00", "2015-02-15 00:00:00"],
    "test": ["2015-02-15 00:00:00", "2015-03-03 00:00:00"],
}
AZURE_FAILURES = [  # (machine, time, component)
    (1, "2015-01-10 06:00:00", "comp1"),
    (1, "2015-01-25 06:00:00", "comp2"),
    (2, "2015-01-20 06:00:00", "comp3"),
    (2, "2015-01-20 06:00:00", "comp4"),  # two components at once -> one "multi" event
    (3, "2015-02-05 06:00:00", "comp2"),
    (4, "2015-02-10 03:00:00", "comp1"),
    (1, "2015-02-20 06:00:00", "comp4"),
    (2, "2015-02-25 06:00:00", "comp2"),
    (3, "2015-02-15 12:00:00", "comp3"),  # lookback starts before the test split
    (4, "2015-03-01 06:00:00", "comp1"),
]


def _ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def write_metropt_csv(path: Path, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    times: list[datetime] = []
    t = METRO_START
    end = METRO_START + timedelta(days=3)
    gap1 = (METRO_START + timedelta(hours=10), METRO_START + timedelta(hours=10, minutes=3))
    gap2 = (
        METRO_START + timedelta(days=1, hours=8),
        METRO_START + timedelta(days=1, hours=8, minutes=30),
    )
    while t < end:
        if not (gap1[0] <= t < gap1[1] or gap2[0] <= t < gap2[1]):
            times.append(t)
        t += timedelta(seconds=int(rng.choice([9, 10, 10, 10, 11])))
    n = len(times)
    phase = (np.arange(n) // 30) % 2  # 5-min load / offload cycles
    fail_a, fail_b = (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in METRO_FAILURE)
    in_fail = np.array([fail_a <= x <= fail_b for x in times])
    cur = np.where(phase == 1, 7.0, 4.0) + rng.normal(0, 0.2, n) + 1.5 * in_fail
    press = 9.0 - 0.3 * phase + rng.normal(0, 0.05, n) - 0.8 * in_fail
    cols = {
        "TP2": np.where(phase == 1, press, 0.0),
        "TP3": press,
        "H1": press - 0.2,
        "DV_pressure": np.where(phase == 1, 0.0, 0.5),
        "Reservoirs": press,
        "Oil_temperature": 60 + 5 * phase + rng.normal(0, 0.5, n),
        "Motor_current": cur,
        "COMP": 1.0 - phase,
        "DV_eletric": phase.astype(float),
        "Towers": (np.arange(n) // 500 % 2).astype(float),
        "MPG": 1.0 - phase,
        "LPS": in_fail.astype(float),
        "Pressure_switch": np.ones(n),
        "Oil_level": np.ones(n),
        "Caudal_impulses": np.ones(n),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("," + ",".join(["timestamp", *METRO_COLUMNS]) + "\n")
        for i, x in enumerate(times):
            vals = ",".join(repr(float(cols[c][i])) for c in METRO_COLUMNS)
            f.write(f"{i * 10},{_ts(x)},{vals}\n")
    return n


def write_azure_csvs(base: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base.mkdir(parents=True, exist_ok=True)
    hours = AZURE_DAYS * 24
    with (base / "PdM_telemetry.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write('"datetime","machineID","volt","rotate","pressure","vibration"\n')
        for m in range(1, 5):
            for h in range(hours):
                t = AZURE_START + timedelta(hours=h)
                v = rng.normal([170, 450, 100, 40], [15, 50, 10, 5])
                f.write(f"{_ts(t)},{m}," + ",".join(repr(float(x)) for x in v) + "\n")
    with (base / "PdM_machines.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write('"machineID","model","age"\n')
        for m in range(1, 5):
            f.write(f'{m},"model{1 + m % 2}",{m * 3}\n')
    with (base / "PdM_failures.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write('"datetime","machineID","failure"\n')
        for m, when, c in AZURE_FAILURES:
            f.write(f'{when},{m},"{c}"\n')
    with (base / "PdM_maint.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write('"datetime","machineID","comp"\n')
        f.write('2014-06-01 06:00:00,1,"comp2"\n')
        for m, when, c in AZURE_FAILURES:
            f.write(f'{when},{m},"{c}"\n')
    with (base / "PdM_errors.csv").open("w", encoding="utf-8", newline="\n") as f:
        f.write('"datetime","machineID","errorID"\n')
        for m, when, _ in AZURE_FAILURES:
            e = datetime.strptime(when, "%Y-%m-%d %H:%M:%S") - timedelta(hours=5)
            f.write(f'{_ts(e)},{m},"error{1 + m % 5}"\n')
        f.write('2015-01-05 07:00:00,3,"error5"\n')


def _dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def real_tree(root: Path, *, holdout_fraction: float = 0.25) -> dict[str, Path]:
    """Write the synthetic data and configs under ``root``; returns the config paths
    (relative to ``root``)."""
    data = root / "data" / "external"
    write_metropt_csv(data / "metropt3" / "MetroPT3(AirCompressor).csv")
    write_azure_csvs(data / "azure_pdm")

    metro = _load(CONFIG_DIR / "datasets" / "metropt3.yaml")
    metro["dir"] = "data/external/metropt3"
    metro["failures"] = [
        {
            "id": 1,
            "start": METRO_FAILURE[0],
            "end": METRO_FAILURE[1],
            "type": "air_leak",
            "severity": "high stress",
            "report": None,
        }
    ]
    metro["pdf_check"] = None
    _dump(root / "config" / "datasets" / "metropt3.yaml", metro)
    azure = _load(CONFIG_DIR / "datasets" / "azure_pdm.yaml")
    azure["dir"] = "data/external/azure_pdm"
    _dump(root / "config" / "datasets" / "azure_pdm.yaml", azure)

    paths: dict[str, Path] = {}
    for name, split in (
        ("real_metropt3", METRO_SPLIT),
        ("real_azure", AZURE_SPLIT),
        ("real_azure_err", AZURE_SPLIT),
        ("real_azure_holdout", AZURE_SPLIT),
    ):
        cfg = _load(CONFIG_DIR / "models" / f"{name}.yaml")
        cfg["split"] = split
        cfg["out_dir"] = "data/ml_real"
        if cfg.get("machine_holdout"):
            cfg["machine_holdout"]["fraction"] = holdout_fraction
        rel = Path("config/models") / f"{name}.yaml"
        _dump(root / rel, cfg)
        paths[name] = rel

    real = _load(CONFIG_DIR / "models" / "real.yaml")
    rel = Path("config/models/real.yaml")
    _dump(root / rel, real)
    paths["real"] = rel
    ev = _load(CONFIG_DIR / "models" / "eval.yaml")
    ev["tracking_dir"] = "mlruns"
    ev["report"]["out_dir"] = "results"
    _dump(root / "config" / "models" / "eval.yaml", ev)
    rob = _load(CONFIG_DIR / "models" / "robustness.yaml")
    _dump(root / "config" / "models" / "robustness.yaml", rob)
    return paths

"""Pytest fixtures and helpers for ML tests (registered as a plugin by the root conftest.py).

Not used by production code. Importing this module loads neither torch nor scikit-learn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from twin_common.enums import MachineState
from twin_ml.config import DatasetConfig, EvalConfig, ModelConfig, load_model_config
from twin_ml.data.dataset import Dataset, SplitData
from twin_ml.data.labels import STATE_TO_HEALTH, STATES

REPO_ROOT = Path(__file__).resolve().parents[2]  # ml/twin_ml/testing.py -> repo
CONFIG_DIR = REPO_ROOT / "config"
XGBOOST_REQUIRED_ENV = "LINETWIN_REQUIRE_XGBOOST"  # set to 1 in CI: a broken XGBoost must fail
ML_SIM_RUNS = 6
ML_SIM_DURATION_S = 1500


_XGBOOST_PROBE = """
try:
    import xgboost  # noqa: F401
except Exception as exc:  # XGBoostError (libomp.dylib not loaded), ImportError, OSError
    text = str(exc).strip()
    print(f"{type(exc).__name__}: {text.splitlines()[0] if text else ''}")
"""


@cache
def xgboost_import_error() -> str | None:
    """None when xgboost imports, else why not (on macOS usually the missing OpenMP runtime).

    Probed in a subprocess: the test process also loads torch, and torch and XGBoost must
    never share a process (two OpenMP runtimes crash on macOS; ADR 0020).
    """
    out = subprocess.run(
        [sys.executable, "-c", _XGBOOST_PROBE], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return f"import probe exited with code {out.returncode}: {out.stderr.strip()[-500:]}"
    return out.stdout.strip() or None


def require_xgboost() -> None:
    """Skip the calling test, with a clear message, when xgboost cannot be imported.

    With LINETWIN_REQUIRE_XGBOOST=1 (CI) the test fails instead, so a broken install is never
    hidden by a skip.
    """
    reason = xgboost_import_error()
    if reason is None:
        return
    msg = (
        f"xgboost cannot be imported ({reason}). On macOS XGBoost needs the OpenMP runtime: "
        "run `brew install libomp` (see docs/setup.md)."
    )
    if os.environ.get(XGBOOST_REQUIRED_ENV) == "1":
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg)


def dataset_config(source: Path, out: Path, **overrides: Any) -> DatasetConfig:
    data: dict[str, Any] = {
        "source_dir": source,
        "out_dir": out,
        "line_config": CONFIG_DIR / "line.yaml",
        "window_s": 30,
        "horizons_s": [10, 30],
        "ttf_cap_s": 600,
        "strides_s": {"train": 5, "val": 5, "test": 2},
        "split": {"fractions": {"train": 0.5, "val": 0.25, "test": 0.25}, "seed": 3},
        "normal_keep_frac": 0.5,
        "subsample_seed": 7,
        "features": {"min_speed_ratio": 0.1},
    }
    data.update(overrides)
    return DatasetConfig.model_validate(data)


def eval_config(tracking_dir: Path, out_dir: Path) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "experiment": "E1_test",
            "tracking_dir": tracking_dir,
            "seeds": [0],
            "models": ["persistence", "threshold", "rf", "xgb", "lstm", "gru", "transformer"],
            "subsets": {"operating": ["RUNNING", "WARNING"], "stopped": ["FAULT", "MAINTENANCE"]},
            "early_warning": {"sustain_s": 3},
            "latency": {"warmup_calls": 2, "timed_calls": 10, "num_threads": 1},
            "bootstrap": {"n_resamples": 50, "seed": 1, "ci": 0.95},
            "report": {"out_dir": out_dir, "name": "E1_test"},
        }
    )


def generate_sim(
    out: Path,
    runs: int = ML_SIM_RUNS,
    duration_s: int = ML_SIM_DURATION_S,
    sensor_faults: dict[str, Any] | None = None,
    faults: dict[str, Any] | None = None,
) -> Path:
    """Fault-dense quick-scenario runs (Parquet + manifest) for ML tests.

    ``sensor_faults``: a scenario ``sensor_faults`` section (Phase 8 sensor-fault sets); the
    machine physics stay identical for the same seed. ``faults``: a replacement ``faults``
    section (Phase 9 E5 runs with only the held-out fault type).
    """
    from twin_common.config import load_line_config
    from twin_common.config.loader import read_yaml
    from twin_sim.batch import generate
    from twin_sim.config import ScenarioConfig, load_sim_config

    data = read_yaml(CONFIG_DIR / "scenarios" / "quick.yaml")
    data["duration_s"] = duration_s
    if sensor_faults is not None:
        data["sensor_faults"] = sensor_faults
    if faults is not None:
        data["faults"] = faults
    scenario = ScenarioConfig.model_validate(data)
    generate(
        line=load_line_config(CONFIG_DIR / "line.yaml"),
        sim=load_sim_config(CONFIG_DIR / "simulator.yaml"),
        scenario=scenario,
        runs=runs,
        base_seed=42,
        out_dir=out,
        workers=1,
    )
    return out


@pytest.fixture(scope="session")
def ml_sim_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_sim(tmp_path_factory.mktemp("ml_sim"))


# Phase 9: E5-like quick runs with only the held-out fault type (same seed as ml_sim_dir).
E5_FAULTS: dict[str, Any] = {
    "random": {
        "rate_per_machine_per_hour": 3.0,
        "types": ["lubrication_loss"],
        "include_held_out": True,
        "earliest_onset_s": 60,
    }
}


@pytest.fixture(scope="session")
def e5_sim_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_sim(tmp_path_factory.mktemp("ml_sim_e5"), faults=E5_FAULTS)


@pytest.fixture(scope="session")
def ml_dataset_cfg(ml_sim_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> DatasetConfig:
    return dataset_config(ml_sim_dir, tmp_path_factory.mktemp("ml_data"))


@pytest.fixture(scope="session")
def ml_dataset(ml_dataset_cfg: DatasetConfig) -> Dataset:
    from twin_ml.data.build import build_dataset

    return Dataset.open(build_dataset(ml_dataset_cfg))


@pytest.fixture
def ml_eval_cfg(tmp_path: Path) -> EvalConfig:
    return eval_config(tmp_path / "mlruns", tmp_path / "results")


def make_split(
    states: list[MachineState],
    *,
    values: npt.NDArray[np.float32] | None = None,
    window: int = 3,
    horizons: tuple[int, ...] = (1, 2),
    stride: int = 1,
    kind: int = 0,
    fault_type: list[int] | None = None,
    ttf: list[float] | None = None,
    censored: list[bool] | None = None,
) -> SplitData:
    """A one-series split built by hand (window ends at every valid row) for unit tests."""
    n = len(states)
    state = np.array([STATES.index(s) for s in states], dtype=np.int8)
    vals = np.zeros((n, 6), dtype=np.float32) if values is None else values.astype(np.float32)
    ends = np.arange(window - 1, n - max(horizons), stride, dtype=np.int64)
    arrays: dict[str, npt.NDArray[Any]] = {
        "raw": vals[:, :5].copy(),
        "values": vals,
        "mask": np.zeros((n, 5), dtype=np.uint8),
        "kind": np.full(n, kind, dtype=np.int8),
        "series": np.zeros(n, dtype=np.int32),
        "t_s": np.arange(1, n + 1, dtype=np.float32),
        "state": state,
        "health": STATE_TO_HEALTH[state],
        "fault_type": np.array(fault_type or [0] * n, dtype=np.int8),
        "ttf": np.array(ttf or [600.0] * n, dtype=np.float32),
        "ttf_censored": np.array(censored or [False] * n, dtype=bool),
        "episode_type": np.zeros(n, dtype=np.int8),
        "ends": ends,
        "weight": np.ones(ends.shape[0], dtype=np.float32),
        "series_run": np.array(["run_0000"]),
        "series_machine": np.array(["CNV-01"]),
        "ep_series": np.zeros(0, dtype=np.int32),
        "ep_fault_type": np.zeros(0, dtype=np.int8),
        "ep_start": np.zeros(0, dtype=np.int64),
        "ep_warn": np.zeros(0, dtype=np.int64),
        "ep_fault": np.zeros(0, dtype=np.int64),
    }
    return SplitData(name="test", window=window, horizons=horizons, stride=stride, arrays=arrays)


# Tiny model configs for end-to-end tests (seconds per model on quick data).
TINY: dict[str, dict[str, Any]] = {
    "persistence": {"type": "persistence"},
    "threshold": {"type": "threshold", "sim_config": str(CONFIG_DIR / "simulator.yaml")},
    "rf": {
        "type": "rf",
        "class_weight": {"power": 0.5, "max": 50.0},
        "classifier": {"n_estimators": 8, "max_depth": 6, "min_samples_leaf": 2},
        "regressor": {"n_estimators": 8, "max_depth": 6, "min_samples_leaf": 2},
        "n_jobs": 2,
    },
    "xgb": {
        "type": "xgb",
        "class_weight": {"power": 0.5, "max": 50.0},
        "classifier": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
        },
        "regressor": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
        },
        "early_stopping_rounds": 5,
        "n_jobs": 2,
    },
}

NN_TRAINING: dict[str, Any] = {
    "early_stopping": {
        "min_epochs": 1,
        "max_epochs": 3,
        "patience": 1,
        "ema_alpha": 0.5,
        "min_delta": 0.0,
    },
    "lr_schedule": {"type": "plateau", "factor": 0.5, "patience": 0, "min_lr": 1e-5},
    "batch_size": 128,
    "lr": 0.003,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "focal_gamma": 0.0,
    "huber_delta": 0.1,
    "loss_weights": {"cls": 1.0, "ttf": 0.5, "fault_type": 0.3},
    "eval_batch_size": 1024,
    "num_threads": 2,
}
for _kind in ("lstm", "gru", "transformer"):
    TINY[_kind] = {
        "type": _kind,
        "class_weight": {"power": 0.5, "max": 50.0},
        "arch": {"hidden": 16, "layers": 2, "dropout": 0.1, "heads": 2, "ff_mult": 2},
        "training": NN_TRAINING,
    }


def tiny_config(model: str, tmp_path: Path) -> ModelConfig:
    """Write ``TINY[model]`` to ``<tmp_path>/<model>.yaml`` and load it."""
    path = tmp_path / f"{model}.yaml"
    path.write_text(json.dumps(TINY[model]), encoding="utf-8")  # JSON is valid YAML
    return load_model_config(path)

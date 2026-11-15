"""Pytest fixtures and helpers for ML tests (registered as a plugin by the root conftest.py).

Not used by production code. Importing this module loads neither torch nor scikit-learn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from twin_common.enums import MachineState
from twin_ml.config import DatasetConfig, EvalConfig
from twin_ml.data.dataset import Dataset, SplitData
from twin_ml.data.labels import STATE_TO_HEALTH, STATES

REPO_ROOT = Path(__file__).resolve().parents[2]  # ml/twin_ml/testing.py -> repo
CONFIG_DIR = REPO_ROOT / "config"
ML_SIM_RUNS = 6
ML_SIM_DURATION_S = 1500


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


def generate_sim(out: Path, runs: int = ML_SIM_RUNS, duration_s: int = ML_SIM_DURATION_S) -> Path:
    """Fault-dense quick-scenario runs (Parquet + manifest) for ML tests."""
    from twin_common.config import load_line_config
    from twin_sim.batch import generate
    from twin_sim.config import load_scenario, load_sim_config

    scenario = load_scenario(CONFIG_DIR / "scenarios" / "quick.yaml").model_copy(
        update={"duration_s": duration_s}
    )
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

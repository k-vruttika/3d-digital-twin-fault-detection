"""Phase 9 E5 test data (ADR 0024): run selection, vocabulary extension, shared plans."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from twin_ml.config import DatasetConfig, E5DataConfig, load_e5_config
from twin_ml.data.dataset import Dataset
from twin_ml.data.labels import EPISODE_GRADUAL
from twin_ml.data.load import read_manifest
from twin_ml.data.test_sets import (
    E5DataError,
    check_shared_plan,
    e5_run_ids,
    e5_split,
    extended_fault_types,
)
from twin_ml.testing import CONFIG_DIR, ML_SIM_RUNS


def e5_data_cfg(source: Path, test_ids: list[str], extra: tuple[int, int] = (6, 6)) -> E5DataConfig:
    return E5DataConfig(
        source_dir=source,
        scenario=Path("config/scenarios/quick.yaml"),
        base_seed=42,
        generate_runs=ML_SIM_RUNS,
        test_split_runs=True,
        extra_run_indices=extra,
        fault_type="lubrication_loss",
        n_runs=len(set(test_ids) | {f"run_{i:04d}" for i in range(*extra)}),
    )


def test_real_selection_is_test_split_plus_200_to_269() -> None:
    e5 = load_e5_config(CONFIG_DIR / "models" / "e5.yaml")
    test = [f"run_{i:04d}" for i in range(0, 200, 6)][:30]
    ids = e5_run_ids(e5.e5_data, test)
    assert len(ids) == 100
    assert set(test) <= set(ids)
    assert set(ids) - set(test) == {f"run_{i:04d}" for i in range(200, 270)}
    with pytest.raises(E5DataError, match="fixes 100"):
        e5_run_ids(e5.e5_data, test[:29])


def test_extended_vocab_keeps_trained_codes(ml_dataset: Dataset) -> None:
    ft = extended_fault_types(ml_dataset, ["lubrication_loss"])
    assert ft[: len(ml_dataset.fault_types)] == list(ml_dataset.fault_types)
    assert ft[-1] == "lubrication_loss"
    assert extended_fault_types(ml_dataset, ["none"]) == list(ml_dataset.fault_types)


def test_e5_split_episodes_are_gradual_held_out(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, e5_sim_dir: Path
) -> None:
    test_ids = ml_dataset.meta["splits"]["test"]
    cfg = e5_data_cfg(e5_sim_dir, test_ids)
    ids = e5_run_ids(cfg, test_ids)
    shared = check_shared_plan(
        read_manifest(ml_dataset_cfg.source_dir), read_manifest(e5_sim_dir), ids
    )
    assert shared == len(ids)
    split = e5_split(ml_dataset, cfg, e5_sim_dir, ids, ml_dataset_cfg.line_config)
    a = split.arrays
    code = len(ml_dataset.fault_types)  # appended
    assert len(a["ep_series"]) > 0
    assert (a["ep_fault_type"] == code).all()
    assert (a["episode_type"][a["ep_fault"] - 1] == EPISODE_GRADUAL).all()
    assert set(np.unique(a["fault_type"]).tolist()) <= {
        list(ml_dataset.fault_types).index("none"),
        code,
    }
    assert split.run_ids == sorted(ids)


def test_shared_plan_mismatch_and_wrong_fault_types(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, ml_sim_dir: Path, e5_sim_dir: Path
) -> None:
    test_ids = ml_dataset.meta["splits"]["test"]
    clean, e5 = read_manifest(ml_sim_dir), read_manifest(e5_sim_dir)
    bad = copy.deepcopy(e5)
    bad["run_entries"][0]["ambient_c"] += 1.0
    with pytest.raises(E5DataError, match="ambient_c"):
        check_shared_plan(clean, bad, [bad["run_entries"][0]["run_id"]])
    cfg = e5_data_cfg(ml_sim_dir, test_ids)  # the clean runs: seen fault types
    with pytest.raises(E5DataError, match="other fault types"):
        e5_split(ml_dataset, cfg, ml_sim_dir, test_ids, ml_dataset_cfg.line_config)
    wrong_seed = cfg.model_copy(update={"base_seed": 7})
    with pytest.raises(E5DataError, match="seed"):
        e5_split(ml_dataset, wrong_seed, e5_sim_dir, test_ids, ml_dataset_cfg.line_config)
    missing = cfg.model_copy(update={"source_dir": e5_sim_dir / "nope"})
    with pytest.raises(E5DataError, match="sim-e5"):
        e5_split(ml_dataset, missing, missing.source_dir, test_ids, ml_dataset_cfg.line_config)

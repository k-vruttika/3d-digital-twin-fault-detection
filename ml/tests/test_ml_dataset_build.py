"""Built dataset: split leakage, train-only normalisation, input restrictions, sample weights."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from twin_ml.config import DatasetConfig
from twin_ml.data.build import build_dataset
from twin_ml.data.dataset import MANIFEST_FILE, Dataset, DatasetNotBuiltError, resolve_dataset_dir
from twin_ml.data.splits import split_runs
from twin_ml.features.inputs import (
    ALLOWED_INPUT_COLUMNS,
    SEQUENCE_CHANNELS,
    ForbiddenInputError,
    check_input_names,
    select_inputs,
)
from twin_ml.features.normalize import Normalizer
from twin_ml.features.summary import summary_names
from twin_ml.testing import dataset_config
from twin_sim.batch import MANIFEST_NAME

# Every simulator column that must never reach a model input.
FORBIDDEN = (
    "state",
    "run_time",
    "seq",
    "quality",
    "fault_type",
    "time_to_fault_s",
    "ttf_censored",
    "root_cause_machine",
    "load_range",
    "sensor_fault",
    "buffer_in",
    "buffer_out",
)


def test_manifest_name_constant_matches_simulator() -> None:
    assert MANIFEST_FILE == MANIFEST_NAME


def test_split_runs_disjoint_complete_deterministic() -> None:
    ids = [f"run_{i:04d}" for i in range(200)]
    a = split_runs(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=2026)
    b = split_runs(list(reversed(ids)), {"train": 0.7, "val": 0.15, "test": 0.15}, seed=2026)
    assert a == b
    assert [len(a[s]) for s in ("train", "val", "test")] == [140, 30, 30]
    assert set(a["train"]).isdisjoint(a["val"])
    assert set(a["train"]).isdisjoint(a["test"])
    assert set(a["val"]).isdisjoint(a["test"])
    assert sorted(a["train"] + a["val"] + a["test"]) == ids
    other = split_runs(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert other["test"] != a["test"]
    with pytest.raises(ValueError, match="at least"):
        split_runs(["a", "b"], {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)


def test_no_run_in_two_splits(ml_dataset: Dataset) -> None:
    runs = {
        s: set(ml_dataset.split(s).arrays["series_run"].tolist()) for s in ("train", "val", "test")
    }
    assert runs["train"].isdisjoint(runs["val"])
    assert runs["train"].isdisjoint(runs["test"])
    assert runs["val"].isdisjoint(runs["test"])
    for s in ("train", "val", "test"):
        assert runs[s] == set(ml_dataset.meta["splits"][s])
        split = ml_dataset.split(s)
        # every window lies inside one series: first and last row share the series id
        ends = split.arrays["ends"]
        assert (
            split.arrays["series"][ends] == split.arrays["series"][ends - split.window + 1]
        ).all()


def test_normalizer_fitted_on_training_rows_only(ml_dataset: Dataset) -> None:
    train = ml_dataset.split("train")
    norm = Normalizer.from_dict(ml_dataset.meta["normalizer"])
    from twin_ml.features.inputs import KINDS

    series = []
    for sid in range(train.arrays["series_run"].shape[0]):
        rows = train.arrays["series"] == sid
        series.append((train.arrays["raw"][rows], KINDS[int(train.arrays["kind"][rows][0])]))
    refit = Normalizer.fit(series, norm.min_speed_ratio)
    np.testing.assert_allclose(refit.mean, norm.mean, rtol=1e-5)
    np.testing.assert_allclose(refit.std, norm.std, rtol=1e-5)
    assert refit.speed_ref == pytest.approx(norm.speed_ref)


def _rewrite_runs(src: Path, dst: Path, runs: set[str] | None, fn: object) -> None:
    shutil.copytree(src, dst)
    manifest = json.loads((dst / MANIFEST_FILE).read_text(encoding="utf-8"))
    for entry in manifest["run_entries"]:
        if runs is not None and entry["run_id"] not in runs:
            continue
        path = dst / entry["files"]["parquet"]
        table = pq.read_table(path)
        pq.write_table(fn(table), path)  # type: ignore[operator]


def test_val_test_data_do_not_change_the_normalizer(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, tmp_path: Path
) -> None:
    held_out = set(ml_dataset.meta["splits"]["val"]) | set(ml_dataset.meta["splits"]["test"])

    def scale(table: pa.Table) -> pa.Table:
        for col in ("temp", "vib"):
            i = table.schema.get_field_index(col)
            table = table.set_column(i, col, pc.multiply(table.column(col), 3.0))
        return table

    _rewrite_runs(ml_dataset_cfg.source_dir, tmp_path / "src", held_out, scale)
    cfg = dataset_config(tmp_path / "src", tmp_path / "out")
    other = Dataset.open(build_dataset(cfg, check_hashes=False))
    assert other.meta["normalizer"] == ml_dataset.meta["normalizer"]
    assert other.meta["splits"] == ml_dataset.meta["splits"]


def test_forbidden_columns_never_reach_model_inputs(
    ml_dataset_cfg: DatasetConfig, tmp_path: Path
) -> None:
    """Scrambling state, labels, run_time, seq, ... leaves every model input bit-identical."""
    rng = np.random.default_rng(0)

    def scramble(table: pa.Table) -> pa.Table:
        for col in FORBIDDEN:
            i = table.schema.get_field_index(col)
            perm = pa.array(rng.permutation(table.num_rows))
            table = table.set_column(i, table.schema.field(i), table.column(col).take(perm))
        return table

    _rewrite_runs(ml_dataset_cfg.source_dir, tmp_path / "src", None, scramble)
    # no subsampling: window selection must not depend on labels for this comparison
    base = Dataset.open(
        build_dataset(
            dataset_config(ml_dataset_cfg.source_dir, tmp_path / "a", normal_keep_frac=1.0)
        )
    )
    other = Dataset.open(
        build_dataset(
            dataset_config(tmp_path / "src", tmp_path / "b", normal_keep_frac=1.0),
            check_hashes=False,
        )
    )
    assert other.meta["normalizer"] == base.meta["normalizer"]
    for s in ("train", "val", "test"):
        a, b = base.split(s), other.split(s)
        for name in ("raw", "values", "mask", "kind", "ends"):
            np.testing.assert_array_equal(a.arrays[name], b.arrays[name], err_msg=f"{s}.{name}")
        idx = np.arange(min(a.n_windows, 500))
        np.testing.assert_array_equal(a.summary(idx), b.summary(idx))
        np.testing.assert_array_equal(a.sequence(idx), b.sequence(idx))
        # the labels did change, so the test really scrambled something
        if s == "train":
            assert not np.array_equal(a.arrays["state"], b.arrays["state"])


def test_input_names_are_restricted() -> None:
    assert set(ALLOWED_INPUT_COLUMNS) == {"speed", "temp", "vib", "load", "current"}
    allowed_roots = {*ALLOWED_INPUT_COLUMNS, "vib_norm", "kind"}
    for name in [*summary_names(), *SEQUENCE_CHANNELS]:
        assert name.split("_")[0] in allowed_roots or name.startswith("vib_norm"), name
    for bad in ("state", "run_time", "time_to_fault_s", "fault_type", "seq"):
        with pytest.raises(ForbiddenInputError):
            check_input_names([bad])
    # extra columns are ignored, never read; missing sensors are an error
    cols = {c: [1.0, 2.0] for c in ALLOWED_INPUT_COLUMNS} | {"state": ["FAULT", "FAULT"]}
    assert select_inputs(cols).shape == (2, 5)
    with pytest.raises(ForbiddenInputError, match="missing"):
        select_inputs({"speed": [1.0]})


def test_weighted_training_prior_matches_unsubsampled(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, tmp_path: Path
) -> None:
    full = Dataset.open(
        build_dataset(dataset_config(ml_dataset_cfg.source_dir, tmp_path, normal_keep_frac=1.0))
    )
    sub, ref = ml_dataset.split("train"), full.split("train")
    assert sub.n_windows < ref.n_windows
    for h in sub.horizons:
        w = np.bincount(sub.target(h), weights=sub.weight, minlength=3)
        r = np.bincount(ref.target(h), minlength=3)
        np.testing.assert_allclose(w / w.sum(), r / r.sum(), atol=0.02)


def test_resolve_dataset_dir(
    ml_dataset: Dataset, ml_dataset_cfg: DatasetConfig, tmp_path: Path
) -> None:
    assert resolve_dataset_dir(ml_dataset_cfg) == ml_dataset.path
    with pytest.raises(DatasetNotBuiltError, match="poe dataset"):
        resolve_dataset_dir(dataset_config(ml_dataset_cfg.source_dir, tmp_path))
    with pytest.raises(DatasetNotBuiltError, match="manifest"):
        resolve_dataset_dir(dataset_config(tmp_path / "nothing", tmp_path))
    meta = ml_dataset.meta
    assert len(meta["source"]["manifest_sha256"]) == 64
    assert len(meta["source"]["content_sha256"]) == 64
    assert "lubrication_loss" not in meta["fault_types"]

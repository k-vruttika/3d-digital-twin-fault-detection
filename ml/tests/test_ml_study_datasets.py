"""Phase 8 study datasets: E4 run filter (load bands) and training augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from twin_common.enums import MachineKind
from twin_ml.config import (
    AugmentationConfig,
    DatasetConfig,
    LoadRangeFilter,
    RunFilterConfig,
    load_dataset_config,
    load_robustness_config,
)
from twin_ml.data.build import augment_series, build_dataset, select_runs
from twin_ml.data.dataset import Dataset, dataset_key
from twin_ml.data.load import MachineSeries
from twin_ml.testing import CONFIG_DIR, dataset_config

E1_KEY_CONFIG = CONFIG_DIR / "models" / "dataset.yaml"


def test_new_optional_fields_do_not_change_the_e1_fingerprint() -> None:
    cfg = load_dataset_config(E1_KEY_CONFIG)
    data = cfg.model_dump(mode="json")
    legacy = {k: v for k, v in data.items() if k not in ("run_filter", "augmentation")}
    # the fingerprint of the Phase 7 config (before the fields existed) is unchanged
    import hashlib
    import json

    paths = {"source_dir", "out_dir", "line_config"}
    expected = hashlib.sha256(
        json.dumps({k: v for k, v in legacy.items() if k not in paths}, sort_keys=True).encode()
    ).hexdigest()
    assert cfg.fingerprint() == expected
    e4 = load_dataset_config(CONFIG_DIR / "models" / "dataset_e4.yaml")
    aug = load_dataset_config(CONFIG_DIR / "models" / "dataset_aug.yaml")
    keys = {dataset_key(c, "x") for c in (cfg, e4, aug)}
    assert len(keys) == 3


def test_select_runs_e4_rule() -> None:
    splits = {
        "train": ["r1", "r2", "r3"],
        "val": ["r4", "r5"],
        "test": ["r6", "r7", "r8"],
    }
    bands = {
        "r1": "low",
        "r2": "high",
        "r3": "medium",
        "r4": "high",
        "r5": "low",
        "r6": "high",
        "r7": "medium",
        "r8": "low",
    }
    lm = LoadRangeFilter(load_ranges=["low", "medium"])
    flt = RunFilterConfig(train=lm, val=lm, test=LoadRangeFilter(load_ranges=["high"]), test_in=lm)
    out = select_runs(splits, bands, flt)
    assert out["train"] == ["r1", "r3"]
    assert out["val"] == ["r5"]
    assert out["test"] == ["r2", "r4", "r6"]  # every high run: none is trained or validated on
    assert out["test_in"] == ["r7", "r8"]
    seen = set(out["train"]) | set(out["val"])
    assert seen.isdisjoint(out["test"])
    assert seen.isdisjoint(out["test_in"])
    assert select_runs(splits, bands, None) == splits


def _series(n: int) -> list[MachineSeries]:
    rng = np.random.default_rng(0)
    return [
        MachineSeries(
            run_id=f"run_{i:04d}",
            machine_id="CNC-01",
            kind=MachineKind.CNC,
            t_s=np.arange(600, dtype=np.float64),
            sensors=rng.normal(50, 5, (600, 5)).astype(np.float32),
            state=["RUNNING"] * 600,
            fault_type=["none"] * 600,
        )
        for i in range(n)
    ]


def test_augment_series_seeded_and_labels_untouched() -> None:
    rcfg = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")
    aug = AugmentationConfig(p=0.5, seed=3, perturbations=rcfg.perturbations)
    sigma = np.ones((3, 5), dtype=np.float32)
    series = _series(200)
    a, tally = augment_series(series, aug, sigma)
    b, _ = augment_series(series, aug, sigma)
    for x, y, s in zip(a, b, series, strict=True):
        assert np.array_equal(x.sensors, y.sensors, equal_nan=True)
        assert x.state == s.state
        assert x.fault_type == s.fault_type
    assert sum(tally.values()) == 200
    assert 70 < tally["clean"] < 130  # p = 0.5
    assert all(tally[k] > 0 for k in ("noise", "mcar", "dropout", "stuck"))
    c, _ = augment_series(series, aug.model_copy(update={"seed": 4}), sigma)
    assert any(
        not np.array_equal(x.sensors, z.sensors, equal_nan=True) for x, z in zip(a, c, strict=True)
    )


def test_built_e4_and_augmented_datasets(ml_sim_dir: Path, tmp_path: Path) -> None:
    """Quick data: the filter gives disjoint run sets; augmentation changes train only."""
    base = dataset_config(ml_sim_dir, tmp_path)
    clean = Dataset.open(build_dataset(base))
    import json

    manifest = json.loads((ml_sim_dir / "manifest.json").read_text(encoding="utf-8"))
    bands = sorted({e["load_range"] for e in manifest["run_entries"]})
    keep, held = bands[:-1], bands[-1:]
    lm = LoadRangeFilter(load_ranges=keep)
    flt = RunFilterConfig(train=lm, val=lm, test=LoadRangeFilter(load_ranges=held), test_in=lm)
    e4 = Dataset.open(build_dataset(base.model_copy(update={"run_filter": flt})))
    sp = e4.meta["splits"]
    band_of = {e["run_id"]: e["load_range"] for e in manifest["run_entries"]}
    assert all(band_of[r] in keep for r in sp["train"] + sp["val"] + sp["test_in"])
    assert all(band_of[r] in held for r in sp["test"])
    assert e4.meta["extra_splits"] == ["test_in"]
    assert e4.split("test_in").stride == e4.split("test").stride
    assert set(e4.meta["run_filter"]) == {"train", "val", "test", "test_in"}

    rcfg = load_robustness_config(CONFIG_DIR / "models" / "robustness.yaml")
    aug_cfg: DatasetConfig = base.model_copy(
        update={"augmentation": AugmentationConfig(p=0.5, seed=1, perturbations=rcfg.perturbations)}
    )
    aug = Dataset.open(build_dataset(aug_cfg))
    assert aug.meta["dataset_key"] != clean.meta["dataset_key"]
    assert aug.meta["normalizer"] == clean.meta["normalizer"]  # fitted on clean training data
    for name in ("val", "test"):
        assert np.array_equal(aug.split(name).arrays["values"], clean.split(name).arrays["values"])
    tr_a, tr_c = aug.split("train").arrays, clean.split("train").arrays
    assert np.array_equal(tr_a["ends"], tr_c["ends"])  # same windows and weights
    assert np.array_equal(tr_a["weight"], tr_c["weight"])
    assert np.array_equal(tr_a["health"], tr_c["health"])
    assert not np.array_equal(tr_a["raw"], tr_c["raw"], equal_nan=True)
    assert sum(aug.meta["augmentation"]["series"].values()) == len(tr_a["series_run"])

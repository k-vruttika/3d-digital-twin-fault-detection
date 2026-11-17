"""Phase 8 test sets rebuilt from raw series (imports twin_sim through ``data.load``).

* perturbed copies of a dataset's test split: the raw test series are perturbed
  (``data.perturb``) and windowed again with the dataset's *clean* normaliser, fault-type
  vocabulary and config, so forward fill and missing masks are recomputed as in live use;
* simulator sensor-fault sets: the same test runs regenerated with sensor faults (same base
  seed). Their state and fault-type labels must equal the clean runs' exactly, otherwise the
  build stops (the sensor-fault RNG stream must not change the physics);
* the Phase 9 E5 set (ADR 0024): runs with the held-out fault type only. Its label vocabulary
  is the dataset's plus the held-out type, appended, so the trained codes are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.config import load_line_config
from twin_ml.config import DatasetConfig, E5DataConfig, PerturbationsConfig
from twin_ml.data.build import build_split
from twin_ml.data.dataset import Dataset, SplitData
from twin_ml.data.load import MachineSeries, iter_series, read_manifest, verify_files
from twin_ml.data.perturb import Condition, perturb_series
from twin_ml.features.normalize import Normalizer
from twin_sim.config import NO_FAULT_LABEL

SENSOR_FAULT_COLUMN = "sensor_fault"


class SensorFaultDataError(ValueError):
    """The sensor-fault runs are missing or do not match the clean runs."""


def dataset_config_of(ds: Dataset) -> DatasetConfig:
    return DatasetConfig.model_validate(ds.meta["config"])


def load_series(
    source_dir: Path,
    run_ids: Sequence[str],
    line_config: Path,
    *,
    check_hashes: bool = True,
    extra_labels: Sequence[str] = (),
) -> list[MachineSeries]:
    manifest = read_manifest(source_dir)
    if check_hashes:
        verify_files(source_dir, manifest)
    known = {e["run_id"] for e in manifest["run_entries"]}
    missing = sorted(set(run_ids) - known)
    if missing:
        raise SensorFaultDataError(
            f"{source_dir} lacks runs {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    line = load_line_config(line_config)
    return list(iter_series(source_dir, manifest, run_ids, line, extra_labels))


def extended_fault_types(ds: Dataset, extra: Sequence[str] = ()) -> list[str]:
    """The dataset's fault-type vocabulary with ``extra`` types appended (codes unchanged)."""
    names = list(ds.fault_types)
    return names + [f for f in extra if f not in names]


def windowed(
    ds: Dataset,
    series: Sequence[MachineSeries],
    split_name: str = "test",
    extra_fault_types: Sequence[str] = (),
) -> SplitData:
    """Window ``series`` exactly like ``ds``'s split ``split_name`` (clean normaliser, vocab).

    ``extra_fault_types`` are appended to the label vocabulary (fault types the dataset never
    saw, e.g. the held-out E5 type); models' fault-type heads keep their trained classes.
    """
    cfg = dataset_config_of(ds)
    norm = Normalizer.from_dict(ds.meta["normalizer"])
    vocab = {f: i for i, f in enumerate(extended_fault_types(ds, extra_fault_types))}
    stride_split = split_name if split_name in ("train", "val", "test") else "test"
    arrays = build_split(stride_split, series, norm, cfg, vocab, np.random.default_rng(0))
    return SplitData(
        name=split_name,
        window=cfg.window_s,
        horizons=tuple(cfg.horizons_s),
        stride=cfg.strides_s.of(stride_split),
        arrays=arrays,
    )


def perturbed_split(
    ds: Dataset,
    clean_series: Sequence[MachineSeries],
    condition: Condition,
    cfg: PerturbationsConfig,
    sigma: npt.NDArray[np.floating[Any]],
    seed: int,
) -> SplitData:
    return windowed(ds, perturb_series(clean_series, condition, cfg, sigma, seed))


def check_same_labels(clean: Sequence[MachineSeries], other: Sequence[MachineSeries]) -> None:
    """Same runs, machines, timestamps, states and fault types, in the same order."""
    if len(clean) != len(other):
        raise SensorFaultDataError("sensor-fault runs have a different number of series")
    for a, b in zip(clean, other, strict=True):
        if (a.run_id, a.machine_id) != (b.run_id, b.machine_id):
            raise SensorFaultDataError(f"series order differs at {a.run_id}/{a.machine_id}")
        if not np.array_equal(a.t_s, b.t_s) or a.state != b.state or a.fault_type != b.fault_type:
            raise SensorFaultDataError(
                f"{a.run_id}/{a.machine_id}: states or fault types differ from the clean run; "
                "the sensor-fault scenario must keep the physics identical (same seed, same "
                "scenario apart from sensor_faults)"
            )


def sensor_fault_rows(series: Sequence[MachineSeries]) -> npt.NDArray[np.bool_]:
    """Row mask (split row order) of rows inside a simulator sensor fault."""
    parts = [
        np.array([v != NO_FAULT_LABEL for v in s.labels[SENSOR_FAULT_COLUMN]], dtype=bool)
        for s in series
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)


def sensor_fault_sensors(series: Sequence[MachineSeries]) -> npt.NDArray[np.int8]:
    """Per row (split row order), the index in ``SENSORS`` of the faulty sensor, -1 if none.

    Simulator labels are ``<type>:<sensor>`` (e.g. ``drift:temp``) or ``none``.
    """
    from twin_ml.features.inputs import SENSORS

    parts = []
    for s in series:
        lab = s.labels[SENSOR_FAULT_COLUMN]
        parts.append(
            np.array(
                [
                    -1 if v == NO_FAULT_LABEL else SENSORS.index(str(v).split(":", 1)[-1])
                    for v in lab
                ],
                dtype=np.int8,
            )
        )
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int8)


def sensor_fault_split_detail(
    ds: Dataset,
    source_dir: Path,
    clean_series: Sequence[MachineSeries],
    line_config: Path,
    *,
    check_hashes: bool = True,
) -> tuple[SplitData, npt.NDArray[np.bool_], npt.NDArray[np.int8]]:
    """As :func:`sensor_fault_split`, plus the faulty sensor per row (-1 if none)."""
    if not (source_dir / "manifest.json").is_file():
        raise SensorFaultDataError(
            f"no sensor-fault runs at {source_dir}; generate them: uv run poe sim-sensor-faults"
        )
    run_ids = sorted({s.run_id for s in clean_series})
    series = load_series(
        source_dir,
        run_ids,
        line_config,
        check_hashes=check_hashes,
        extra_labels=(SENSOR_FAULT_COLUMN,),
    )
    check_same_labels(clean_series, series)
    return windowed(ds, series), sensor_fault_rows(series), sensor_fault_sensors(series)


def sensor_fault_split(
    ds: Dataset,
    source_dir: Path,
    clean_series: Sequence[MachineSeries],
    line_config: Path,
    *,
    check_hashes: bool = True,
) -> tuple[SplitData, npt.NDArray[np.bool_]]:
    """The test runs with simulator sensor faults, and the row mask of active sensor faults."""
    if not (source_dir / "manifest.json").is_file():
        raise SensorFaultDataError(
            f"no sensor-fault runs at {source_dir}; generate them: uv run poe sim-sensor-faults"
        )
    run_ids = sorted({s.run_id for s in clean_series})
    series = load_series(
        source_dir,
        run_ids,
        line_config,
        check_hashes=check_hashes,
        extra_labels=(SENSOR_FAULT_COLUMN,),
    )
    check_same_labels(clean_series, series)
    return windowed(ds, series), sensor_fault_rows(series)


# ----------------------------------------------------------------------------- E5 (Phase 9)


class E5DataError(ValueError):
    """The E5 runs are missing or do not match the fixed E5 design (ADR 0024)."""


def e5_run_ids(cfg: E5DataConfig, test_run_ids: Sequence[str]) -> list[str]:
    """The fixed E5 run selection: the E1 test-split run ids plus the fresh indices."""
    lo, hi = cfg.extra_run_indices
    ids = set(test_run_ids) if cfg.test_split_runs else set()
    ids |= {f"run_{i:04d}" for i in range(lo, hi)}
    out = sorted(ids)
    if len(out) != cfg.n_runs:
        raise E5DataError(f"E5 selection has {len(out)} runs, the design fixes {cfg.n_runs}")
    return out


PLAN_FIELDS = ("load_range", "ambient_c", "load_mu", "thresholds")


def check_shared_plan(
    clean_manifest: dict[str, Any], e5_manifest: dict[str, Any], run_ids: Sequence[str]
) -> int:
    """E5 runs that share an index with a clean run must share its operating plan.

    Load band, ambient, per-machine load level and thresholds come from the simulator's plan
    stream before any fault is drawn, so with the same base seed they must be identical.
    Returns the number of shared runs checked.
    """
    clean = {e["run_id"]: e for e in clean_manifest["run_entries"]}
    e5 = {e["run_id"]: e for e in e5_manifest["run_entries"]}
    shared = [r for r in run_ids if r in clean and r in e5]
    for r in shared:
        for f in PLAN_FIELDS:
            if clean[r].get(f) != e5[r].get(f):
                raise E5DataError(
                    f"{r}: {f} differs from the clean run; generate the E5 runs with the clean "
                    "base seed (uv run poe sim-e5)"
                )
    return len(shared)


def e5_split(
    ds: Dataset,
    cfg: E5DataConfig,
    source_dir: Path,
    run_ids: Sequence[str],
    line_config: Path,
    *,
    check_hashes: bool = True,
) -> SplitData:
    """The E5 runs windowed like ``ds``'s test split, with the held-out type in the vocab."""
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise E5DataError(f"no E5 runs at {source_dir}; generate them: uv run poe sim-e5")
    manifest = read_manifest(source_dir)
    seed = manifest.get("base_seed", manifest.get("seed"))
    if seed is not None and int(seed) != cfg.base_seed:
        raise E5DataError(f"{source_dir} was generated with seed {seed}, not {cfg.base_seed}")
    series = load_series(source_dir, run_ids, line_config, check_hashes=check_hashes)
    unexpected = {f for s in series for f in s.fault_type} - {NO_FAULT_LABEL, cfg.fault_type}
    if unexpected:
        raise E5DataError(f"E5 runs contain other fault types: {sorted(unexpected)}")
    return windowed(ds, series, "e5", extra_fault_types=(cfg.fault_type,))

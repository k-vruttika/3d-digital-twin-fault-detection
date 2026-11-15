"""Build the windowed dataset: ``python -m twin_ml.data build`` (poe task ``dataset``).

Steps: read the simulator manifest and runs, split by run, label every row, fit the
normaliser on training rows only, transform every split, cut windows, subsample far-normal
training windows (with compensating sample weights) and save ``<out_dir>/<key>/``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.config import load_line_config
from twin_common.logging import get_logger
from twin_ml.config import SPLITS, AugmentationConfig, DatasetConfig, RunFilterConfig
from twin_ml.data.dataset import META_FILE, dataset_key, manifest_hashes, save_split
from twin_ml.data.labels import (
    EPISODE_NAMES,
    FAULT_STATE,
    HEALTH,
    censor_ttf,
    episode_types,
    gradual_episodes,
    health_codes,
    state_codes,
)
from twin_ml.data.load import MachineSeries, iter_series, read_manifest, verify_files
from twin_ml.data.splits import split_runs
from twin_ml.data.windows import far_normal, subsample_far_normal, window_ends
from twin_ml.features.inputs import KINDS, SEQUENCE_CHANNELS, VALUE_CHANNELS
from twin_ml.features.normalize import Normalizer
from twin_ml.features.summary import summary_names
from twin_sim.config import GRADUAL_FAULTS, NO_FAULT_LABEL, FaultType
from twin_sim.labels import time_to_fault

log = get_logger(__name__)

GRADUAL = tuple(f.value for f in GRADUAL_FAULTS)
JAM = (FaultType.BELT_JAM.value,)


def _series_arrays(
    s: MachineSeries,
    norm: Normalizer,
    cfg: DatasetConfig,
    vocab: dict[str, int],
) -> dict[str, npt.NDArray[Any]]:
    n = s.t_s.shape[0]
    state = state_codes(s.state)
    health = health_codes(state)
    ttf_raw, no_fault_ahead = time_to_fault(s.t_s, state == FAULT_STATE, cap_s=np.inf)
    ttf, censored = censor_ttf(ttf_raw, no_fault_ahead, cfg.ttf_cap_s)
    values, mask = norm.transform(s.sensors, s.kind)
    return {
        "raw": s.sensors,
        "values": values,
        "mask": mask,
        "kind": np.full(n, KINDS.index(s.kind), dtype=np.int8),
        "t_s": s.t_s.astype(np.float32),
        "state": state,
        "health": health,
        "fault_type": np.array([vocab[f] for f in s.fault_type], dtype=np.int8),
        "ttf": ttf.astype(np.float32),
        "ttf_censored": censored,
        "episode_type": episode_types(state, s.fault_type, s.t_s, cfg.ttf_cap_s, GRADUAL, JAM),
        # helpers for windowing only (not saved)
        "_fault_within_cap": ~no_fault_ahead & (ttf_raw <= cfg.ttf_cap_s),
        "_fault_active": np.array([f != NO_FAULT_LABEL for f in s.fault_type]),
    }


def build_split(
    name: str,
    series: Sequence[MachineSeries],
    norm: Normalizer,
    cfg: DatasetConfig,
    vocab: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, npt.NDArray[Any]]:
    """Concatenate the series of one split and cut its windows."""
    rows: dict[str, list[npt.NDArray[Any]]] = {}
    ends: list[npt.NDArray[np.int64]] = []
    weights: list[npt.NDArray[np.float32]] = []
    episodes: dict[str, list[int]] = {k: [] for k in ("series", "ft", "start", "warn", "fault")}
    offset = 0
    stride = cfg.strides_s.of(name if name in SPLITS else "test")  # extra splits: test-like
    max_h = max(cfg.horizons_s)
    for sid, s in enumerate(series):
        arr = _series_arrays(s, norm, cfg, vocab)
        n = s.t_s.shape[0]
        arr["series"] = np.full(n, sid, dtype=np.int32)
        local = window_ends(n, cfg.window_s, max_h, stride)
        weight = np.ones(local.shape[0], dtype=np.float32)
        if name == "train" and cfg.normal_keep_frac < 1:
            far = far_normal(
                arr["health"][local],
                [arr["health"][local + h] for h in cfg.horizons_s],
                arr["_fault_active"][local],
                arr["_fault_within_cap"][local],
            )
            keep, weight = subsample_far_normal(far, cfg.normal_keep_frac, rng)
            local, weight = local[keep], weight[keep]
        ends.append(local + offset)
        weights.append(weight)
        for ep in gradual_episodes(arr["state"], s.fault_type, GRADUAL):
            episodes["series"].append(sid)
            episodes["ft"].append(vocab[ep.fault_type])
            episodes["start"].append(ep.i_start + offset)
            episodes["warn"].append(-1 if ep.i_warn is None else ep.i_warn + offset)
            episodes["fault"].append(ep.i_fault + offset)
        for key, value in arr.items():
            if not key.startswith("_"):
                rows.setdefault(key, []).append(value)
        offset += n
    out: dict[str, npt.NDArray[Any]] = {k: np.concatenate(v) for k, v in rows.items()}
    out["ends"] = np.concatenate(ends) if ends else np.zeros(0, dtype=np.int64)
    out["weight"] = np.concatenate(weights) if weights else np.zeros(0, dtype=np.float32)
    out["series_run"] = np.array([s.run_id for s in series], dtype=str)
    out["series_machine"] = np.array([s.machine_id for s in series], dtype=str)
    out["ep_series"] = np.array(episodes["series"], dtype=np.int32)
    out["ep_fault_type"] = np.array(episodes["ft"], dtype=np.int8)
    out["ep_start"] = np.array(episodes["start"], dtype=np.int64)
    out["ep_warn"] = np.array(episodes["warn"], dtype=np.int64)
    out["ep_fault"] = np.array(episodes["fault"], dtype=np.int64)
    return out


def _class_counts(
    arrays: dict[str, npt.NDArray[Any]], horizons: Sequence[int]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for h in horizons:
        y = arrays["health"][arrays["ends"] + h]
        w = arrays["weight"]
        out[f"h{h}"] = {HEALTH[c].value: int((y == c).sum()) for c in range(len(HEALTH))} | {
            f"{HEALTH[c].value}_weighted": float(w[y == c].sum()) for c in range(len(HEALTH))
        }
    return out


def select_runs(
    splits: dict[str, list[str]],
    load_range: dict[str, str],
    run_filter: RunFilterConfig | None,
) -> dict[str, list[str]]:
    """Apply a load-band run filter to the fixed split (E4; ADR 0023).

    train / val keep their runs of the listed bands. ``test`` becomes every run of its bands
    that is in neither the filtered train nor the filtered val split; ``test_in`` keeps the
    original test runs of its bands. Without a filter the split is returned unchanged.
    """
    if run_filter is None:
        return dict(splits)

    def keep(runs: list[str], bands: list[str]) -> list[str]:
        return sorted(r for r in runs if load_range[r] in bands)

    out = dict(splits)
    if run_filter.train is not None:
        out["train"] = keep(splits["train"], run_filter.train.load_ranges)
    if run_filter.val is not None:
        out["val"] = keep(splits["val"], run_filter.val.load_ranges)
    if run_filter.test is not None:
        used = set(out["train"]) | set(out["val"])
        out["test"] = sorted(
            r for r in load_range if load_range[r] in run_filter.test.load_ranges and r not in used
        )
    if run_filter.test_in is not None:
        out["test_in"] = keep(splits["test"], run_filter.test_in.load_ranges)
    return out


def augment_series(
    series: Sequence[MachineSeries],
    aug: AugmentationConfig,
    sigma: npt.NDArray[np.float32],
) -> tuple[list[MachineSeries], dict[str, int]]:
    """Replace each training series with a perturbed copy with probability ``aug.p`` (ADR 0021).

    Type uniform over the configured perturbations, level uniform in [min, max] of that type's
    grid, all from ``SeedSequence(aug.seed, (run, machine))`` (independent of the test seed).
    """
    import dataclasses

    from twin_ml.config import PERTURBATION_KINDS
    from twin_ml.data.perturb import Condition, perturb_array, series_rng

    out: list[MachineSeries] = []
    tally: dict[str, int] = {"clean": 0, **dict.fromkeys(PERTURBATION_KINDS, 0)}
    for s in series:
        rng = series_rng(aug.seed, "augment", s.run_id, s.machine_id)
        if rng.random() >= aug.p:
            out.append(s)
            tally["clean"] += 1
            continue
        kind = PERTURBATION_KINDS[int(rng.integers(len(PERTURBATION_KINDS)))]
        levels = aug.perturbations.grid(kind).levels_pct
        level = float(rng.uniform(min(levels), max(levels))) / 100.0
        cond = Condition(f"augment_{kind}", kind, level)
        raw = perturb_array(s.sensors, cond, aug.perturbations, sigma[KINDS.index(s.kind)], rng)
        out.append(dataclasses.replace(s, sensors=raw))
        tally[kind] += 1
    return out, tally


def series_normal_std(
    series: Sequence[MachineSeries], aug: AugmentationConfig
) -> npt.NDArray[np.float32]:
    """Normal-operation std per kind and sensor of clean training series (as for the tests)."""
    from twin_ml.data.perturb import normal_std

    raw = np.concatenate([s.sensors for s in series])
    state = np.concatenate([state_codes(s.state) for s in series])
    kind = np.concatenate([np.full(len(s.state), KINDS.index(s.kind)) for s in series])
    no_fault = np.concatenate(
        [np.array([f == NO_FAULT_LABEL for f in s.fault_type]) for s in series]
    )
    return normal_std(
        raw, state, kind, np.where(no_fault, 0, 1), 0, aug.perturbations.normal_states
    )


def build_dataset(cfg: DatasetConfig, root: Path = Path(), *, check_hashes: bool = True) -> Path:
    """Build the dataset for ``cfg`` and return its directory."""
    started = time.perf_counter()
    source = root / cfg.source_dir
    manifest = read_manifest(source)
    manifest_sha, content_sha = manifest_hashes(source)
    if check_hashes:
        verify_files(source, manifest)
    line = load_line_config(root / cfg.line_config)
    run_ids = [e["run_id"] for e in manifest["run_entries"]]
    load_range = {e["run_id"]: str(e.get("load_range")) for e in manifest["run_entries"]}
    splits = select_runs(
        split_runs(run_ids, cfg.split.fractions, cfg.split.seed), load_range, cfg.run_filter
    )
    names = [*SPLITS, *[n for n in splits if n not in SPLITS]]

    series = {name: list(iter_series(source, manifest, splits[name], line)) for name in names}
    vocab_names = sorted({f for name in names for s in series[name] for f in s.fault_type})
    vocab = {f: i for i, f in enumerate(vocab_names)}
    norm = Normalizer.fit(
        [(s.sensors, s.kind) for s in series["train"]], cfg.features.min_speed_ratio
    )
    augmentation: dict[str, Any] | None = None
    if cfg.augmentation is not None:  # after the normaliser: it stays fitted on clean data
        sigma = series_normal_std(series["train"], cfg.augmentation)
        series["train"], tally = augment_series(series["train"], cfg.augmentation, sigma)
        augmentation = {"series": tally, "normal_std": sigma.tolist()}
    key = dataset_key(cfg, content_sha)
    out = root / cfg.out_dir / key
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.subsample_seed)
    counts: dict[str, Any] = {}
    for name in names:
        arrays = build_split(name, series[name], norm, cfg, vocab, rng)
        save_split(out / f"{name}.npz", arrays)
        counts[name] = {
            "runs": len(splits[name]),
            "series": len(series[name]),
            "rows": int(arrays["t_s"].shape[0]),
            "windows": int(arrays["ends"].shape[0]),
            "gradual_episodes": int(arrays["ep_fault"].shape[0]),
            "classes": _class_counts(arrays, cfg.horizons_s),
        }
        log.info(
            "dataset.split", split=name, **{k: v for k, v in counts[name].items() if k != "classes"}
        )

    meta = {
        "dataset_key": key,
        "created_at": datetime.now(UTC).isoformat(),
        "build_time_s": round(time.perf_counter() - started, 2),
        "config": cfg.model_dump(mode="json"),
        "config_fingerprint": cfg.fingerprint(),
        "source": {
            "dir": cfg.source_dir.as_posix(),
            "manifest_sha256": manifest_sha,
            "content_sha256": content_sha,
            "scenario": manifest.get("scenario", {}).get("name"),
            "runs": manifest.get("runs"),
            "duration_s": manifest.get("duration_s"),
            "base_seed": manifest.get("base_seed"),
            "held_out_fault_types": manifest.get("held_out_fault_types", []),
            "git_commit": manifest.get("git_commit"),
        },
        "splits": splits,
        "extra_splits": [n for n in names if n not in SPLITS],
        "run_filter": (
            None
            if cfg.run_filter is None
            else {
                n: {
                    "runs": len(splits[n]),
                    "by_load": {
                        b: sum(load_range[r] == b for r in splits[n])
                        for b in sorted(set(load_range.values()))
                    },
                }
                for n in names
            }
        ),
        "augmentation": augmentation,
        "counts": counts,
        "normalizer": norm.to_dict(),
        "fault_types": vocab_names,
        "gradual_fault_types": [f for f in GRADUAL if f in vocab],
        "jam_fault_types": [f for f in JAM if f in vocab],
        "episode_types": list(EPISODE_NAMES),
        "health_classes": [h.value for h in HEALTH],
        "channels": {
            "values": list(VALUE_CHANNELS),
            "sequence": list(SEQUENCE_CHANNELS),
            "summary": summary_names(),
        },
        "kinds": [k.value for k in KINDS],
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("dataset.done", path=str(out), seconds=meta["build_time_s"])
    return out

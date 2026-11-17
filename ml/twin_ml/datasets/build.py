"""Build a windowed dataset from a public dataset (Phase 10, ADR 0027 / 0028).

The output has the layout of the simulator datasets (``<split>.npz`` + ``meta.json``, read by
:class:`twin_ml.data.dataset.Dataset`), so every model, detector and evaluation helper runs on
it unchanged. Additions in ``meta.json``: ``schema`` (sensors, kinds, seconds per row), ``rows``
(window, horizons, strides in rows), ``events`` (failure events per split, for the real-data
early-warning evaluation) and ``series`` (machine and time span of every series).

Steps: load the regular grids and failure events -> label every row from the events (full
machine timeline, before any cut) -> split each machine's grid into contiguous series at long
gaps -> cut every series by the time split -> fit the normaliser on training rows -> cut
windows (training far-normal windows subsampled with prior-restoring weights, as in Phase 7).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from twin_ml.config import SPLITS
from twin_ml.data.dataset import META_FILE, DatasetNotBuiltError, save_split
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
from twin_ml.data.windows import far_normal, subsample_far_normal, window_ends
from twin_ml.datasets import azure_pdm, metropt3
from twin_ml.datasets.common import (
    EPOCH,
    NO_FAULT,
    Event,
    Grid,
    label_rows,
    segments,
    time_to_fault,
)
from twin_ml.datasets.config import (
    DatasetFacts,
    RealDatasetConfig,
    load_facts,
    load_real_dataset_config,
)
from twin_ml.features.normalize import Normalizer
from twin_ml.features.schema import ChannelSchema
from twin_ml.features.summary import summary_names

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Loaded:
    """A public dataset on its regular grid, with its failure events."""

    facts: DatasetFacts
    t0: int  # grid origin, seconds since 1970 (naive clock)
    grids: list[Grid]
    events: list[Event]
    schema: ChannelSchema
    source_sha256: str


@dataclass(frozen=True)
class Series:
    """One contiguous stretch of one machine inside one split."""

    split: str
    machine_id: str
    kind: str
    t_s: npt.NDArray[np.float64]
    raw: npt.NDArray[np.float32]
    state: list[str]
    fault_type: list[str]

    @property
    def run_id(self) -> str:  # bootstrap / grouping unit: the machine
        return self.machine_id


def load(cfg: RealDatasetConfig, root: Path = Path(), *, sha: bool = True) -> Loaded:
    facts = load_facts(root / cfg.facts)
    if facts.name == "metropt3":
        raw_m = metropt3.read_raw(facts, root, sha=sha)
        t0, g = metropt3.grid(facts, raw_m, cfg.step_s)
        kinds: tuple[str, ...] = (g.kind,)
        grids, evs, digest = [g], metropt3.events(facts, t0), raw_m.sha256
        sensors = facts.sensor_names
    else:
        raw_a = azure_pdm.read_raw(facts, root, sha=sha)
        t0, grids = azure_pdm.grids(facts, raw_a, cfg.step_s, error_channels=cfg.error_channels)
        evs, digest = azure_pdm.events(raw_a, t0), raw_a.sha256
        kinds = tuple(sorted(set(raw_a.kinds.values())))
        sensors = (*facts.sensor_names, *(facts.error_codes if cfg.error_channels else ()))
    schema = ChannelSchema(sensors=tuple(sensors), kinds=kinds, vib_norm=False, step_s=cfg.step_s)
    return Loaded(facts, t0, grids, evs, schema, digest)


def holdout_machines(machines: list[str], fraction: float, seed: int) -> list[str]:
    """Held-out machines: a seeded permutation of the sorted ids, first ``round(f n)``."""
    ids = sorted(machines)
    perm = np.random.default_rng(seed).permutation(len(ids))
    return sorted(ids[i] for i in perm[: round(fraction * len(ids))])


def split_series(data: Loaded, cfg: RealDatasetConfig) -> dict[str, list[Series]]:
    """Label each machine's full grid, split it at long gaps, then cut by the time split."""
    held: set[str] = set()
    if cfg.machine_holdout is not None:
        held = set(
            holdout_machines(
                [g.machine_id for g in data.grids],
                cfg.machine_holdout.fraction,
                cfg.machine_holdout.seed,
            )
        )
    bounds = {
        name: tuple(
            float((np.datetime64(t, "s") - EPOCH).astype(np.int64) - data.t0)
            for t in cfg.split.of(name)
        )
        for name in SPLITS
    }
    out: dict[str, list[Series]] = {name: [] for name in SPLITS}
    for g in data.grids:
        evs = [e for e in data.events if e.machine_id == g.machine_id]
        state, ftype = label_rows(g.t_s, evs, cfg.labels, cfg.step_s)
        st, ft = np.asarray(state, dtype=object), np.asarray(ftype, dtype=object)
        for lo, hi in segments(g.observed, cfg.max_gap_rows):
            for name in SPLITS:
                if held and ((name == "test") != (g.machine_id in held)):
                    continue
                a, b = bounds[name]
                t = g.t_s[lo:hi]
                sel = np.flatnonzero((t >= a) & (t < b))
                if sel.size == 0:
                    continue
                # the cut keeps observed rows at both ends
                obs = g.observed[lo:hi][sel]
                first, last = np.flatnonzero(obs)[[0, -1]] if obs.any() else (0, -1)
                sel = sel[first : last + 1]
                if sel.size == 0:
                    continue
                rows = lo + sel
                out[name].append(
                    Series(
                        split=name,
                        machine_id=g.machine_id,
                        kind=g.kind,
                        t_s=g.t_s[rows],
                        raw=g.values[rows],
                        state=[str(x) for x in st[rows]],
                        fault_type=[str(x) for x in ft[rows]],
                    )
                )
    return out


def _series_arrays(
    s: Series,
    norm: Normalizer,
    schema: ChannelSchema,
    cfg: RealDatasetConfig,
    vocab: dict[str, int],
) -> dict[str, npt.NDArray[Any]]:
    n = s.t_s.shape[0]
    state = state_codes(s.state)
    health = health_codes(state)
    cap_s = float(cfg.ttf_cap * cfg.step_s)
    ttf_raw, no_fault_ahead = time_to_fault(s.t_s, state == FAULT_STATE)
    ttf, censored = censor_ttf(ttf_raw, no_fault_ahead, cap_s)
    values, mask = norm.transform(s.raw, s.kind)
    types = {f for f in vocab if f != NO_FAULT}
    return {
        "raw": s.raw,
        "values": values,
        "mask": mask,
        "kind": np.full(n, schema.kind_index(s.kind), dtype=np.int8),
        "t_s": s.t_s.astype(np.float64),
        "state": state,
        "health": health,
        "fault_type": np.array([vocab[f] for f in s.fault_type], dtype=np.int8),
        "ttf": ttf.astype(np.float32),
        "ttf_censored": censored,
        "episode_type": episode_types(state, s.fault_type, s.t_s, cap_s, types, ()),
        "_fault_within_cap": ~no_fault_ahead & (ttf_raw <= cap_s),
        "_fault_active": np.array([f != NO_FAULT for f in s.fault_type]),
    }


def build_split(
    name: str,
    series: list[Series],
    norm: Normalizer,
    schema: ChannelSchema,
    cfg: RealDatasetConfig,
    vocab: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, npt.NDArray[Any]]:
    """Concatenate the series of one split and cut its windows (as ``data.build``)."""
    rows: dict[str, list[npt.NDArray[Any]]] = {}
    ends: list[npt.NDArray[np.int64]] = []
    weights: list[npt.NDArray[np.float32]] = []
    episodes: dict[str, list[int]] = {k: [] for k in ("series", "ft", "start", "warn", "fault")}
    offset = 0
    stride = getattr(cfg.strides, name)
    max_h = max(cfg.horizons)
    types = {f for f in vocab if f != NO_FAULT}
    for sid, s in enumerate(series):
        arr = _series_arrays(s, norm, schema, cfg, vocab)
        n = s.t_s.shape[0]
        arr["series"] = np.full(n, sid, dtype=np.int32)
        local = window_ends(n, cfg.window, max_h, stride)
        weight = np.ones(local.shape[0], dtype=np.float32)
        if name == "train" and cfg.normal_keep_frac < 1:
            far = far_normal(
                arr["health"][local],
                [arr["health"][local + h] for h in cfg.horizons],
                arr["_fault_active"][local],
                arr["_fault_within_cap"][local],
            )
            keep, weight = subsample_far_normal(far, cfg.normal_keep_frac, rng)
            local, weight = local[keep], weight[keep]
        ends.append(local + offset)
        weights.append(weight)
        for ep in gradual_episodes(arr["state"], s.fault_type, types):
            episodes["series"].append(sid)
            episodes["ft"].append(vocab[ep.fault_type])
            episodes["start"].append(ep.i_start + offset)
            episodes["warn"].append(-1)  # no machine WARNING signal in real data
            episodes["fault"].append(ep.i_fault + offset)
        for key, value in arr.items():
            if not key.startswith("_"):
                rows.setdefault(key, []).append(value)
        offset += n
    out: dict[str, npt.NDArray[Any]] = {k: np.concatenate(v) for k, v in rows.items()}
    if not rows:
        raise DatasetNotBuiltError(f"split {name} has no data; check the split dates")
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


def dataset_key(cfg: RealDatasetConfig, facts: DatasetFacts, source_sha256: str) -> str:
    import hashlib

    return hashlib.sha256(f"{cfg.fingerprint(facts)}:{source_sha256}".encode()).hexdigest()[:16]


def resolve_real_dataset(cfg_path: Path, root: Path = Path()) -> Path:
    """The built dataset for the current files and config, or a clear error."""
    cfg = load_real_dataset_config(root / cfg_path)
    facts = load_facts(root / cfg.facts)
    from twin_ml.datasets.common import files_sha256, require_files

    paths = require_files(facts, root)
    digest = files_sha256([paths[k] for k in sorted(paths)])  # as the loaders hash them
    path = root / cfg.out_dir / dataset_key(cfg, facts, digest)
    if not (path / META_FILE).is_file():
        raise DatasetNotBuiltError(
            f"no built dataset at {path} for {cfg_path}; build it with: uv run poe real-data"
        )
    return path


def _class_counts(arrays: dict[str, npt.NDArray[Any]], horizons: list[int]) -> dict[str, Any]:
    ends = arrays["ends"]
    out: dict[str, Any] = {}
    for h in horizons:
        y = arrays["health"][ends + h]
        out[f"h{h}"] = {HEALTH[c].value: int((y == c).sum()) for c in range(len(HEALTH))}
    return out


def _iso(t0: int, t_s: float) -> str:
    return str(EPOCH + np.timedelta64(int(t0 + t_s), "s")).replace("T", " ")


def build_real_dataset(cfg_path: Path, root: Path = Path()) -> Path:
    """Build ``config/models/real_<name>.yaml`` and return the dataset directory."""
    started = time.perf_counter()
    cfg = load_real_dataset_config(root / cfg_path)
    data = load(cfg, root)
    series = split_series(data, cfg)
    for name in SPLITS:
        if not series[name]:
            raise DatasetNotBuiltError(f"split {name} has no series; check the split dates")
    vocab_names = sorted(
        {NO_FAULT} | {f for name in SPLITS for s in series[name] for f in s.fault_type}
    )
    vocab = {f: i for i, f in enumerate(vocab_names)}
    norm = Normalizer.fit([(s.raw, s.kind) for s in series["train"]], 0.1, schema=data.schema)
    key = dataset_key(cfg, data.facts, data.source_sha256)
    out = root / cfg.out_dir / key
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.subsample_seed)
    counts: dict[str, Any] = {}
    split_events: dict[str, list[dict[str, Any]]] = {}
    series_meta: dict[str, list[dict[str, Any]]] = {}
    for name in SPLITS:
        arrays = build_split(name, series[name], norm, data.schema, cfg, vocab, rng)
        save_split(out / f"{name}.npz", arrays)
        machines = sorted({s.machine_id for s in series[name]})
        a, b = (
            float((np.datetime64(t, "s") - EPOCH).astype(np.int64) - data.t0)
            for t in cfg.split.of(name)
        )
        split_events[name] = [
            {
                "id": e.id,
                "machine_id": e.machine_id,
                "start_s": e.start_s,
                "end_s": e.end_s,
                "start": _iso(data.t0, e.start_s),
                "end": _iso(data.t0, e.end_s),
                "fault_type": e.fault_type,
                "components": list(e.components),
            }
            for e in data.events
            if e.machine_id in machines and a <= e.start_s < b
        ]
        series_meta[name] = [
            {
                "machine_id": s.machine_id,
                "kind": s.kind,
                "rows": int(s.t_s.shape[0]),
                "start": _iso(data.t0, float(s.t_s[0])),
                "end": _iso(data.t0, float(s.t_s[-1])),
            }
            for s in series[name]
        ]
        counts[name] = {
            "machines": len(machines),
            "series": len(series[name]),
            "rows": int(arrays["t_s"].shape[0]),
            "windows": int(arrays["ends"].shape[0]),
            "events": len(split_events[name]),
            "episodes": int(arrays["ep_fault"].shape[0]),
            "classes": _class_counts(arrays, cfg.horizons),
        }
        log.info(
            "real_dataset.split",
            split=name,
            **{k: v for k, v in counts[name].items() if k != "classes"},
        )
    meta = {
        "dataset_key": key,
        "dataset": data.facts.name,
        "created_at": datetime.now(UTC).isoformat(),
        "build_time_s": round(time.perf_counter() - started, 2),
        "config": cfg.model_dump(mode="json"),
        "config_path": cfg_path.as_posix(),
        "config_fingerprint": cfg.fingerprint(data.facts),
        "source": {
            "dir": data.facts.dir.as_posix(),
            "files": dict(data.facts.files),
            "sha256": data.source_sha256,
            "origin": data.facts.origin,
        },
        "t0": _iso(data.t0, 0.0),
        "schema": data.schema.to_dict(),
        "rows": {
            "window": cfg.window,
            "horizons": list(cfg.horizons),
            "strides": cfg.strides.model_dump(),
            "ttf_cap_s": float(cfg.ttf_cap * cfg.step_s),
        },
        "holdout_machines": (
            None
            if cfg.machine_holdout is None
            else holdout_machines(
                [g.machine_id for g in data.grids],
                cfg.machine_holdout.fraction,
                cfg.machine_holdout.seed,
            )
        ),
        "splits": {name: sorted({s.machine_id for s in series[name]}) for name in SPLITS},
        "extra_splits": [],
        "series": series_meta,
        "events": split_events,
        "counts": counts,
        "normalizer": norm.to_dict(),
        "fault_types": vocab_names,
        "gradual_fault_types": [f for f in vocab_names if f != NO_FAULT],
        "jam_fault_types": [],
        "episode_types": list(EPISODE_NAMES),
        "health_classes": [h.value for h in HEALTH],
        "channels": {
            "values": list(data.schema.value_channels),
            "sequence": list(data.schema.sequence_channels),
            "summary": summary_names(data.schema),
        },
        "kinds": list(data.schema.kinds),
        "sensor_parts": {
            **data.facts.sensor_parts,
            **(dict.fromkeys(data.facts.error_codes, "error_log") if cfg.error_channels else {}),
        },
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("real_dataset.done", path=str(out), seconds=meta["build_time_s"])
    return out

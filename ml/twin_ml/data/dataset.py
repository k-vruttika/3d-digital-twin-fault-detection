"""Built dataset access (no simulator import): split arrays, windows, labels, batches.

Layout of ``<out_dir>/<dataset_key>/``: ``meta.json`` plus ``<split>.npz`` per split. Rows of
all machine series of a split are concatenated; a window is identified by the row index of its
last sample (``ends``) and never crosses a series boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.config import SPLITS, DatasetConfig
from twin_ml.features.inputs import KINDS, FloatArray, MaskArray
from twin_ml.features.summary import summary_features

MANIFEST_FILE = "manifest.json"  # == twin_sim.batch.MANIFEST_NAME (checked by a test)
META_FILE = "meta.json"

ROW_FIELDS = (
    "raw",
    "values",
    "mask",
    "kind",
    "series",
    "t_s",
    "state",
    "health",
    "fault_type",
    "ttf",
    "ttf_censored",
    "episode_type",
)
WINDOW_FIELDS = ("ends", "weight")
SERIES_FIELDS = ("series_run", "series_machine")
EPISODE_FIELDS = ("ep_series", "ep_fault_type", "ep_start", "ep_warn", "ep_fault")


class DatasetNotBuiltError(FileNotFoundError):
    pass


def manifest_hashes(source_dir: Path) -> tuple[str, str]:
    """(sha256 of manifest.json's bytes, content_sha256 recorded in it)."""
    path = source_dir / MANIFEST_FILE
    if not path.is_file():
        raise DatasetNotBuiltError(
            f"no simulator manifest at {path}; generate the data first: uv run poe dataset"
        )
    raw = path.read_bytes()
    manifest = json.loads(raw)
    content = manifest.get("content_sha256")
    if not content:
        raise DatasetNotBuiltError(
            f"{path} has no content_sha256 (made by an older simulator); regenerate it with "
            "uv run poe dataset"
        )
    return hashlib.sha256(raw).hexdigest(), str(content)


def dataset_key(cfg: DatasetConfig, content_sha256: str) -> str:
    return hashlib.sha256(f"{cfg.fingerprint()}:{content_sha256}".encode()).hexdigest()[:16]


def resolve_dataset_dir(cfg: DatasetConfig, root: Path = Path()) -> Path:
    """The built dataset for the current source data and config, or a clear error."""
    _, content = manifest_hashes(root / cfg.source_dir)
    path = root / cfg.out_dir / dataset_key(cfg, content)
    if not (path / META_FILE).is_file():
        raise DatasetNotBuiltError(
            f"no built dataset at {path} for the current data and config/models/dataset.yaml; "
            "build it with: uv run poe dataset"
        )
    return path


@dataclass
class SplitData:
    """One split of the built dataset. Row arrays have R rows, window arrays N windows."""

    name: str
    window: int
    horizons: tuple[int, ...]
    stride: int
    arrays: dict[str, npt.NDArray[Any]] = field(repr=False)

    def __getattr__(self, item: str) -> npt.NDArray[Any]:
        arrays = self.__dict__.get("arrays")
        if arrays is not None and item in arrays:
            value: npt.NDArray[Any] = arrays[item]
            return value
        raise AttributeError(item)

    # ------------------------------------------------------------------ sizes
    @property
    def n_windows(self) -> int:
        return int(self.arrays["ends"].shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.arrays["t_s"].shape[0])

    # ------------------------------------------------------------------ labels
    def at_end(self, name: str, idx: npt.NDArray[np.int64] | None = None) -> npt.NDArray[Any]:
        """A row field at each window's last sample."""
        ends = self.arrays["ends"] if idx is None else self.arrays["ends"][idx]
        out: npt.NDArray[Any] = self.arrays[name][ends]
        return out

    def target(self, horizon: int, idx: npt.NDArray[np.int64] | None = None) -> npt.NDArray[Any]:
        """Health class ``horizon`` seconds after each window's last sample."""
        if horizon not in self.horizons:
            raise KeyError(f"horizon {horizon} not in {self.horizons}")
        ends = self.arrays["ends"] if idx is None else self.arrays["ends"][idx]
        out: npt.NDArray[Any] = self.arrays["health"][ends + horizon]
        return out

    def window_run(self) -> npt.NDArray[np.int64]:
        """Index into ``run_ids`` of each window's run."""
        runs = self.run_ids
        lookup = {r: i for i, r in enumerate(runs)}
        series_run = np.array([lookup[r] for r in self.arrays["series_run"]], dtype=np.int64)
        out: npt.NDArray[np.int64] = series_run[self.at_end("series")]
        return out

    @property
    def run_ids(self) -> list[str]:
        return sorted({str(r) for r in self.arrays["series_run"]})

    # ------------------------------------------------------------------ inputs
    def gather(self, idx: npt.NDArray[np.int64]) -> tuple[FloatArray, MaskArray, FloatArray]:
        """Windows ``idx`` -> values (B, W, 6), mask (B, W, 5), kind one-hot (B, kinds)."""
        ends = self.arrays["ends"][idx]
        rows = ends[:, None] + np.arange(-self.window + 1, 1)[None, :]
        values = self.arrays["values"][rows]
        mask = self.arrays["mask"][rows]
        kind = np.eye(len(KINDS), dtype=np.float32)[self.arrays["kind"][ends]]
        return values, mask, kind

    def sequence(self, idx: npt.NDArray[np.int64]) -> FloatArray:
        """Windows as the sequence models see them: (B, W, SEQUENCE_CHANNELS)."""
        values, mask, kind = self.gather(idx)
        kind_t = np.broadcast_to(kind[:, None, :], (*values.shape[:2], kind.shape[1]))
        return np.concatenate([values, mask.astype(np.float32), kind_t], axis=2)

    def summary(self, idx: npt.NDArray[np.int64] | None = None, chunk: int = 50_000) -> FloatArray:
        """Window summary features for the tree models, computed in chunks."""
        all_idx = np.arange(self.n_windows) if idx is None else idx
        parts = [
            summary_features(*self.gather(all_idx[i : i + chunk]))
            for i in range(0, all_idx.shape[0], chunk)
        ]
        if not parts:
            from twin_ml.features.summary import summary_names

            return np.zeros((0, len(summary_names())), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def batches(
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> Iterator[npt.NDArray[np.int64]]:
        """Window index batches, shuffled when ``rng`` is given."""
        order = np.arange(self.n_windows) if rng is None else rng.permutation(self.n_windows)
        for i in range(0, order.shape[0], batch_size):
            yield order[i : i + batch_size]


@dataclass
class Dataset:
    path: Path
    meta: dict[str, Any]
    _splits: dict[str, SplitData] = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, path: Path) -> Dataset:
        meta_path = path / META_FILE
        if not meta_path.is_file():
            raise DatasetNotBuiltError(f"no built dataset at {path}")
        return cls(path=path, meta=json.loads(meta_path.read_text(encoding="utf-8")))

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(int(h) for h in self.meta["config"]["horizons_s"])

    @property
    def fault_types(self) -> list[str]:
        return list(self.meta["fault_types"])

    @property
    def ttf_cap_s(self) -> float:
        return float(self.meta["config"]["ttf_cap_s"])

    def split(self, name: str) -> SplitData:
        """``train`` / ``val`` / ``test`` or an extra split of this build (e.g. ``test_in``)."""
        extra = list(self.meta.get("extra_splits", []))
        if name not in SPLITS and name not in extra:
            raise KeyError(f"unknown split {name!r}")
        if name not in self._splits:
            with np.load(self.path / f"{name}.npz", allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files}
            self._splits[name] = SplitData(
                name=name,
                window=int(self.meta["config"]["window_s"]),
                horizons=self.horizons,
                stride=int(self.meta["config"]["strides_s"][name if name in SPLITS else "test"]),
                arrays=arrays,
            )
        return self._splits[name]


def save_split(path: Path, arrays: dict[str, npt.NDArray[Any]]) -> None:
    expected = set(ROW_FIELDS) | set(WINDOW_FIELDS) | set(SERIES_FIELDS) | set(EPISODE_FIELDS)
    if set(arrays) != expected:
        raise ValueError(f"split arrays mismatch: {sorted(set(arrays) ^ expected)}")
    np.savez(path, **arrays)  # type: ignore[arg-type]

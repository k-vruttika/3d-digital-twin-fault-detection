"""Common interface of every model and baseline, and the shared class-weight rule."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from twin_ml.config import ClassWeightConfig
from twin_ml.data.dataset import SplitData
from twin_ml.evaluate.evaluator import Predictions


class Predictor(Protocol):
    name: str

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        """Train; returns training information (epochs, best validation score, ...)."""
        ...

    def predict(self, split: SplitData) -> Predictions:
        """Predictions for every window of ``split``."""
        ...

    def predict_one(self, split: SplitData, i: int) -> None:
        """One window end to end (features included), for the batch-size-1 latency."""
        ...


def class_weights(
    y: npt.NDArray[Any],
    sample_weight: npt.NDArray[np.float32],
    n_classes: int,
    cfg: ClassWeightConfig,
) -> npt.NDArray[np.float64]:
    """Per-class weight ``min((W / (K * W_c)) ** power, max)`` from *weighted* counts.

    The counts use the subsampling weights, so they describe the original class prior; a class
    without samples gets weight 0. ``power`` 0 disables class weighting.
    """
    counts = np.bincount(y.astype(np.int64), weights=sample_weight, minlength=n_classes)
    present = counts > 0
    k = int(present.sum())
    out = np.zeros(n_classes, dtype=np.float64)
    if k == 0:
        return out
    out[present] = np.minimum((counts.sum() / (k * counts[present])) ** cfg.power, cfg.max)
    return out


def effective_weights(
    y: npt.NDArray[Any], sample_weight: npt.NDArray[np.float32], cw: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Sample weight times class weight: the weight each training window gets in the loss."""
    return sample_weight.astype(np.float64) * cw[y.astype(np.int64)]

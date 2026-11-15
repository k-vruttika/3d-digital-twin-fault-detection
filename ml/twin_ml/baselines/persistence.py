"""Persistence baseline: the current health class continues at every horizon.

It reads the machine ``state`` at the window end *by design*: it stands for what the twin
already shows today. It is evaluated next to the models but never used as a model input.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import HEALTH
from twin_ml.evaluate.evaluator import Predictions


class PersistenceBaseline:
    name = "persistence"

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        return {}

    def predict(self, split: SplitData) -> Predictions:
        onehot = np.eye(len(HEALTH), dtype=np.float32)[split.at_end("health").astype(np.int64)]
        return Predictions(probs=dict.fromkeys(split.horizons, onehot))

    def predict_one(self, split: SplitData, i: int) -> None:
        idx = np.array([i])
        np.eye(len(HEALTH), dtype=np.float32)[split.at_end("health", idx).astype(np.int64)]

"""Probability calibration fitted on clean validation data (Phase 8, ADR 0022).

* sequence models (``nn``): temperature scaling of each horizon head's logits;
* tree models: one-vs-rest isotonic or Platt scaling, chosen by run-grouped CV NLL;
* baselines: none (their one-hot outputs are reported as they are).

No torch import; scikit-learn only inside the tree calibrators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.calibrate.temperature import fit_temperature, softmax
from twin_ml.calibrate.trees import OvRCalibrator, mean_nll, select_method
from twin_ml.config import CalibrationConfig


@dataclass
class HeadCalibration:
    """Calibration map of one horizon head."""

    method: str  # temperature | isotonic | platt | none
    temperature: float | None = None
    ovr: OvRCalibrator | None = None
    cv_nll: dict[str, float] = field(default_factory=dict)
    val_nll_before: float = float("nan")
    val_nll_after: float = float("nan")

    def apply(
        self,
        probs: npt.NDArray[np.floating[Any]],
        logits: npt.NDArray[np.floating[Any]] | None = None,
    ) -> npt.NDArray[np.float64]:
        if self.method == "temperature":
            if logits is None or self.temperature is None:
                raise ValueError("temperature scaling needs the logits")
            return softmax(logits, self.temperature)
        if self.ovr is not None:
            return self.ovr.transform(probs)
        return np.asarray(probs, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "cv_nll": self.cv_nll,
            "val_nll_before": self.val_nll_before,
            "val_nll_after": self.val_nll_after,
        }


def fit_head(
    family: str,
    probs: npt.NDArray[np.floating[Any]],
    logits: npt.NDArray[np.floating[Any]] | None,
    y: npt.NDArray[Any],
    groups: npt.NDArray[Any],
    cfg: CalibrationConfig,
) -> HeadCalibration:
    """Fit the calibration of one head on clean validation windows."""
    before = mean_nll(probs, y)
    if family == "nn":
        if logits is None:
            raise ValueError("sequence-model calibration needs validation logits")
        t = fit_temperature(logits, y, cfg.temperature_bounds)
        cal = HeadCalibration("temperature", temperature=t, val_nll_before=before)
    elif family == "tree":
        method, scores = select_method(probs, y, groups, cfg.tree_methods, cfg.cv_folds)
        cal = HeadCalibration(
            method, ovr=OvRCalibrator(method).fit(probs, y), cv_nll=scores, val_nll_before=before
        )
    else:
        cal = HeadCalibration("none", val_nll_before=before)
    cal.val_nll_after = mean_nll(cal.apply(probs, logits), y)
    return cal

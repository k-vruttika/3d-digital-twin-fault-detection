"""Exported calibration and conformal maps for live inference (pure numpy; ADR 0022, 0029).

The Phase 8 maps (``twin_ml.calibrate``, ``twin_ml.conformal``) are fitted with scikit-learn
on the clean validation split and were never stored. The deploy bundle refits them with the
same code and exports them as plain numbers, so the live worker applies them without
scikit-learn:

* temperature scaling: ``T`` per horizon head;
* one-vs-rest Platt scaling: ``(coef, intercept)`` per class (``predict_proba`` of a binary
  logistic regression is ``expit(coef * logit(p) + intercept)``);
* one-vs-rest isotonic regression: the fitted thresholds (``np.interp`` with clipping, as
  ``IsotonicRegression(out_of_bounds="clip").predict``);
* Mondrian LAC: one threshold per class (``conformal.classification.lac_sets``);
* time to fault: the absolute-residual quantile (``conformal.regression.interval``).

``LiveCalibration.apply`` reproduces ``HeadCalibration.apply`` (checked by a test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.calibrate.temperature import softmax
from twin_ml.calibrate.trees import _logit
from twin_ml.conformal.classification import lac_sets
from twin_ml.conformal.regression import interval

Probs = npt.NDArray[np.float64]


def _expit(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-z))
    return out


@dataclass
class LiveHeadCalibration:
    """One horizon head's calibration map as plain numbers."""

    method: str  # temperature | platt | isotonic | none
    temperature: float | None = None
    platt: dict[int, tuple[float, float]] = field(default_factory=dict)
    isotonic: dict[int, tuple[list[float], list[float]]] = field(default_factory=dict)

    @classmethod
    def from_head(cls, head: Any) -> LiveHeadCalibration:
        """From a fitted ``twin_ml.calibrate.HeadCalibration``."""
        if head.method == "temperature":
            return cls("temperature", temperature=float(head.temperature))
        if head.method in ("platt", "isotonic"):
            out = cls(head.method)
            for k, m in head.ovr.maps.items():
                if head.method == "platt":
                    out.platt[int(k)] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
                else:
                    out.isotonic[int(k)] = (
                        [float(x) for x in m.X_thresholds_],
                        [float(y) for y in m.y_thresholds_],
                    )
            return out
        return cls("none")

    def apply(
        self,
        probs: npt.NDArray[np.floating[Any]],
        logits: npt.NDArray[np.floating[Any]] | None = None,
    ) -> Probs:
        if self.method == "temperature":
            if logits is None or self.temperature is None:
                raise ValueError("temperature scaling needs the logits")
            return softmax(logits, self.temperature)
        p = np.asarray(probs, dtype=np.float64)
        if self.method == "none":
            return p
        out = p.copy()
        if self.method == "platt":
            for k, (a, b) in self.platt.items():
                out[:, k] = _expit(a * _logit(p[:, k]) + b)
        else:
            for k, (xs, ys) in self.isotonic.items():
                x = np.asarray(xs, dtype=np.float64)
                out[:, k] = np.interp(np.clip(p[:, k], x[0], x[-1]), x, np.asarray(ys))
        # renormalise exactly as OvRCalibrator.transform
        total = out.sum(axis=1, keepdims=True)
        bad = ~(total[:, 0] > 0)
        out = np.where(total > 0, out / np.where(total > 0, total, 1.0), p)
        if bad.any():
            out[bad] = p[bad]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "platt": {str(k): list(v) for k, v in self.platt.items()},
            "isotonic": {str(k): {"x": v[0], "y": v[1]} for k, v in self.isotonic.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiveHeadCalibration:
        return cls(
            method=str(d["method"]),
            temperature=None if d.get("temperature") is None else float(d["temperature"]),
            platt={int(k): (float(v[0]), float(v[1])) for k, v in d.get("platt", {}).items()},
            isotonic={
                int(k): ([float(x) for x in v["x"]], [float(y) for y in v["y"]])
                for k, v in d.get("isotonic", {}).items()
            },
        )


def _q_out(q: float) -> float | None:
    return None if math.isinf(q) else float(q)


def _q_in(q: float | None) -> float:
    return math.inf if q is None else float(q)


@dataclass
class LiveCalibration:
    """Calibration + Mondrian LAC per horizon, plus the time-to-fault interval quantile."""

    heads: dict[int, LiveHeadCalibration]
    mondrian_q: dict[int, npt.NDArray[np.float64]]  # per class; inf = always in the set
    alpha: float
    ttf_q: float | None
    ttf_cap_s: float

    def calibrated(
        self,
        horizon: int,
        probs: npt.NDArray[np.floating[Any]],
        logits: npt.NDArray[np.floating[Any]] | None,
    ) -> Probs:
        return self.heads[horizon].apply(probs, logits)

    def sets(self, horizon: int, cal_probs: Probs) -> npt.NDArray[np.bool_]:
        return lac_sets(cal_probs, self.mondrian_q[horizon])

    def ttf_interval(
        self, ttf: npt.NDArray[np.floating[Any]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        if self.ttf_q is None:
            return None
        return interval(ttf, self.ttf_q, self.ttf_cap_s)

    @classmethod
    def from_fitted(cls, cal: Any, ttf_cap_s: float) -> LiveCalibration:
        """From a fitted ``twin_ml.experiments.robust_eval.Calibration``."""
        heads = {int(h): LiveHeadCalibration.from_head(c) for h, c in cal.heads.items()}
        mq = {
            int(h): np.asarray(c.q["mondrian_lac"], dtype=np.float64)
            for h, c in cal.conformal.items()
        }
        alpha = float(next(iter(cal.conformal.values())).alpha)
        return cls(heads, mq, alpha, cal.ttf_q, float(ttf_cap_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "heads": {str(h): c.to_dict() for h, c in self.heads.items()},
            "mondrian_q": {str(h): [_q_out(x) for x in q] for h, q in self.mondrian_q.items()},
            "ttf_q": None if self.ttf_q is None else _q_out(self.ttf_q),
            "ttf_cap_s": self.ttf_cap_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LiveCalibration:
        return cls(
            heads={int(h): LiveHeadCalibration.from_dict(c) for h, c in d["heads"].items()},
            mondrian_q={
                int(h): np.asarray([_q_in(x) for x in q], dtype=np.float64)
                for h, q in d["mondrian_q"].items()
            },
            alpha=float(d["alpha"]),
            ttf_q=None if d.get("ttf_q") is None else float(d["ttf_q"]),
            ttf_cap_s=float(d["ttf_cap_s"]),
        )

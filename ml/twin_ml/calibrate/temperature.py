"""Temperature scaling for the sequence models (Guo et al., 2017). Pure numpy.

One scalar T per horizon head, fitted on clean validation logits by minimising the negative
log-likelihood of ``softmax(logits / T)``. T does not change the argmax, so alarms are
unaffected. The NLL is minimised over log T with a golden-section search inside the configured
bounds (the NLL is smooth and unimodal in T for practical purposes; the bracket is checked).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt


def softmax(
    logits: npt.NDArray[np.floating[Any]], temperature: float = 1.0
) -> npt.NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: npt.NDArray[np.float64] = e / e.sum(axis=1, keepdims=True)
    return out


def nll(logits: npt.NDArray[np.floating[Any]], y: npt.NDArray[Any], temperature: float) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_p[np.arange(y.shape[0]), y.astype(np.int64)].mean())


def fit_temperature(
    logits: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[Any],
    bounds: tuple[float, float],
    tol: float = 1e-4,
) -> float:
    """The T in ``bounds`` that minimises the validation NLL (golden-section on log T)."""
    if y.shape[0] == 0:
        return 1.0
    a, b = math.log(bounds[0]), math.log(bounds[1])
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, math.exp(c)), nll(logits, y, math.exp(d))
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, math.exp(d))
    return math.exp((a + b) / 2.0)

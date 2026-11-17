"""Attributions and explanations (Phase 9, ADR 0026).

:func:`explain` is the entry point Phase 11 calls: one machine's current window in, the
per-sensor attribution mapped to 3D parts out (``[{sensor, part, weight}]``). It works on the
model objects of ``twin_ml.models`` (XGBoost / Random Forest ``TreeModel``, LSTM / GRU /
Transformer ``SequenceModel``) and imports neither the simulator nor, at import time, any ML
library: torch is loaded only when a sequence model is explained.

The explained output is the model's predicted alarm class at the horizon: whichever of
WARNING and FAULT has the higher probability. Positive weights are evidence for it.

**One OpenMP library per process (ADR 0020).** torch and XGBoost each bring an OpenMP runtime,
and on macOS the two in one process segfault. A process may explain XGBoost models or
sequence models, never both: :func:`require_alone` raises before either library is used if the
other one is already loaded. Phase 11 live inference must keep tree and sequence models in
separate processes.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_common.enums import MachineKind
from twin_ml.data.labels import WARNING
from twin_ml.explain.aggregate import (
    normalise,
    sequence_channel_sensors,
    summary_feature_sensors,
    to_parts,
    to_sensors,
)
from twin_ml.features.inputs import KINDS
from twin_ml.features.summary import summary_features

DEFAULT_IG_STEPS = 64  # ADR 0024 / e5.yaml explain.ig_steps
OPENMP_LIBRARIES = ("torch", "xgboost")  # never in one process (ADR 0020)


class OpenMPClashError(RuntimeError):
    """torch and XGBoost would end up in one process (segfaults on macOS; ADR 0020)."""


def require_alone(library: str) -> None:
    """Raise :class:`OpenMPClashError` if the other library of ``OPENMP_LIBRARIES`` is loaded."""
    for other in OPENMP_LIBRARIES:
        if other != library and other in sys.modules:
            raise OpenMPClashError(
                f"cannot use {library} for explanations: {other} is already loaded in this "
                "process, and their two OpenMP runtimes crash on macOS. Explain tree models "
                "and sequence models in separate processes (ADR 0020, ADR 0026)."
            )


def _require_for(fam: str) -> None:
    if fam == "xgb":
        require_alone("xgboost")
    elif fam == "nn":
        require_alone("torch")


def family(model: Any) -> str:
    """'xgb', 'rf' or 'nn' (by class, without importing the model modules)."""
    name = type(model).__name__
    if name == "TreeModel":
        return str(model.cfg.type)
    if name == "SequenceModel":
        return "nn"
    raise TypeError(f"cannot explain a {name}")


@dataclass
class Attribution:
    sensors: npt.NDArray[np.float64]  # (B, 5) signed per-sensor sums
    cls: npt.NDArray[np.int64]  # explained class per window (WARNING or FAULT)
    completeness_error: npt.NDArray[np.float64] | None  # IG only
    delta: npt.NDArray[np.float64] | None  # IG only: F(x) - F(baseline)


def _sequence(values: Any, mask: Any, kind: Any) -> npt.NDArray[np.float32]:
    k = np.broadcast_to(kind[:, None, :], (*values.shape[:2], kind.shape[1]))
    return np.concatenate([values, mask.astype(np.float32), k], axis=2).astype(np.float32)


def _alarm_class(probs: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.int64]:
    out: npt.NDArray[np.int64] = (np.argmax(probs[:, WARNING:], axis=1) + WARNING).astype(np.int64)
    return out


def probabilities(
    model: Any, values: Any, mask: Any, kind: Any, horizon: int
) -> npt.NDArray[np.float64]:
    """(B, 3) health probabilities at ``horizon`` for windows given as arrays."""
    fam = family(model)
    _require_for(fam)
    if fam in ("xgb", "rf"):
        x = summary_features(values, mask, kind)
        return np.asarray(model.heads[f"h{horizon}"].proba(x), dtype=np.float64)
    import torch

    net = model.net
    net.eval()
    h = list(model.horizons).index(horizon)
    with torch.no_grad():
        logits = net(torch.from_numpy(_sequence(values, mask, kind)))[0][h]
        return np.asarray(torch.softmax(logits, dim=1).numpy(), dtype=np.float64)


def attribute(
    model: Any,
    values: Any,
    mask: Any,
    kind: Any,
    horizon: int,
    ig_steps: int = DEFAULT_IG_STEPS,
    cls: npt.NDArray[np.int64] | None = None,
) -> Attribution:
    """Per-sensor attributions of windows given as (values (B, W, 6), mask (B, W, 5), kind
    one-hot (B, 3)), for ``cls`` or else the predicted alarm class."""
    fam = family(model)
    _require_for(fam)
    if cls is None:
        cls = _alarm_class(probabilities(model, values, mask, kind, horizon))
    if fam in ("xgb", "rf"):
        from twin_ml.explain.trees import saabas_contributions, xgb_contributions

        x = summary_features(values, mask, kind)
        head = model.heads[f"h{horizon}"]
        fn = xgb_contributions if fam == "xgb" else saabas_contributions
        contrib, _ = fn(head, x, cls)
        return Attribution(to_sensors(contrib, summary_feature_sensors()), cls, None, None)
    import torch

    from twin_ml.explain.ig import integrated_gradients

    net = model.net
    net.eval()
    h = list(model.horizons).index(horizon)

    def f(t: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = net(t)[0][h]
        return out

    x_t = torch.from_numpy(_sequence(values, mask, kind))
    attr, err, delta = integrated_gradients(f, x_t, torch.from_numpy(cls), ig_steps)
    per_channel = attr.sum(axis=1)  # over time -> (B, channels)
    return Attribution(to_sensors(per_channel, sequence_channel_sensors()), cls, err, delta)


def explain(
    model: Any,
    values: npt.NDArray[np.floating[Any]],
    mask: npt.NDArray[Any],
    *,
    machine_kind: MachineKind | str,
    sensor_parts: Mapping[str, str],
    horizon: int = 30,
    ig_steps: int = DEFAULT_IG_STEPS,
) -> list[dict[str, Any]]:
    """Explain one window: ``[{sensor, part, weight}]``, largest |weight| first.

    ``values`` (W, 6) are the normalised value channels and ``mask`` (W, 5) the missing flags,
    exactly as ``Normalizer.transform`` produces them; ``sensor_parts`` is the machine's
    ``sensor_parts`` from the line config. Weights are normalised to sum(|w|) = 1.
    """
    kind = np.zeros((1, len(KINDS)), dtype=np.float32)
    kind[0, KINDS.index(MachineKind(machine_kind))] = 1.0
    att = attribute(
        model,
        np.asarray(values, dtype=np.float32)[None],
        np.asarray(mask)[None],
        kind,
        horizon,
        ig_steps,
    )
    return to_parts(normalise(att.sensors)[0], sensor_parts)


__all__ = [
    "Attribution",
    "OpenMPClashError",
    "attribute",
    "explain",
    "family",
    "probabilities",
    "require_alone",
    "to_parts",
]

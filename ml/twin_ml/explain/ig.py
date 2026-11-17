"""Integrated Gradients for the sequence models, implemented in torch (Phase 9, ADR 0026).

IG_i(x) = (x_i - b_i) * integral_0^1 dF(b + a (x - b)) / dx_i da, approximated with the
trapezoid rule on ``steps`` equal intervals (``steps + 1`` points). Completeness: the
attributions sum to F(x) - F(b) up to the approximation error, which is returned per window.

Baseline (fixed in ADR 0024): all value channels 0 (the training mean after normalisation),
every missing-mask channel 0 ("not missing"), the kind channels equal to the input's own, so
the machine kind gets no attribution. F is the logit of the explained class at one horizon.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from twin_ml.features.schema import SIM_SCHEMA, ChannelSchema

KIND_IDX = [SIM_SCHEMA.sequence_channels.index(c) for c in SIM_SCHEMA.kind_channels]


def kind_index(schema: ChannelSchema) -> list[int]:
    return [schema.sequence_channels.index(c) for c in schema.kind_channels]


def baseline_of(x: torch.Tensor, schema: ChannelSchema = SIM_SCHEMA) -> torch.Tensor:
    """Zeros for values and masks; the input's own kind channels."""
    idx = kind_index(schema)
    b = torch.zeros_like(x)
    b[..., idx] = x[..., idx]
    return b


def trapezoid_weights(steps: int) -> npt.NDArray[np.float64]:
    w = np.full(steps + 1, 1.0 / steps)
    w[0] = w[-1] = 0.5 / steps
    return w


def integrated_gradients(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    baseline: torch.Tensor | None = None,
    schema: ChannelSchema = SIM_SCHEMA,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """IG of ``f(x)[i, target[i]]`` for a batch.

    ``f`` maps (B, ...) inputs to (B, classes) scores. Returns (attributions shaped like x,
    completeness error |sum attr - (F(x) - F(b))| per row, F(x) - F(b) per row).
    """
    b = baseline_of(x, schema) if baseline is None else baseline
    rows = torch.arange(x.shape[0])
    alphas = np.linspace(0.0, 1.0, steps + 1)
    weights = trapezoid_weights(steps)
    grad_sum = torch.zeros_like(x, dtype=torch.float64)
    diff = x - b
    for a, w in zip(alphas, weights, strict=True):
        xi = (b + float(a) * diff).detach().requires_grad_(True)
        out = f(xi)[rows, target].sum()
        (g,) = torch.autograd.grad(out, xi)
        grad_sum += float(w) * g.to(torch.float64)
    attr = (diff.to(torch.float64) * grad_sum).detach()
    with torch.no_grad():
        fx = f(x)[rows, target].to(torch.float64)
        fb = f(b)[rows, target].to(torch.float64)
    delta = (fx - fb).numpy()
    total = attr.reshape(attr.shape[0], -1).sum(dim=1).numpy()
    return attr.numpy(), np.abs(total - delta), delta

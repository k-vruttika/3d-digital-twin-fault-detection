"""Losses for the sequence models (torch): weighted cross-entropy / focal, censored ttf.

Weights are per sample (subsampling weight x class weight, ADR 0019); every loss is the
weighted mean over the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

_EPS = 1e-12


def weighted_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, focal_gamma: float = 0.0
) -> torch.Tensor:
    """Weighted mean of the (optionally focal) negative log-likelihood.

    Focal loss multiplies each term by ``(1 - p_true) ** gamma``; gamma 0 is plain
    cross-entropy.
    """
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    if focal_gamma > 0:
        nll = (1.0 - torch.exp(-nll)).pow(focal_gamma) * nll
    return (weight * nll).sum() / weight.sum().clamp_min(_EPS)


def censored_ttf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    censored: torch.Tensor,
    weight: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    """Time-to-fault loss, in units of the cap (pred and target in [0, 1]).

    Known rows: Huber loss on (pred - target). Right-censored rows: only ``true >= target`` is
    known, so a prediction at or above the bound costs nothing and one below it costs the hinge
    ``target - pred``.
    """
    huber = F.huber_loss(pred, target, reduction="none", delta=huber_delta)
    hinge = F.relu(target - pred)
    per_row = torch.where(censored, hinge, huber)
    return (weight * per_row).sum() / weight.sum().clamp_min(_EPS)

"""Losses of the sequence models: censoring, class/sample weighting, focal term."""

from __future__ import annotations

import math

import pytest
import torch

from twin_ml.train.losses import censored_ttf_loss, weighted_cross_entropy


def test_censored_rows_cost_nothing_at_or_above_the_bound() -> None:
    target = torch.tensor([0.4, 0.4, 0.4])
    pred = torch.tensor([0.4, 0.9, 0.1])
    cens = torch.tensor([True, True, True])
    w = torch.ones(3)
    per = [
        censored_ttf_loss(pred[i : i + 1], target[i : i + 1], cens[i : i + 1], w[:1], 0.1).item()
        for i in range(3)
    ]
    assert per[0] == 0.0
    assert per[1] == 0.0  # above the lower bound: consistent with a censored row
    assert per[2] == pytest.approx(0.3)  # hinge below the bound


def test_known_rows_use_huber_both_ways() -> None:
    target = torch.tensor([0.5, 0.5])
    pred = torch.tensor([0.9, 0.1])
    cens = torch.tensor([False, False])
    loss = censored_ttf_loss(pred, target, cens, torch.ones(2), huber_delta=0.1)
    # Huber with delta 0.1 at |err| 0.4: delta * (|err| - delta / 2) = 0.035
    assert loss.item() == pytest.approx(0.035)


def test_censored_loss_is_a_weighted_mean() -> None:
    target = torch.tensor([0.5, 0.5])
    pred = torch.tensor([0.1, 0.5])
    cens = torch.tensor([True, False])
    loss = censored_ttf_loss(pred, target, cens, torch.tensor([3.0, 1.0]), 0.1)
    assert loss.item() == pytest.approx(3 * 0.4 / 4)


def test_weighted_cross_entropy_and_focal() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    y = torch.tensor([0, 2])
    nll0 = -math.log(math.exp(2) / (math.exp(2) + 2))
    nll1 = math.log(3)
    plain = weighted_cross_entropy(logits, y, torch.tensor([1.0, 1.0]))
    assert plain.item() == pytest.approx((nll0 + nll1) / 2, rel=1e-5)
    weighted = weighted_cross_entropy(logits, y, torch.tensor([1.0, 3.0]))
    assert weighted.item() == pytest.approx((nll0 + 3 * nll1) / 4, rel=1e-5)
    focal = weighted_cross_entropy(logits, y, torch.tensor([1.0, 1.0]), focal_gamma=2.0)
    p0, p1 = math.exp(-nll0), 1 / 3
    expected = ((1 - p0) ** 2 * nll0 + (1 - p1) ** 2 * nll1) / 2
    assert focal.item() == pytest.approx(expected, rel=1e-5)
    assert focal.item() < plain.item()  # easy examples are down-weighted

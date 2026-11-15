"""Split by run, never by window, so no run contributes to two splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from twin_ml.config import SPLITS


def split_runs(
    run_ids: Sequence[str], fractions: Mapping[Any, float], seed: int
) -> dict[str, list[str]]:
    """Shuffle the (sorted) run ids with ``seed`` and cut them into train / val / test.

    Sizes: ``round(f * n)`` for train and val, the rest for test; each split gets at least one
    run when there are at least three.
    """
    ids = sorted(set(run_ids))
    if len(ids) != len(run_ids):
        raise ValueError("duplicate run ids")
    n = len(ids)
    if n < len(SPLITS):
        raise ValueError(f"need at least {len(SPLITS)} runs to split, got {n}")
    order = np.random.default_rng(seed).permutation(n)
    n_train = max(1, round(fractions["train"] * n))
    n_val = max(1, round(fractions["val"] * n))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    cuts = {"train": order[:n_train], "val": order[n_train : n_train + n_val]}
    cuts["test"] = order[n_train + n_val :]
    return {name: sorted(ids[i] for i in cuts[name]) for name in SPLITS}

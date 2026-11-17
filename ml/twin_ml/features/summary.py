"""Window summary features for the tree models (RF, XGBoost)."""

from __future__ import annotations

import numpy as np

from twin_ml.features.inputs import FloatArray, MaskArray
from twin_ml.features.schema import SIM_SCHEMA, ChannelSchema

STATS: tuple[str, ...] = ("mean", "std", "min", "max", "slope", "last", "delta")


def summary_names(schema: ChannelSchema = SIM_SCHEMA) -> list[str]:
    names = [f"{c}_{s}" for c in schema.value_channels for s in STATS]
    names += [f"{m}_frac" for m in schema.mask_channels]
    names += list(schema.kind_channels)
    return names


def summary_features(
    values: FloatArray,
    mask: MaskArray,
    kind_onehot: FloatArray,
    schema: ChannelSchema = SIM_SCHEMA,
) -> FloatArray:
    """(B, W, value channels), (B, W, mask channels), (B, kinds) -> (B, features).

    Per value channel: mean, std, min, max, least-squares slope per sample, last value and
    delta (last - first). Per sensor: the fraction of missing samples. Plus the kind one-hot.
    """
    if values.ndim != 3 or values.shape[2] != len(schema.value_channels):
        raise ValueError("values must be (B, W, len(schema.value_channels))")
    b, w, _ = values.shape
    t = np.arange(w, dtype=np.float32) - np.float32((w - 1) / 2.0)
    mean = values.mean(axis=1)
    slope = np.einsum("bwc,w->bc", values - mean[:, None, :], t) / np.float32((t**2).sum())
    stats = np.stack(
        [
            mean,
            values.std(axis=1),
            values.min(axis=1),
            values.max(axis=1),
            slope,
            values[:, -1, :],
            values[:, -1, :] - values[:, 0, :],
        ],
        axis=2,
    ).reshape(b, -1)  # channel-major: all stats of channel 0, then channel 1, ...
    frac = mask.mean(axis=1, dtype=np.float32)
    return np.concatenate([stats, frac, kind_onehot], axis=1).astype(np.float32)

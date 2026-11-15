"""Inference latency at batch size 1 on CPU (features included)."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from twin_ml.config import LatencyConfig
from twin_ml.evaluate.metrics import percentiles_ms


def measure(
    call: Callable[[int], None], n_windows: int, cfg: LatencyConfig, seed: int
) -> dict[str, float]:
    """Time ``call(window_index)`` on random windows after a warm-up; p50/p95 in ms."""
    if n_windows == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_windows, cfg.warmup_calls + cfg.timed_calls)
    for i in idx[: cfg.warmup_calls]:
        call(int(i))
    times: list[float] = []
    for i in idx[cfg.warmup_calls :]:
        t0 = time.perf_counter()
        call(int(i))
        times.append(time.perf_counter() - t0)
    out = percentiles_ms(times)
    out["calls"] = float(cfg.timed_calls)
    out["threads"] = float(cfg.num_threads)
    return out

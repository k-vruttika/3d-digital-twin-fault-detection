"""Reproducible random streams.

Run ``i`` of a batch with base seed ``s`` uses ``SeedSequence(s, spawn_key=(i,))``, which is
identical to ``SeedSequence(s).spawn(n)[i]`` for any ``n``, so results do not depend on the
number of runs or workers. Each run then spawns independent named streams so that, for
example, enabling sensor faults never changes the physics of a run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Append only: spawned children are keyed by index, so adding a stream keeps the others intact.
STREAMS: tuple[str, ...] = ("plan", "sensor_plan", "process", "measure", "ops", "speed_plan")


def run_seed_sequence(base_seed: int, run_index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(base_seed, spawn_key=(run_index,))


def make_streams(base_seed: int, run_index: int) -> dict[str, np.random.Generator]:
    children = run_seed_sequence(base_seed, run_index).spawn(len(STREAMS))
    return {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(STREAMS, children, strict=True)
    }


def get_state(gen: np.random.Generator) -> dict[str, Any]:
    """JSON-serialisable bit-generator state (PCG64 uses plain ints)."""
    return dict(gen.bit_generator.state)


def set_state(gen: np.random.Generator, state: dict[str, Any]) -> None:
    gen.bit_generator.state = state

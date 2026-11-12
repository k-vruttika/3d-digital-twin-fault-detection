"""FakeClock and the reconnect Backoff helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from twin_backend.adapters.base import Backoff
from twin_backend.clock import FakeClock, SystemClock
from twin_common.config.sources import ReconnectConfig


def _cfg(**kw: float) -> ReconnectConfig:
    base: dict[str, float] = {
        "initial_s": 0.5,
        "max_s": 4.0,
        "multiplier": 2.0,
        "jitter": 0.0,
        "seed": 1,
    }
    base.update(kw)
    return ReconnectConfig.model_validate(base)


def test_backoff_exponential_and_capped() -> None:
    b = Backoff(_cfg())
    assert [b.next_delay() for _ in range(6)] == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0]
    b.reset()
    assert b.next_delay() == 0.5


def test_backoff_jitter_bounded_and_seeded() -> None:
    a = Backoff(_cfg(jitter=0.2, seed=3))
    b = Backoff(_cfg(jitter=0.2, seed=3))
    seq_a = [a.next_delay() for _ in range(8)]
    assert seq_a == [b.next_delay() for _ in range(8)]
    for n, d in enumerate(seq_a):
        base = min(4.0, 0.5 * 2.0**n)
        assert base * 0.8 <= d <= base * 1.2


async def test_fake_clock() -> None:
    start = datetime(2026, 3, 1, tzinfo=UTC)
    clock = FakeClock(start)
    assert clock.monotonic() == 0.0
    clock.advance(2.5)
    assert clock.monotonic() == 2.5
    assert clock.now() == start + timedelta(seconds=2.5)
    await clock.sleep(10)  # no auto-advance: time stands still
    assert clock.monotonic() == 2.5
    auto = FakeClock(auto_advance=True)
    await auto.sleep(3)
    assert auto.monotonic() == 3.0
    assert auto.slept == [3]
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)


async def test_system_clock_is_utc() -> None:
    clock = SystemClock()
    assert clock.now().tzinfo is UTC
    t0 = clock.monotonic()
    await clock.sleep(0)
    assert clock.monotonic() >= t0

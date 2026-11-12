"""ChaosAdapter: wraps any adapter and injects seeded transport faults (tests and demos).

Per message, in this order, each with its own probability from ``ChaosConfig``:

1. **drop:** the message disappears.
2. **corrupt:** the payload is damaged so it can never validate: garbage bytes, truncated
   JSON, NaN in every numeric field, or a string in every numeric field.
3. **duplicate:** the message is emitted twice.
4. **delay:** held back ``delay_s`` (on the injected clock) and released with the first
   message that arrives after that, or at end of stream.
5. **reorder:** inserted up to ``reorder_max_displacement`` positions earlier in a small
   output buffer, so it overtakes messages that arrived before it.

The same seed and input give the same output. ``send_command``, ``capabilities`` and
``health`` pass through to the wrapped adapter.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from collections.abc import AsyncIterator, Mapping
from typing import Any

from twin_backend.adapters.base import AdapterHealth, RawMessage, SourceAdapter
from twin_backend.clock import Clock
from twin_common.config.sources import ChaosConfig
from twin_common.contracts import Command, CommandAck
from twin_common.enums import AdapterKind, Capability

CORRUPTIONS = ("garbage", "truncate", "nan", "type")
_GARBAGE = b"\x00\xff\xfe<chaos>"
_TYPE_JUNK = "#corrupt#"


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _as_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        return dict(payload)
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        obj = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def corrupt_payload(payload: Any, variant: str) -> bytes | str | dict[str, Any]:
    data = _as_dict(payload)
    if variant == "garbage" or data is None:
        return _GARBAGE
    if variant == "truncate":
        text = json.dumps(data)
        return text[: max(1, len(text) // 2)]
    if variant == "nan":
        return {k: (math.nan if _is_number(v) else v) for k, v in data.items()}
    return {k: (_TYPE_JUNK if _is_number(v) else v) for k, v in data.items()}


class ChaosAdapter:
    def __init__(self, inner: SourceAdapter, cfg: ChaosConfig, clock: Clock) -> None:
        self.inner = inner
        self.cfg = cfg
        self.clock = clock
        self.kind: AdapterKind = inner.kind
        self.capabilities: frozenset[Capability] = inner.capabilities
        self._rng = random.Random(cfg.seed)
        self.stats = {
            "in": 0,
            "dropped": 0,
            "corrupted": 0,
            "duplicated": 0,
            "delayed": 0,
            "reordered": 0,
        }

    @property
    def time_scale(self) -> float:
        return self.inner.time_scale

    async def start(self) -> None:
        await self.inner.start()

    async def stop(self) -> None:
        await self.inner.stop()

    def health(self) -> AdapterHealth:
        return self.inner.health()

    async def send_command(self, cmd: Command) -> CommandAck:
        return await self.inner.send_command(cmd)

    async def stream(self) -> AsyncIterator[RawMessage]:
        cfg, rng = self.cfg, self._rng
        window: deque[RawMessage] = deque()
        delayed: list[tuple[float, int, RawMessage]] = []
        counter = 0
        # The reorder buffer only exists when reordering is on; otherwise it would add latency.
        hold = cfg.reorder_max_displacement if cfg.reorder > 0 else 0

        def place(msg: RawMessage) -> None:
            if window and rng.random() < cfg.reorder:
                back = rng.randint(1, min(cfg.reorder_max_displacement, len(window)))
                window.insert(len(window) - back, msg)
                self.stats["reordered"] += 1
            else:
                window.append(msg)

        async for msg in self.inner.stream():
            self.stats["in"] += 1
            now = self.clock.monotonic()
            due = [d for d in delayed if d[0] <= now]
            if due:
                delayed[:] = [d for d in delayed if d[0] > now]
                for _, _, held in sorted(due):
                    place(held)
            if rng.random() < cfg.drop:
                self.stats["dropped"] += 1
                continue
            if rng.random() < cfg.corrupt:
                variant = CORRUPTIONS[rng.randrange(len(CORRUPTIONS))]
                msg = msg.with_payload(corrupt_payload(msg.payload, variant))
                self.stats["corrupted"] += 1
            copies = 1
            if rng.random() < cfg.duplicate:
                copies = 2
                self.stats["duplicated"] += 1
            for _ in range(copies):
                if rng.random() < cfg.delay:
                    counter += 1
                    delayed.append((now + cfg.delay_s, counter, msg))
                    self.stats["delayed"] += 1
                    continue
                place(msg)
            while len(window) > hold:
                yield window.popleft()
        for _, _, held in sorted(delayed):
            window.append(held)
        while window:
            yield window.popleft()

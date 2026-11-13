"""Mock server adapter: WebSocket stream with REST fallback and gap backfill.

- **Normal operation:** readings arrive over ``WS {ws_path}`` as
  ``{"reading": ..., "sentAt": ...}``.
- **After every (re)connect:** missed readings are fetched from ``GET {history_path}``
  (``sinceSeq`` = the last seq seen per machine). Overlap is harmless, dedup removes it.
- **While the WebSocket is down:** ``/history`` is polled every ``rest_poll_s`` until the next
  reconnect attempt (exponential backoff).
- **Commands:** ``POST {command_path}``.
- **Scenario injection** (simulation only, ADR 0015): ``POST {scenario_path}``.

Machine ids come from the loaded line config.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from typing import Any

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from twin_backend.adapters.base import BaseAdapter, scenario_ack
from twin_backend.clock import Clock
from twin_common.config.line import LineConfig
from twin_common.config.sources import MockSourceConfig, ReconnectConfig
from twin_common.contracts import Command, CommandAck, ScenarioAck, ScenarioRequest
from twin_common.enums import AckStatus, AdapterKind, Capability
from twin_common.logging import get_logger

log = get_logger(__name__)

_TRANSPORT_ERRORS = (OSError, TimeoutError, WebSocketException, httpx.HTTPError)


def _ws_url(base_url: str, path: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url.removeprefix("https://").rstrip("/") + path
    return "ws://" + base_url.removeprefix("http://").rstrip("/") + path


class MockServerAdapter(BaseAdapter):
    kind = AdapterKind.MOCK
    capabilities = frozenset({Capability.COMMANDS, Capability.HISTORY, Capability.SCENARIO})

    def __init__(
        self,
        cfg: MockSourceConfig,
        line: LineConfig,
        *,
        reconnect: ReconnectConfig,
        command_timeout_s: float,
        clock: Clock,
    ) -> None:
        super().__init__(clock, reconnect)
        self.cfg = cfg
        self.machine_ids = [m.id for m in line.machines]
        self.command_timeout_s = command_timeout_s
        self.ws_url = _ws_url(cfg.base_url, cfg.ws_path)
        self._last_seq: dict[str, int] = {}
        self._http: httpx.AsyncClient | None = None
        self.backfilled = 0
        self.rest_polled = 0

    async def _prepare(self) -> None:
        self._http = httpx.AsyncClient(base_url=self.cfg.base_url, timeout=self.command_timeout_s)

    async def _cleanup(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # -- receiving ---------------------------------------------------------------------------
    def _track(self, reading: Any) -> None:
        if not isinstance(reading, dict):
            return
        mid, seq = reading.get("machineId"), reading.get("seq")
        if (
            isinstance(mid, str)
            and isinstance(seq, int)
            and not isinstance(seq, bool)
            and seq > self._last_seq.get(mid, 0)
        ):
            self._last_seq[mid] = seq

    async def _emit_envelope(self, text: str | bytes) -> None:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self._emit(text, meta={"via": "ws"})  # quarantined by the pipeline
            return
        if isinstance(obj, dict) and isinstance(obj.get("reading"), dict):
            sent_at = None
            if isinstance(obj.get("sentAt"), str):
                try:
                    sent_at = datetime.fromisoformat(obj["sentAt"])
                except ValueError:
                    sent_at = None
            self._track(obj["reading"])
            await self._emit(obj["reading"], sent_at=sent_at, meta={"via": "ws"})
        else:
            await self._emit(text, meta={"via": "ws"})

    async def _backfill(self, via: str) -> int:
        """Fetch readings newer than the last seen seq for every machine."""
        assert self._http is not None
        count = 0
        for mid in self.machine_ids:
            while True:
                since = self._last_seq.get(mid, 0)
                resp = await self._http.get(
                    self.cfg.history_path,
                    params={"machineId": mid, "sinceSeq": since, "limit": self.cfg.history_limit},
                )
                resp.raise_for_status()
                items = resp.json()
                if not isinstance(items, list):
                    break
                for item in items:
                    self._track(item)
                    await self._emit(
                        item if isinstance(item, dict) else json.dumps(item), meta={"via": via}
                    )
                count += len(items)
                if len(items) < self.cfg.history_limit or self._last_seq.get(mid, 0) <= since:
                    break
        return count

    async def _poll_until(self, deadline: float) -> None:
        """REST fallback while the WebSocket is down."""
        while self.clock.monotonic() < deadline and not self._stopping:
            with contextlib.suppress(*_TRANSPORT_ERRORS):
                self.rest_polled += await self._backfill("rest")
            remaining = deadline - self.clock.monotonic()
            await self.clock.sleep(min(self.cfg.rest_poll_s, max(0.0, remaining)))

    async def _run(self) -> None:
        assert self.backoff is not None
        while not self._stopping:
            try:
                async with connect(self.ws_url, open_timeout=self.command_timeout_s) as ws:
                    self._mark_connected()
                    log.info("mock.connected", url=self.ws_url)
                    self.backfilled += await self._backfill("backfill")
                    async for text in ws:
                        await self._emit_envelope(text)
                # Server closed the stream cleanly: treat like a transport error and retry.
                raise ConnectionError("stream closed by server")
            except _TRANSPORT_ERRORS as exc:
                start = self.clock.monotonic()
                await self._poll_until(start + self._begin_reconnect(exc))

    # -- commands ----------------------------------------------------------------------------
    async def send_command(self, cmd: Command) -> CommandAck:
        if self._http is None:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.REJECTED,
                reason="mock adapter is not started",
                ts=self.clock.now(),
            )
        try:
            resp = await self._http.post(
                self.cfg.command_path,
                content=cmd.model_dump_json(),
                headers={"content-type": "application/json"},
                timeout=self.command_timeout_s,
            )
            resp.raise_for_status()
            return CommandAck.model_validate(resp.json())
        except httpx.TimeoutException:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.TIMEOUT,
                reason=f"no ack within {self.command_timeout_s:g} s",
                ts=self.clock.now(),
            )
        except (httpx.HTTPError, ValueError) as exc:
            return CommandAck(
                command_id=cmd.command_id,
                status=AckStatus.REJECTED,
                reason=f"command transport failed: {type(exc).__name__}: {exc}",
                ts=self.clock.now(),
            )

    async def send_scenario(self, req: ScenarioRequest) -> ScenarioAck:
        now = self.clock.now
        if self._http is None:
            return scenario_ack(req, AckStatus.REJECTED, "mock adapter is not started", now())
        try:
            resp = await self._http.post(
                self.cfg.scenario_path,
                content=req.model_dump_json(),
                headers={"content-type": "application/json"},
                timeout=self.command_timeout_s,
            )
            resp.raise_for_status()
            return ScenarioAck.model_validate(resp.json())
        except httpx.TimeoutException:
            return scenario_ack(
                req, AckStatus.TIMEOUT, f"no ack within {self.command_timeout_s:g} s", now()
            )
        except (httpx.HTTPError, ValueError) as exc:
            reason = f"scenario transport failed: {type(exc).__name__}: {exc}"
            return scenario_ack(req, AckStatus.REJECTED, reason, now())

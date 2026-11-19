"""Alert lifecycle (ISA-18.2, ADR 0011): raise, clear, acknowledge, re-raise.

The manager tracks one *condition* per (machine, kind). It never publishes itself: every method
returns the new Alert events, and the twin engine puts them on the bus (the store keeps them
like every other event).

- **Raise:** the condition has held for ``raise_after_s`` of event time (debounce). While the
  alert is active nothing more is emitted (deduplication).
- **Clear:** the condition ended; the alert stays listed until it is acknowledged.
- **Acknowledge:** possible while active or after clearing. Repeating an ack changes nothing.
- **Re-raise:** the condition returns while the alert is cleared but not yet acknowledged:
  the same alert becomes active again. After clear + ack the alert is resolved, and a new
  occurrence gets a new id.

Alert ids are deterministic (``{machine}:{kind}:{raise ts in µs}``), so a replay of the same
data raises the same ids.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from twin_backend.storage.base import to_us
from twin_common.config.engine import AlertKind, AlertsConfig
from twin_common.contracts import Alert


@dataclass(frozen=True, slots=True)
class AckResult:
    alert: Alert | None  # None: unknown or already resolved
    changed: bool  # False: was already acknowledged (idempotent)


class AlertManager:
    def __init__(self, cfg: AlertsConfig) -> None:
        self.cfg = cfg
        self._current: dict[str, Alert] = {}  # unresolved alerts by id (latest rev)
        self._open: dict[tuple[str, AlertKind], str] = {}  # condition -> unresolved alert id
        self._pending: dict[tuple[str, AlertKind], datetime] = {}  # condition first seen

    # -- queries -----------------------------------------------------------------------------
    def get(self, alert_id: str) -> Alert | None:
        return self._current.get(alert_id)

    def current(
        self,
        *,
        active: bool | None = None,
        acknowledged: bool | None = None,
        machine_id: str | None = None,
    ) -> list[Alert]:
        out = [
            a
            for a in self._current.values()
            if (active is None or a.active == active)
            and (acknowledged is None or a.acknowledged == acknowledged)
            and (machine_id is None or a.machine_id == machine_id)
        ]
        return sorted(out, key=lambda a: (a.ts, a.alert_id))

    def open_alert(self, machine_id: str, kind: AlertKind) -> Alert | None:
        alert_id = self._open.get((machine_id, kind))
        return self._current.get(alert_id) if alert_id is not None else None

    # -- condition updates -------------------------------------------------------------------
    def update(
        self, machine_id: str, kind: AlertKind, condition: bool, ts: datetime, message: str
    ) -> list[Alert]:
        key = (machine_id, kind)
        kind_cfg = self.cfg.kinds[kind]
        alert = self.open_alert(machine_id, kind)
        if not condition:
            self._pending.pop(key, None)
            if alert is None or not alert.active:
                return []
            return [self._store(self._next(alert, ts, active=False))]
        if not kind_cfg.enabled:
            return []
        if alert is not None:
            if alert.active:
                return []
            # cleared but not acknowledged: the same alert returns (no debounce: it was real)
            return [self._store(self._next(alert, ts, active=True, message=message))]
        first = self._pending.setdefault(key, ts)
        if ts - first < timedelta(seconds=kind_cfg.raise_after_s):
            return []
        self._pending.pop(key, None)
        new = Alert(
            alert_id=f"{machine_id}:{kind.value}:{to_us(ts)}",
            machine_id=machine_id,
            ts=ts,
            severity=kind_cfg.severity,
            kind=kind.value,
            message=message,
            source=self.cfg.source_of(kind),
        )
        self._open[key] = new.alert_id
        return [self._store(new)]

    def acknowledge(self, alert_id: str, by: str, ts: datetime, at: datetime) -> AckResult:
        """``ts``: event time (the machine's watermark); ``at``: wall-clock time of the ack."""
        alert = self._current.get(alert_id)
        if alert is None:
            return AckResult(None, False)
        if alert.acknowledged:
            return AckResult(alert, False)
        acked = alert.model_copy(
            update={
                "ts": max(ts, alert.ts),
                "acknowledged": True,
                "acked_by": by,
                "acked_at": at,
                "rev": alert.rev + 1,
            }
        )
        return AckResult(self._store(acked), True)

    # -- restore -----------------------------------------------------------------------------
    def load(self, events: Iterable[Alert]) -> int:
        """Rebuild from stored alert events (any order); returns the unresolved count."""
        latest: dict[str, Alert] = {}
        sources = self.cfg.sources()
        for a in events:
            if a.source not in sources:
                continue
            old = latest.get(a.alert_id)
            if old is None or a.rev > old.rev:
                latest[a.alert_id] = a
        self._current.clear()
        self._open.clear()
        for a in sorted(latest.values(), key=lambda x: (x.ts, x.alert_id)):
            self._store(a)
        return len(self._current)

    # -- helpers -----------------------------------------------------------------------------
    def _next(
        self, alert: Alert, ts: datetime, *, active: bool, message: str | None = None
    ) -> Alert:
        update: dict[str, object] = {
            "ts": max(ts, alert.ts),
            "active": active,
            "rev": alert.rev + 1,
        }
        if message is not None:
            update["message"] = message
        return alert.model_copy(update=update)

    def _store(self, alert: Alert) -> Alert:
        try:
            key = (alert.machine_id, AlertKind(alert.kind))
        except ValueError:
            key = None
        if alert.resolved:
            self._current.pop(alert.alert_id, None)
            if key is not None and self._open.get(key) == alert.alert_id:
                del self._open[key]
        else:
            self._current[alert.alert_id] = alert
            if key is not None:
                self._open[key] = alert.alert_id
        return alert

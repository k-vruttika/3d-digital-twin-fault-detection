/**
 * ISA-18.2 alert views (pure functions; ADR 0011 lifecycle).
 *
 * An alert is current at its highest rev. It is resolved (and leaves the panel) once it is both
 * cleared and acknowledged, the same rule as the engine's alert markers. The three visible
 * groups are: active + unacknowledged (needs attention), active + acknowledged (known, still
 * present), cleared + unacknowledged (went away, nobody saw it).
 */
import type { Alert, AlertSeverity } from "../../api/types.ts";

export type AlertGroup = "activeUnacked" | "activeAcked" | "clearedUnacked";

export const GROUP_ORDER: AlertGroup[] = ["activeUnacked", "activeAcked", "clearedUnacked"];

export const GROUP_TITLE: Record<AlertGroup, string> = {
  activeUnacked: "Active, unacknowledged",
  activeAcked: "Active, acknowledged",
  clearedUnacked: "Cleared, unacknowledged",
};

const SEVERITY_RANK: Record<AlertSeverity, number> = { CRITICAL: 0, WARNING: 1, INFO: 2 };

export function isResolved(a: Alert): boolean {
  return !a.active && a.acknowledged;
}

export function groupOf(a: Alert): AlertGroup | null {
  if (isResolved(a)) return null;
  if (a.active) return a.acknowledged ? "activeAcked" : "activeUnacked";
  return "clearedUnacked";
}

/** Apply one lifecycle event to the current alerts (by id, highest rev wins). */
export function applyAlertEvent(current: ReadonlyMap<string, Alert>, a: Alert): Map<string, Alert> {
  const prev = current.get(a.alertId);
  if (prev !== undefined && prev.rev > a.rev) return new Map(current);
  const next = new Map(current);
  if (isResolved(a)) next.delete(a.alertId);
  else next.set(a.alertId, a);
  return next;
}

export interface AlertFilter {
  machineId: string | null;
  severity: AlertSeverity | null;
}

/** Visible alerts per group: severity first, newest first within a severity. */
export function groupAlerts(alerts: Iterable<Alert>, filter: AlertFilter = { machineId: null, severity: null }): Record<AlertGroup, Alert[]> {
  const out: Record<AlertGroup, Alert[]> = { activeUnacked: [], activeAcked: [], clearedUnacked: [] };
  for (const a of alerts) {
    if (filter.machineId !== null && a.machineId !== filter.machineId) continue;
    if (filter.severity !== null && a.severity !== filter.severity) continue;
    const g = groupOf(a);
    if (g !== null) out[g].push(a);
  }
  for (const g of GROUP_ORDER) {
    out[g].sort((x, y) => SEVERITY_RANK[x.severity] - SEVERITY_RANK[y.severity] || Date.parse(y.ts) - Date.parse(x.ts));
  }
  return out;
}

export interface AlertTimes {
  /** When the alert was raised (its rev 0); null if that event is not known. */
  raisedAt: string | null;
  /** When it last went from active to cleared; null while active. */
  clearedAt: string | null;
  /** When it was acknowledged; null if not (yet). */
  ackedAt: string | null;
}

/**
 * Raise, clear and ack times of one alert from its lifecycle events (any order, duplicates
 * allowed). Each Alert event's `ts` is the time of *that* transition, so the current event alone
 * cannot tell when the alert was raised. Only events up to `current.rev` count, so a replay at t
 * never shows a later clear or ack.
 */
export function alertTimes(current: Alert, events: readonly Alert[]): AlertTimes {
  const byRev = new Map<number, Alert>();
  for (const e of [...events, current]) if (e.alertId === current.alertId && e.rev <= current.rev) byRev.set(e.rev, e);
  const ordered = [...byRev.values()].sort((a, b) => a.rev - b.rev);
  const raise = ordered.find((e) => e.rev === 0 && e.active);
  let clearedAt: string | null = null;
  let prevActive: boolean | null = null;
  let firstAck: string | null = null;
  for (const e of ordered) {
    if (!e.active && prevActive !== false) clearedAt = e.ts;
    if (e.active) clearedAt = null;
    prevActive = e.active;
    if (e.acknowledged && firstAck === null) firstAck = e.ts;
  }
  return {
    raisedAt: raise?.ts ?? null,
    clearedAt: current.active ? null : clearedAt,
    ackedAt: current.acknowledged ? (current.ackedAt ?? firstAck) : null,
  };
}

/** Alert events kept per alert id, merged without duplicates (by rev). */
export function mergeAlertEvents(
  current: ReadonlyMap<string, readonly Alert[]>,
  events: readonly Alert[],
): Map<string, Alert[]> {
  const next = new Map<string, Alert[]>([...current].map(([k, v]) => [k, [...v]]));
  for (const e of events) {
    const list = next.get(e.alertId) ?? [];
    if (!list.some((x) => x.rev === e.rev)) {
      list.push(e);
      list.sort((a, b) => a.rev - b.rev);
    }
    next.set(e.alertId, list);
  }
  return next;
}

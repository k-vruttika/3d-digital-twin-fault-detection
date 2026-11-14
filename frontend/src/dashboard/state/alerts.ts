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

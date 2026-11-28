/**
 * Alerts in the three ISA-18.2 groups, filterable by machine and severity. Operators
 * acknowledge; clicking a row selects the machine. The new rev arrives through the stream.
 */
import { useMemo, useState } from "react";

import { ApiError } from "../../api/rest.ts";
import type { AlertSeverity } from "../../api/types.ts";
import { useCanAct, useModel, useRest } from "../context.ts";
import { GROUP_ORDER, GROUP_TITLE, groupAlerts } from "../state/alerts.ts";
import { alertKindView } from "../state/predictions.ts";
import { useStore } from "../state/store.ts";
import { AlertTimesText } from "./AlertTimesText.tsx";

const SEVERITIES: AlertSeverity[] = ["CRITICAL", "WARNING", "INFO"];

/** An alert kind as text; inference kinds look distinct (ADR 0030, DESIGN.md §3.4). */
export function AlertKindBadge({ kind, machineId }: { kind: string; machineId?: string }) {
  const model = useModel();
  const unusual = useStore(model.unusual);
  const v = alertKindView(kind);
  if (kind === "unusual_behaviour" && machineId !== undefined && unusual[machineId]?.intermittent === true) {
    v.label = "unusual behaviour (intermittent)";
  }
  if (v.className === "") return <span>{kind}</span>;
  return (
    <span className={`badge ${v.className} kind-${kind}`} title={v.note ?? undefined} data-testid={`kind-${kind}`}>
      {v.label}
      {kind === "unusual_behaviour" && <span className="small"> · cause unknown</span>}
    </span>
  );
}

export function AlertsPanel() {
  const model = useModel();
  const rest = useRest();
  const alerts = useStore(model.alerts);
  const line = useStore(model.line);
  const { canAct, operator, why } = useCanAct();
  const [machineId, setMachineId] = useState<string | null>(null);
  const [severity, setSeverity] = useState<AlertSeverity | null>(null);
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const groups = useMemo(() => groupAlerts(alerts.values(), { machineId, severity }), [alerts, machineId, severity]);
  const total = GROUP_ORDER.reduce((n, g) => n + groups[g].length, 0);

  const ack = (alertId: string): void => {
    setPending((p) => new Set(p).add(alertId));
    setError(null);
    rest
      .ackAlert(alertId)
      .then((a) => {
        model.applyAlert(a); // the stream sends it too; applying twice is harmless (same rev)
      })
      .catch((e: unknown) => {
        setError(e instanceof ApiError ? `Acknowledge failed: ${String(e.detail)}` : `Acknowledge failed: ${String(e)}`);
      })
      .finally(() => {
        setPending((p) => {
          const n = new Set(p);
          n.delete(alertId);
          return n;
        });
      });
  };

  return (
    <div data-testid="alerts-panel">
      <div className="toolbar">
        <label>
          Machine{" "}
          <select
            value={machineId ?? ""}
            onChange={(e) => {
              setMachineId(e.target.value === "" ? null : e.target.value);
            }}
          >
            <option value="">all</option>
            {(line?.topology.order ?? []).map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Severity{" "}
          <select
            value={severity ?? ""}
            onChange={(e) => {
              setSeverity(e.target.value === "" ? null : (e.target.value as AlertSeverity));
            }}
          >
            <option value="">all</option>
            {SEVERITIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span className="muted small">{total} shown</span>
        {!operator && <span className="muted small">Acknowledging needs an operator key.</span>}
      </div>
      {error !== null && (
        <p className="error small" role="alert">
          {error}
        </p>
      )}
      {GROUP_ORDER.map((g) => (
        <section key={g} className={`alert-group group-${g}`} aria-label={GROUP_TITLE[g]} data-testid={`alert-group-${g}`}>
          <h3>
            {GROUP_TITLE[g]} <span className="muted">({groups[g].length})</span>
          </h3>
          {groups[g].length === 0 ? (
            <p className="muted small">None.</p>
          ) : (
            <table className="alerts">
              <tbody>
                {groups[g].map((a) => (
                  <tr key={a.alertId} data-testid={`alert-${a.alertId}`} className={alertKindView(a.kind).className || undefined}>
                    <td>
                      <span className={`sev sev-${a.severity}`}>{a.severity}</span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="link"
                        onClick={() => {
                          model.select(a.machineId);
                        }}
                        title={`Select ${a.machineId}`}
                      >
                        {a.machineId}
                      </button>
                    </td>
                    <td className="kind">
                      <AlertKindBadge kind={a.kind} machineId={a.machineId} />
                    </td>
                    <td>{a.message}</td>
                    <td className="small muted">
                      <AlertTimesText alert={a} />
                    </td>
                    <td>
                      {operator && !a.acknowledged && (
                        <button
                          type="button"
                          disabled={!canAct || pending.has(a.alertId)}
                          title={why ?? `Acknowledge ${a.kind} on ${a.machineId}`}
                          onClick={() => {
                            ack(a.alertId);
                          }}
                        >
                          {pending.has(a.alertId) ? "…" : "Acknowledge"}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      ))}
    </div>
  );
}

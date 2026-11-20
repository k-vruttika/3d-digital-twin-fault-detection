/**
 * Details of the selected machine: state with reason / root cause, readings with units, data
 * quality, and its active alerts.
 */
import { SENSOR_FIELDS } from "../../api/types.ts";
import { stateLabel } from "../../engine/visuals.ts";
import { useModel } from "../context.ts";
import { displayUnit, formatAge, formatDateTime, formatPct, formatValue } from "../format.ts";
import { groupOf } from "../state/alerts.ts";
import { rootCauseOf } from "../state/model.ts";
import { useStore } from "../state/store.ts";
import { AlertKindBadge } from "./AlertsPanel.tsx";
import { AlertTimesText } from "./AlertTimesText.tsx";
import { PredictionSummary } from "./PredictionPanel.tsx";
import { StateBadge } from "./StateBadge.tsx";

export function DetailsPanel() {
  const model = useModel();
  const id = useStore(model.selection);
  const line = useStore(model.line);
  const machines = useStore(model.machines);
  const lineView = useStore(model.lineView);
  const alerts = useStore(model.alerts);
  const asOf = useStore(model.asOf);
  const conn = useStore(model.connection);
  const mode = useStore(model.modeStore);

  if (id === null || line === null) {
    return (
      <section className="panel details" aria-label="Machine details">
        <h2>Details</h2>
        <p className="muted">Select a machine in the list or the 3D view.</p>
      </section>
    );
  }
  const cfg = line.machines.find((m) => m.id === id);
  const m = machines[id];
  const state = m?.state ?? null;
  const reading = m?.reading ?? null;
  const label = stateLabel({
    state,
    rootCause: rootCauseOf(lineView, id),
    reason: m?.reason ?? null,
    parts: [],
    connected: conn.status === "connected",
  });
  const dq = m?.dataQuality ?? null;
  // Age on the event-time axis: the newest event the screen shows minus this reading.
  const age = reading !== null && asOf !== null ? Date.parse(asOf) - Date.parse(reading.ts) : null;
  const machineAlerts = [...alerts.values()].filter((a) => a.machineId === id && groupOf(a) !== null);

  return (
    <section className="panel details" aria-label={`Details of ${id}`} data-testid="details-panel">
      <h2>
        {id} <span className="muted">· {cfg?.name ?? ""}</span>
      </h2>
      <div className="state-line">
        <StateBadge state={state} />
        <span data-testid="details-state-label">{label}</span>
      </div>
      {m?.reason !== null && m?.reason !== undefined && m.reason !== label && <p className="small muted">Reason: {m.reason}</p>}
      <p className="small muted">Since {formatDateTime(m?.stateTs ?? null)}</p>

      <h3>Prediction</h3>
      <PredictionSummary id={id} />

      <h3>Readings</h3>
      <table className="kv" data-testid="details-readings">
        <tbody>
          {(cfg?.sensors ?? SENSOR_FIELDS).map((s) => {
            const spec = line.sensors[s];
            const v = reading === null ? null : (reading as unknown as Record<string, number | null | undefined>)[s];
            return (
              <tr key={s}>
                <th scope="row">{s}</th>
                <td className={v === null || v === undefined ? "missing" : undefined}>
                  {formatValue(v, s)} <span className="unit">{spec ? displayUnit(spec.unit) : ""}</span>
                </td>
              </tr>
            );
          })}
          <tr>
            <th scope="row">quality</th>
            <td>{reading?.quality ?? "–"}</td>
          </tr>
          <tr>
            <th scope="row">reading</th>
            <td>
              {reading === null ? "–" : `#${String(reading.seq)} · ${formatDateTime(reading.ts)}`}
              {age !== null && age > 0 && <span className="muted"> ({formatAge(age)} older than the newest data)</span>}
            </td>
          </tr>
        </tbody>
      </table>

      <h3>Data quality</h3>
      {dq === null ? (
        <p className="muted small">
          {mode === "replay" ? "Not replayed: data quality is measured over a live window." : "No report yet."}
        </p>
      ) : (
        <div data-testid="details-dq">
          <span className={`badge ${dq.status === "OK" ? "tone-ok" : "tone-warning"}`}>{dq.status}</span>{" "}
          <span className="small muted">
            late {formatPct(dq.latePct)} · quarantined {formatPct(dq.quarantinePct)} · partial {formatPct(dq.partialPct)} ·{" "}
            {dq.samples} samples{dq.linkDown ? " · link down" : ""}
          </span>
          {(dq.reasons ?? []).length > 0 && (
            <ul className="small reasons">
              {(dq.reasons ?? []).map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <h3>Active alerts</h3>
      {machineAlerts.length === 0 ? (
        <p className="muted small">None.</p>
      ) : (
        <ul className="small alert-mini">
          {machineAlerts.map((a) => (
            <li key={a.alertId}>
              <span className={`sev sev-${a.severity}`}>{a.severity}</span>{" "}
              <AlertKindBadge kind={a.kind} machineId={a.machineId} /> {a.message}
              <div className="muted">
                <AlertTimesText alert={a} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

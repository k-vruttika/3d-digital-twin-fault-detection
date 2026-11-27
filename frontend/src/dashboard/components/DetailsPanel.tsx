/**
 * Inspector: the selected machine (DESIGN.md §5). Identity, the state as a large word with its
 * reason / root cause, the +30 s prediction, the readings as a stat grid (value in mono, unit
 * muted; a missing value is a dash, never 0), data quality and the machine's active alerts.
 */
import { SENSOR_FIELDS } from "../../api/types.ts";
import { stateLabel } from "../../engine/visuals.ts";
import { useModel } from "../context.ts";
import { displayUnit, formatAge, formatDateTime, formatPct, formatValue, stateTone } from "../format.ts";
import { groupOf } from "../state/alerts.ts";
import { rootCauseOf } from "../state/model.ts";
import { useStore } from "../state/store.ts";
import { AlertKindBadge } from "./AlertsPanel.tsx";
import { AlertTimesText } from "./AlertTimesText.tsx";
import { PredictionSummary } from "./PredictionPanel.tsx";

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
      <section className="panel details details-empty" aria-label="Machine details">
        <h2 className="micro">Inspector</h2>
        <p className="muted">Select a machine in the line strip or the 3D view.</p>
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
      <header className="identity">
        <h2>{id}</h2>
        <span className="muted">{cfg?.name ?? ""}</span>
      </header>
      <div className="state-block" data-tone={stateTone(state)}>
        <span key={state ?? "none"} className="state-word" data-testid="details-state-label">
          {label}
        </span>
        <p className="state-meta">
          {m?.reason !== null && m?.reason !== undefined && m.reason !== label && <span>Reason: {m.reason} · </span>}
          <span>Since {formatDateTime(m?.stateTs ?? null)}</span>
        </p>
      </div>

      <h3>Prediction · +30 s</h3>
      <PredictionSummary id={id} />

      <h3>Readings</h3>
      <table className="readings" data-testid="details-readings">
        <tbody>
          {(cfg?.sensors ?? SENSOR_FIELDS).map((s) => {
            const spec = line.sensors[s];
            const v = reading === null ? null : (reading as unknown as Record<string, number | null | undefined>)[s];
            const missing = v === null || v === undefined;
            return (
              <tr key={s}>
                <th scope="row">{s}</th>
                <td className={missing ? "missing" : undefined}>
                  <span className="value">{formatValue(v, s)}</span> <span className="unit">{spec ? displayUnit(spec.unit) : ""}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <table className="kv reading-meta">
        <tbody>
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
          <span className={`badge ${dq.status === "OK" ? "tone-ok" : "tone-degraded"}`}>{dq.status}</span>{" "}
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
        <ul className="alert-mini">
          {machineAlerts.map((a) => (
            <li key={a.alertId}>
              <span className={`sev sev-${a.severity}`}>{a.severity}</span> <AlertKindBadge kind={a.kind} machineId={a.machineId} />{" "}
              {a.message}
              <div className="muted alert-mini-times">
                <AlertTimesText alert={a} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/**
 * The active data source: adapter, capabilities, health, ingestion counters (late %,
 * quarantined, partial readings, latency) and per-machine data quality. Operators can switch
 * to another configured source.
 */
import { useState } from "react";

import { ApiError } from "../../api/rest.ts";
import type { AdapterKind } from "../../api/types.ts";
import { useCanAct, useModel, useRest } from "../context.ts";
import { formatDateTime, formatPct } from "../format.ts";
import { useStore } from "../state/store.ts";
import { ConfirmDialog } from "./ConfirmDialog.tsx";

function ms(v: number | null | undefined): string {
  return v === null || v === undefined ? "–" : `${(v * 1000).toFixed(1)} ms`;
}

export function SourcesPanel() {
  const model = useModel();
  const rest = useRest();
  const src = useStore(model.sources);
  const machines = useStore(model.machines);
  const line = useStore(model.line);
  const { canAct, operator, why } = useCanAct();
  const [target, setTarget] = useState<AdapterKind | "">("");
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (src === null) return <p className="muted">Loading source information…</p>;
  const h = src.health;
  const ing = src.ingest;
  const others = src.configured.filter((k) => k !== src.active);

  const doSwitch = (kind: AdapterKind): void => {
    setBusy(true);
    setMessage(null);
    rest
      .switchSource(kind)
      .then((s) => {
        model.sources.set(s);
        setMessage(`Switched to ${String(s.active)}.`);
      })
      .catch((e: unknown) => {
        setMessage(e instanceof ApiError ? `Switch failed: ${String(e.detail)}` : `Switch failed: ${String(e)}`);
      })
      .finally(() => {
        setBusy(false);
      });
  };

  return (
    <div data-testid="sources-panel" className="sources">
      <table className="kv">
        <tbody>
          <tr>
            <th scope="row">active source</th>
            <td data-testid="source-active">
              <strong>{src.active ?? "none"}</strong> {h && <span className={`badge ${h.connected ? "tone-ok" : "tone-fault"}`}>{h.status}</span>}
            </td>
          </tr>
          <tr>
            <th scope="row">capabilities</th>
            <td>{(src.capabilities ?? []).length === 0 ? "–" : (src.capabilities ?? []).join(", ")}</td>
          </tr>
          <tr>
            <th scope="row">messages / reconnects</th>
            <td>
              {h?.messages ?? "–"} / {h?.reconnects ?? "–"}
            </td>
          </tr>
          <tr>
            <th scope="row">last message</th>
            <td>
              {formatDateTime(h?.lastMessageAt ?? null)} <span className="muted">(wall clock)</span>
            </td>
          </tr>
          {h?.lastError && (
            <tr>
              <th scope="row">last error</th>
              <td className="error">{h.lastError}</td>
            </tr>
          )}
          <tr>
            <th scope="row">switches</th>
            <td>{src.switches}</td>
          </tr>
        </tbody>
      </table>

      <h3>Ingestion (since the backend started)</h3>
      {ing ? (
        <table className="kv" data-testid="ingest-stats">
          <tbody>
            <tr>
              <th scope="row">received / published</th>
              <td>
                {ing.messagesIn} / {ing.published}
              </td>
            </tr>
            <tr>
              <th scope="row">late</th>
              <td>
                {ing.late} ({formatPct(ing.latePct)})
              </td>
            </tr>
            <tr>
              <th scope="row">quarantined</th>
              <td>
                {ing.quarantined}
                {ing.quarantined > 0 && (
                  <span className="muted">
                    {" "}
                    (
                    {Object.entries(ing.quarantinedByReason)
                      .filter(([, n]) => n > 0)
                      .map(([r, n]) => `${r} ${String(n)}`)
                      .join(", ")}
                    )
                  </span>
                )}
              </td>
            </tr>
            <tr>
              <th scope="row">partial readings</th>
              <td>
                {ing.partial} ({formatPct(ing.partialPct)})
              </td>
            </tr>
            <tr>
              <th scope="row">duplicates dropped</th>
              <td>{ing.duplicates}</td>
            </tr>
            <tr>
              <th scope="row">latency mean / p95</th>
              <td>
                {ms(ing.latency.meanS)} / {ing.latency.p95S === null || ing.latency.p95S === undefined ? "–" : `≤ ${ms(ing.latency.p95S)}`}{" "}
                <span className="muted">({ing.latency.count} samples)</span>
              </td>
            </tr>
          </tbody>
        </table>
      ) : (
        <p className="muted small">No ingestion counters.</p>
      )}

      <h3>Data quality per machine (rolling window)</h3>
      <table className="kv small">
        <tbody>
          {(line?.topology.order ?? []).map((id) => {
            const dq = machines[id]?.dataQuality ?? null;
            return (
              <tr key={id}>
                <th scope="row">{id}</th>
                <td>
                  {dq === null ? (
                    "–"
                  ) : (
                    <>
                      <span className={`badge ${dq.status === "OK" ? "tone-ok" : "tone-warning"}`}>{dq.status}</span> late {formatPct(dq.latePct)} · quarantined{" "}
                      {formatPct(dq.quarantinePct)} · partial {formatPct(dq.partialPct)}
                    </>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {operator && (
        <>
          <h3>Switch source</h3>
          <div className="form-row">
            <select
              aria-label="Source to switch to"
              value={target}
              onChange={(e) => {
                setTarget(e.target.value as AdapterKind | "");
              }}
            >
              <option value="">choose…</option>
              {others.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={!canAct || target === "" || busy}
              title={why ?? undefined}
              onClick={() => {
                setConfirm(true);
              }}
            >
              Switch
            </button>
          </div>
          {message !== null && (
            <p className="small" role="status">
              {message}
            </p>
          )}
        </>
      )}
      {confirm && target !== "" && (
        <ConfirmDialog
          title={`Switch to ${target}?`}
          confirmLabel="Switch"
          onCancel={() => {
            setConfirm(false);
          }}
          onConfirm={() => {
            setConfirm(false);
            doSwitch(target);
          }}
        >
          Ingestion moves to the {target} source. If it cannot start, the current source keeps running.
        </ConfirmDialog>
      )}
    </div>
  );
}

/**
 * What-if (Phase 11 Part C, ADR 0031): fork a copy of the simulation, apply one change now and
 * see the projected trajectory next to the unchanged one. Projection lines are dashed, the
 * projected states are "ghosts" beside the real state, and operators can apply the change for
 * real ("apply as command", always after a confirmation, through the normal command path).
 * Nothing is sent to a machine by running a what-if.
 */
import { useMemo, useState } from "react";

import { ApiError } from "../../api/rest.ts";
import type { CommandType, MachineState, WhatIfResponse } from "../../api/types.ts";
import { ProjectionChart } from "../charts/ProjectionChart.tsx";
import { useCanAct, useModel, useRest } from "../context.ts";
import { formatTime, stateTone } from "../format.ts";
import { newIdempotencyKey } from "../state/commandFlow.ts";
import { useStore } from "../state/store.ts";
import {
  CHANGES,
  SERIES_FIELDS,
  applyBody,
  buildPayload,
  changeSpec,
  commandText,
  ghostRows,
  hasData,
  horizonLabel,
  outcomeLines,
  projectionSeries,
  type SeriesField,
} from "../state/whatif.ts";
import { ConfirmDialog } from "./ConfirmDialog.tsx";
import { StateBadge } from "./StateBadge.tsx";

function GhostBadge({ state }: { state: MachineState | null }) {
  if (state === null) return <span className="muted">–</span>;
  return <span className={`badge ghost tone-${stateTone(state)}`}>{state}</span>;
}

function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    const detail = typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail);
    if (e.status === 409) return `Not available: ${detail}`;
    if (e.status === 422) return `Refused: ${detail}`;
    return `What-if failed (HTTP ${String(e.status)}): ${detail}`;
  }
  return `What-if failed: ${String(e)}`;
}

type Applied = { kind: "none" } | { kind: "sending" } | { kind: "done"; status: string; reason: string | null } | { kind: "error"; detail: string };

export function WhatIfPanel() {
  const model = useModel();
  const rest = useRest();
  const line = useStore(model.line);
  const selection = useStore(model.selection);
  const sources = useStore(model.sources);
  const mode = useStore(model.modeStore);
  const { canAct, operator, why } = useCanAct();
  const order = useMemo(() => line?.topology.order ?? [], [line]);

  const [picked, setPicked] = useState<string | null>(null);
  const [type, setType] = useState<CommandType>("STOP");
  const [paramText, setParamText] = useState("");
  const [horizonS, setHorizonS] = useState(model.cfg.whatifDefaultHorizonS);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WhatIfResponse | null>(null);
  const [chartMachine, setChartMachine] = useState<string | null>(null);
  const [field, setField] = useState<SeriesField>("vib");
  const [confirm, setConfirm] = useState(false);
  const [applied, setApplied] = useState<Applied>({ kind: "none" });

  const machineId = picked ?? selection ?? order[0] ?? null;
  const spec = changeSpec(type);
  const capable = sources === null || (sources.capabilities ?? []).includes("whatif");
  const blocked = mode === "replay" ? "not available in replay mode: a what-if forks the running simulation" : sources !== null && !capable ? `the active source (${sources.active ?? "?"}) cannot fork a simulation (a simulation-only capability)` : null;

  const run = (): void => {
    const built = buildPayload(machineId, type, paramText, horizonS);
    if (!built.ok) {
      setError(built.error);
      return;
    }
    setError(null);
    setApplied({ kind: "none" });
    setRunning(true);
    rest
      .postWhatif(built.payload)
      .then((r) => {
        setResult(r);
        setChartMachine(r.machineId);
        const firstModel = SERIES_FIELDS.find((f) => !f.model);
        if (r.model === null || r.model === undefined) setField((f) => (SERIES_FIELDS.find((x) => x.field === f)?.model === true ? (firstModel?.field ?? "vib") : f));
      })
      .catch((e: unknown) => {
        setError(errorText(e));
      })
      .finally(() => {
        setRunning(false);
      });
  };

  const apply = (): void => {
    if (result === null) return;
    setApplied({ kind: "sending" });
    rest
      .postCommand({ ...applyBody(result), idempotencyKey: newIdempotencyKey() })
      .then((ack) => {
        setApplied({ kind: "done", status: ack.status, reason: ack.reason ?? null });
      })
      .catch((e: unknown) => {
        setApplied({ kind: "error", detail: e instanceof ApiError ? `HTTP ${String(e.status)}: ${typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail)}` : String(e) });
      });
  };

  const shown = chartMachine ?? result?.machineId ?? null;
  const series = result !== null && shown !== null ? projectionSeries(result, shown, field) : null;
  const fieldSpec = SERIES_FIELDS.find((f) => f.field === field) ?? SERIES_FIELDS[0];
  const rows = result === null ? [] : ghostRows(result, order);
  const hasModel = result?.model !== null && result?.model !== undefined;

  return (
    <div data-testid="whatif-panel">
      <p className="small muted">
        What-if: fork a copy of the simulation now, change one setting, and see the projected trajectory next to what happens if nothing is changed. A
        projection is a simulation, not a promise, and it sends nothing to the machine.
      </p>
      <div className="form-row" data-testid="whatif-form">
        <label>
          Machine{" "}
          <select
            value={machineId ?? ""}
            onChange={(e) => {
              setPicked(e.target.value);
            }}
            data-testid="whatif-machine"
          >
            {order.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <label>
          Change{" "}
          <select
            value={type}
            onChange={(e) => {
              setType(e.target.value as CommandType);
              setParamText("");
            }}
            data-testid="whatif-type"
          >
            {CHANGES.map((c) => (
              <option key={c.type} value={c.type}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        {spec.param !== null && (
          <label>
            {spec.param.label} ({spec.param.unit}){" "}
            <input
              type="number"
              inputMode="decimal"
              step="any"
              value={paramText}
              onChange={(e) => {
                setParamText(e.target.value);
              }}
              data-testid="whatif-param"
            />
          </label>
        )}
        <label>
          Project{" "}
          <select
            value={horizonS}
            onChange={(e) => {
              setHorizonS(Number(e.target.value));
            }}
            data-testid="whatif-horizon"
          >
            {model.cfg.whatifHorizonsS.map((h) => (
              <option key={h} value={h}>
                {String(h)} s
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={run} disabled={running || blocked !== null} title={blocked ?? undefined} data-testid="whatif-run">
          {running ? "Projecting…" : "Run what-if"}
        </button>
      </div>
      {blocked !== null && (
        <p className="small muted" data-testid="whatif-blocked">
          {blocked}.
        </p>
      )}
      {error !== null && (
        <p className="error small" role="alert" data-testid="whatif-error">
          {error}
        </p>
      )}

      {result !== null && series !== null && (
        <div data-testid="whatif-result">
          <h3>
            {commandText(result)} <span className="muted small">· projected {String(result.horizonS)} s from {formatTime(result.asOf)}</span>
          </h3>
          {result.modelNote !== null && result.modelNote !== undefined && (
            <p className="small muted" data-testid="whatif-model-note">
              {result.modelNote}
            </p>
          )}
          {hasModel && (
            <p className="small muted" data-testid="whatif-model">
              Predictions by {result.model}, scored in separate workers; the alarm is the evaluated rule.
            </p>
          )}

          <table className="whatif-ghosts" data-testid="whatif-ghosts">
            <thead>
              <tr>
                <th scope="col">Machine</th>
                <th scope="col">Now</th>
                {(rows[0]?.cells ?? []).map((c) => (
                  <th scope="col" key={c.horizon}>
                    {horizonLabel(c.horizon, result.horizonS)}
                    <div className="small muted">no change → with change</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.machineId}
                  className={[r.changed ? "changed" : "", r.machineId === shown ? "selected" : ""].join(" ").trim() || undefined}
                  onClick={() => {
                    setChartMachine(r.machineId);
                  }}
                  data-testid={`whatif-row-${r.machineId}`}
                >
                  <th scope="row">{r.machineId}</th>
                  <td>
                    <StateBadge state={r.now} />
                  </td>
                  {r.cells.map((c) => (
                    <td key={c.horizon} className={c.differs ? "differs" : undefined} data-testid={`whatif-ghost-${r.machineId}-${c.horizon}`}>
                      <GhostBadge state={c.baseline} /> <span aria-hidden="true">→</span> <GhostBadge state={c.scenario} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <ul className="small" data-testid="whatif-outcome">
            {shown !== null && outcomeLines(result, shown).map((l) => <li key={l}>{l}</li>)}
          </ul>

          <div className="toolbar">
            {SERIES_FIELDS.filter((f) => !f.model || hasModel).map((f) => (
              <button
                key={f.field}
                type="button"
                className={f.field === field ? "tab active" : "tab"}
                aria-pressed={f.field === field}
                onClick={() => {
                  setField(f.field);
                }}
                data-testid={`whatif-field-${f.field}`}
              >
                {f.label}
              </button>
            ))}
            <span className="muted small">{shown}</span>
          </div>
          {hasData(series) ? (
            <ProjectionChart
              testId="whatif-chart"
              data={{
                label: `${shown ?? ""} ${fieldSpec?.label ?? ""}`,
                yTitle: `${fieldSpec?.label ?? ""} (${fieldSpec?.unit ?? ""})`,
                ...series,
                ...(fieldSpec?.model === true ? { yMin: 0, yMax: 1 } : {}),
              }}
            />
          ) : (
            <p className="small muted">No values for this signal in the projection.</p>
          )}

          <div className="form-row">
            {operator ? (
              <button
                type="button"
                disabled={!canAct || applied.kind === "sending"}
                title={why ?? "Sends this change to the machine, after you confirm"}
                onClick={() => {
                  setConfirm(true);
                }}
                data-testid="whatif-apply"
              >
                Apply as command…
              </button>
            ) : (
              <span className="small muted" data-testid="whatif-apply-need-operator">
                Applying a change as a command needs an operator API key.
              </span>
            )}
            {applied.kind === "sending" && <span className="small muted">Sending…</span>}
            {applied.kind === "done" && (
              <span className="small" data-testid="whatif-applied" role="status">
                Command {applied.status}
                {applied.reason !== null ? ` (${applied.reason})` : ""}. This projection started before it; run it again to see the new future.
              </span>
            )}
            {applied.kind === "error" && (
              <span className="small error" role="alert" data-testid="whatif-apply-error">
                {applied.detail}
              </span>
            )}
          </div>
        </div>
      )}

      {confirm && result !== null && (
        <ConfirmDialog
          title="Apply as a command?"
          confirmLabel="Send command"
          danger={result.type === "STOP"}
          onCancel={() => {
            setConfirm(false);
          }}
          onConfirm={() => {
            setConfirm(false);
            apply();
          }}
        >
          <span data-testid="whatif-confirm-text">
            This sends <strong>{commandText(result)}</strong> to the machine now, through the normal command path (safety rules apply). The projection is
            a simulation, not a promise.
          </span>
        </ConfirmDialog>
      )}
    </div>
  );
}

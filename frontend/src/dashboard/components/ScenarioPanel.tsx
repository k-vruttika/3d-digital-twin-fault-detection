/**
 * Scenario injection (SIMULATION ONLY, ADR 0015): shown only when the active source has the
 * `scenario` capability. Injects a fault on a chosen machine; fault types come from the
 * simulator config via `GET /scenario/options`. Recent requests come from the audit trail.
 */
import { useEffect, useMemo, useState } from "react";

import { ApiError } from "../../api/rest.ts";
import type { ScenarioOptionsResponse, ScenarioRecord } from "../../api/types.ts";
import { useCanAct, useModel, useRest } from "../context.ts";
import { formatTime } from "../format.ts";
import { useStore } from "../state/store.ts";

export function ScenarioPanel() {
  const model = useModel();
  const rest = useRest();
  const line = useStore(model.line);
  const selected = useStore(model.selection);
  const asOf = useStore(model.asOf);
  const { canAct, operator, why } = useCanAct();
  const [options, setOptions] = useState<ScenarioOptionsResponse | null>(null);
  const [machineId, setMachineId] = useState<string>("");
  const [faultType, setFaultType] = useState("");
  const [progression, setProgression] = useState("");
  const [last, setLast] = useState<{ record: ScenarioRecord | null; error: string | null } | null>(null);
  const [recent, setRecent] = useState<ScenarioRecord[]>([]);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    rest
      .getScenarioOptions()
      .then((o) => {
        if (!cancelled) setOptions(o);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [rest]);

  const anchor = asOf === null ? null : Math.floor(Date.parse(asOf) / 60_000) * 60_000;
  useEffect(() => {
    let cancelled = false;
    const from = anchor === null ? undefined : new Date(anchor - model.cfg.auditLookbackS * 1000).toISOString();
    rest
      .getScenarios(from === undefined ? {} : { from })
      .then((r) => {
        if (!cancelled) setRecent(r.requests.slice(-6).reverse());
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [rest, anchor, tick, model.cfg.auditLookbackS]);

  const machine = machineId !== "" ? machineId : (selected ?? line?.topology.order[0] ?? "");
  const kind = line?.machines.find((m) => m.id === machine)?.kind ?? null;
  const faults = useMemo(() => (options?.faults ?? []).filter((f) => kind !== null && f.kinds.includes(kind)), [options, kind]);
  const fault = faults.find((f) => f.faultType === faultType) ?? faults[0] ?? null;
  const bounds = options?.progressionS ?? null;
  const progressionValue = progression.trim() === "" ? null : Number(progression);
  const progressionOk =
    progressionValue === null ||
    (Number.isFinite(progressionValue) && progressionValue > 0 && (bounds === null || (progressionValue >= bounds.min && progressionValue <= bounds.max)));

  const inject = (): void => {
    if (fault === null || machine === "") return;
    setLast(null);
    rest
      .postScenario({
        action: "INJECT_FAULT",
        machineId: machine,
        faultType: fault.faultType,
        ...(fault.gradual && progressionValue !== null ? { progressionS: progressionValue } : {}),
      })
      .then((record) => {
        setLast({ record, error: null });
        setTick((t) => t + 1);
      })
      .catch((e: unknown) => {
        setLast({ record: null, error: e instanceof ApiError ? `HTTP ${String(e.status)}: ${String(e.detail)}` : String(e) });
      });
  };

  const lastAck = (last?.record?.acks ?? []).at(-1) ?? null;
  return (
    <div data-testid="scenario-panel">
      <div className="sim-only" role="note">
        SIMULATION ONLY: injects a fault into the simulator behind the active source
      </div>
      {!operator ? (
        <p className="muted small">Injecting needs an operator API key.</p>
      ) : (
        <>
          <div className="form-row">
            <label>
              Machine{" "}
              <select
                value={machine}
                onChange={(e) => {
                  setMachineId(e.target.value);
                  setFaultType("");
                }}
                data-testid="scenario-machine"
              >
                {(line?.topology.order ?? []).map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Fault{" "}
              <select
                value={fault?.faultType ?? ""}
                onChange={(e) => {
                  setFaultType(e.target.value);
                }}
                data-testid="scenario-fault"
              >
                {faults.map((f) => (
                  <option key={f.faultType} value={f.faultType}>
                    {f.faultType}
                    {f.heldOut ? " (held out of training)" : ""}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {fault?.gradual === true && (
            <div className="form-row">
              <label>
                Progression{" "}
                <input
                  type="number"
                  value={progression}
                  min={bounds?.min}
                  max={bounds?.max}
                  placeholder="default"
                  onChange={(e) => {
                    setProgression(e.target.value);
                  }}
                  data-testid="scenario-progression"
                />
              </label>
              <span className="small muted">
                seconds to reach FAULT{bounds ? `, ${String(bounds.min)}–${String(bounds.max)}` : ""} (empty: the simulator&apos;s
                fastest)
              </span>
            </div>
          )}
          {options !== null && options.faults.length === 0 && <p className="small muted">The backend has no simulator config: no fault types.</p>}
          {!progressionOk && <p className="small error">Progression must be within the configured range.</p>}
          <button
            type="button"
            disabled={!canAct || fault === null || !progressionOk}
            title={why ?? undefined}
            onClick={inject}
            data-testid="scenario-inject"
          >
            Inject fault
          </button>
          {last !== null && (
            <div className="result" aria-live="polite" data-testid="scenario-result">
              {last.error !== null ? (
                <span className="error">{last.error}</span>
              ) : (
                <>
                  {last.record?.request.faultType} on {last.record?.request.machineId}:{" "}
                  <strong>{lastAck?.status ?? "sent"}</strong>
                  {lastAck?.reason ? <span className="muted"> ({lastAck.reason})</span> : null}
                </>
              )}
            </div>
          )}
        </>
      )}
      <h3>Recent scenario requests</h3>
      {recent.length === 0 ? (
        <p className="muted small">None in the last hour.</p>
      ) : (
        <ul className="small recent">
          {recent.map((r) => (
            <li key={r.request.requestId}>
              {formatTime(r.request.ts)} {r.request.action} {r.request.faultType ?? r.request.scenario ?? ""} {r.request.machineId ?? ""} by{" "}
              {r.request.requestedBy} → {(r.acks ?? []).at(-1)?.status ?? "…"}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

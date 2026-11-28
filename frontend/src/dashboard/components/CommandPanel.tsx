/**
 * Commands for the selected machine (operator only). STOP is always available (with a
 * confirmation); the others take their parameters within the backend's limits for the
 * machine's kind. Each click gets a new idempotency key; the result follows
 * pending -> ACCEPTED -> EXECUTED / REJECTED (reason) / TIMEOUT. Recent commands come from the
 * audit trail.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError } from "../../api/rest.ts";
import type { CommandLimitsResponse, CommandRecord, CommandType } from "../../api/types.ts";
import { useCanAct, useModel, useRest } from "../context.ts";
import { formatTime } from "../format.ts";
import { isFinal, newIdempotencyKey, reduceFlows, ruleHint, type CommandFlow } from "../state/commandFlow.ts";
import { useStore } from "../state/store.ts";
import { ConfirmDialog } from "./ConfirmDialog.tsx";

const OTHER_COMMANDS: CommandType[] = ["START", "RESET", "SCHEDULE_MAINTENANCE"];

function FlowResult({ flow }: { flow: CommandFlow }) {
  const tone =
    flow.status === "EXECUTED" ? "tone-ok" : flow.status === "pending" || flow.status === "ACCEPTED" ? "tone-idle" : "tone-fault";
  return (
    <div className="result" data-testid="command-result" aria-live="polite">
      <strong>{flow.type}</strong> {flow.machineId}
      {Object.keys(flow.params).length > 0 && <span className="muted"> {JSON.stringify(flow.params)}</span>}{" "}
      <span className={`badge ${tone}`} data-testid="command-status">
        {flow.status === "pending" ? "PENDING" : flow.status}
      </span>
      <div className="small muted">{flow.trail.map((s) => (s === "pending" ? "pending" : s)).join(" → ")}</div>
      {flow.reason !== null && (flow.status === "REJECTED" || flow.status === "TIMEOUT" || flow.status === "ERROR") && (
        <div className="small" data-testid="command-reason">
          Reason: {flow.reason}
        </div>
      )}
    </div>
  );
}

export function CommandPanel() {
  const model = useModel();
  const rest = useRest();
  const id = useStore(model.selection);
  const line = useStore(model.line);
  const machines = useStore(model.machines);
  const asOf = useStore(model.asOf);
  const { canAct, operator, why } = useCanAct();
  const [limits, setLimits] = useState<CommandLimitsResponse | null>(null);
  const [flows, setFlows] = useState<CommandFlow[]>([]);
  const [confirmStop, setConfirmStop] = useState(false);
  const [speed, setSpeed] = useState("");
  const [load, setLoad] = useState("");
  const [recent, setRecent] = useState<CommandRecord[]>([]);
  const [auditTick, setAuditTick] = useState(0);

  useEffect(() => {
    if (!operator) return;
    let cancelled = false;
    rest
      .getCommandLimits()
      .then((l) => {
        // ignore a malformed answer rather than crash; the backend still validates every command
        if (!cancelled && typeof l === "object" && "limits" in l) setLimits(l);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [rest, operator]);

  useEffect(
    () =>
      model.onCommandEvent((e) => {
        if (e.command) {
          const command = e.command;
          setFlows((f) => reduceFlows(f, { kind: "command", command }));
        }
        if (e.ack) {
          const ack = e.ack;
          setFlows((f) => reduceFlows(f, { kind: "ack", ack }));
          if (isFinal(ack.status)) setAuditTick((t) => t + 1);
        }
      }),
    [model],
  );

  // Recent commands of this machine from the audit trail (event time).
  const anchor = asOf === null ? null : Math.floor(Date.parse(asOf) / 60_000) * 60_000;
  useEffect(() => {
    if (id === null || !operator) return;
    let cancelled = false;
    const from = anchor === null ? undefined : new Date(anchor - model.cfg.auditLookbackS * 1000).toISOString();
    rest
      .getCommands({ machines: [id], ...(from === undefined ? {} : { from }) })
      .then((r) => {
        if (!cancelled) setRecent(r.commands.slice(-8).reverse());
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [id, rest, operator, anchor, auditTick, model.cfg.auditLookbackS]);

  const kind = line?.machines.find((m) => m.id === id)?.kind ?? null;
  const kindLimits = kind === null ? null : (limits?.limits[kind] ?? null);
  const state = id === null ? null : (machines[id]?.state ?? null);

  const send = useCallback(
    (type: CommandType, params: Record<string, number> = {}) => {
      if (id === null) return;
      const idempotencyKey = newIdempotencyKey();
      setFlows((f) => reduceFlows(f, { kind: "sent", idempotencyKey, machineId: id, type, params }));
      rest
        .postCommand({ machineId: id, type, params, idempotencyKey })
        .then((ack) => {
          setFlows((f) => reduceFlows(f, { kind: "response", idempotencyKey, ack }));
          setAuditTick((t) => t + 1);
        })
        .catch((e: unknown) => {
          const detail = e instanceof ApiError ? `HTTP ${String(e.status)}: ${typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail)}` : String(e);
          setFlows((f) => reduceFlows(f, { kind: "error", idempotencyKey, detail }));
        });
    },
    [id, rest],
  );

  const numberIn = (raw: string, range: { min: number; max: number } | undefined): number | null => {
    if (raw.trim() === "") return null;
    const v = Number(raw);
    if (!Number.isFinite(v)) return null;
    if (range !== undefined && (v < range.min || v > range.max)) return null;
    return v;
  };
  const speedValue = numberIn(speed, kindLimits?.speedRpm);
  const loadValue = numberIn(load, kindLimits?.loadPct);
  const latest = useMemo(() => flows.find((f) => f.machineId === id) ?? null, [flows, id]);

  if (!operator) {
    return (
      <section className="panel" aria-label="Commands">
        <h2 className="section-head">Commands</h2>
        <p className="muted small" data-testid="commands-need-operator">
          Commands need an operator API key.
        </p>
      </section>
    );
  }
  if (id === null) {
    return (
      <section className="panel" aria-label="Commands">
        <h2 className="section-head">Commands</h2>
        <p className="muted small">Select a machine to send it a command.</p>
      </section>
    );
  }
  const hintFor = (t: CommandType): string | null => ruleHint(limits, t, state);
  const likelyRejected = (["START", "SET_SPEED", "SET_LOAD", "SCHEDULE_MAINTENANCE", "RESET"] as CommandType[]).filter(
    (t) => hintFor(t) !== null,
  );
  return (
    <section className="panel commands" aria-label={`Commands for ${id}`} data-testid="command-panel">
      <h2 className="section-head">
        Commands <span className="muted">· {id}</span>
      </h2>
      {why !== null && <p className="small muted">Disabled: {why}.</p>}
      <div className="form-row command-row">
        <button
          type="button"
          className="danger"
          disabled={!canAct}
          title={why ?? "STOP is always allowed"}
          onClick={() => {
            setConfirmStop(true);
          }}
          data-testid="cmd-stop"
        >
          STOP
        </button>
        {OTHER_COMMANDS.map((t) => (
          <button
            key={t}
            type="button"
            disabled={!canAct}
            title={why ?? hintFor(t) ?? undefined}
            onClick={() => {
              send(t);
            }}
            data-testid={`cmd-${t}`}
          >
            {t === "SCHEDULE_MAINTENANCE" ? "Maintenance" : t === "START" ? "Start" : "Reset"}
          </button>
        ))}
      </div>
      {latest !== null && <FlowResult flow={latest} />}
      <div className="form-row">
        <label>
          Speed{" "}
          <input
            type="number"
            inputMode="decimal"
            value={speed}
            min={kindLimits?.speedRpm.min}
            max={kindLimits?.speedRpm.max}
            step="any"
            aria-describedby="speed-range"
            onChange={(e) => {
              setSpeed(e.target.value);
            }}
            data-testid="input-speed"
          />
        </label>
        <span id="speed-range" className="small muted">
          RPM{kindLimits ? ` (${String(kindLimits.speedRpm.min)}–${String(kindLimits.speedRpm.max)})` : ""}
        </span>
        <button
          type="button"
          disabled={!canAct || speedValue === null}
          title={why ?? hintFor("SET_SPEED") ?? undefined}
          onClick={() => {
            if (speedValue !== null) send("SET_SPEED", { speed: speedValue });
          }}
          data-testid="cmd-SET_SPEED"
        >
          Set speed
        </button>
      </div>
      <div className="form-row">
        <label>
          Load{" "}
          <input
            type="number"
            inputMode="decimal"
            value={load}
            min={kindLimits?.loadPct.min}
            max={kindLimits?.loadPct.max}
            step="any"
            aria-describedby="load-range"
            onChange={(e) => {
              setLoad(e.target.value);
            }}
            data-testid="input-load"
          />
        </label>
        <span id="load-range" className="small muted">
          %{kindLimits ? ` (${String(kindLimits.loadPct.min)}–${String(kindLimits.loadPct.max)})` : ""}
        </span>
        <button
          type="button"
          disabled={!canAct || loadValue === null}
          title={why ?? hintFor("SET_LOAD") ?? undefined}
          onClick={() => {
            if (loadValue !== null) send("SET_LOAD", { load: loadValue });
          }}
          data-testid="cmd-SET_LOAD"
        >
          Set load
        </button>
      </div>
      {(speed !== "" && speedValue === null) || (load !== "" && loadValue === null) ? (
        <p className="small error" role="alert">
          Value outside the allowed range for a {kind ?? "machine"}.
        </p>
      ) : null}
      {likelyRejected.length > 0 && (
        <p className="small muted" data-testid="rule-hints">
          Likely rejected now: {likelyRejected.join(", ")} (hover a button for the reason).
        </p>
      )}

      <h3>Recent commands</h3>
      {recent.length === 0 ? (
        <p className="muted small">None in the last hour.</p>
      ) : (
        <ul className="small recent" data-testid="recent-commands">
          {recent.map((r) => {
            const last = (r.acks ?? []).at(-1);
            return (
              <li key={r.command.commandId}>
                {formatTime(r.command.ts)} <strong>{r.command.type}</strong> by {r.command.issuedBy} →{" "}
                {last?.status ?? "…"}
                {last?.reason ? <span className="muted"> ({last.reason})</span> : null}
              </li>
            );
          })}
        </ul>
      )}

      {confirmStop && (
        <ConfirmDialog
          title={`Stop ${id}?`}
          confirmLabel="STOP"
          danger
          onCancel={() => {
            setConfirmStop(false);
          }}
          onConfirm={() => {
            setConfirmStop(false);
            send("STOP");
          }}
        >
          The machine stops at once. Upstream machines will become BLOCKED and downstream ones STARVED.
        </ConfirmDialog>
      )}
    </section>
  );
}

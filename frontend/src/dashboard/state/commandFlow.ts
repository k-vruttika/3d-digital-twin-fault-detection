/**
 * The life of one command click (pure reducer, unit tested).
 *
 * pending (POST sent) -> ACCEPTED (the stream's `ack`; matched to this click through the
 * stream's `command` frame, which carries our idempotency key and the server's commandId)
 * -> EXECUTED / REJECTED (with the reason) / TIMEOUT, from the POST response or the stream,
 * whichever arrives first. An HTTP error (404, 409, 422, network) ends it as ERROR with the
 * detail. A final status never goes back to ACCEPTED.
 */
import type { AckStatus, Command, CommandAck, CommandLimitsResponse, CommandType, MachineState } from "../../api/types.ts";

export type FlowStatus = "pending" | AckStatus | "ERROR";

export interface CommandFlow {
  idempotencyKey: string;
  machineId: string;
  type: CommandType;
  params: Record<string, number>;
  commandId: string | null;
  status: FlowStatus;
  reason: string | null;
  /** Statuses in the order they were reached (shown as a trail). */
  trail: FlowStatus[];
}

export type FlowEvent =
  | { kind: "sent"; idempotencyKey: string; machineId: string; type: CommandType; params: Record<string, number> }
  | { kind: "command"; command: Command }
  | { kind: "ack"; ack: CommandAck }
  | { kind: "response"; idempotencyKey: string; ack: CommandAck }
  | { kind: "error"; idempotencyKey: string; detail: string };

export const FINAL: readonly FlowStatus[] = ["EXECUTED", "REJECTED", "TIMEOUT", "ERROR"];

export function isFinal(s: FlowStatus): boolean {
  return FINAL.includes(s);
}

function advance(f: CommandFlow, status: FlowStatus, reason: string | null): CommandFlow {
  if (isFinal(f.status)) return f; // the first final status wins
  if (status === f.status) return reason !== null && f.reason === null ? { ...f, reason } : f;
  if (status === "ACCEPTED" && f.status !== "pending") return f;
  return { ...f, status, reason: reason ?? f.reason, trail: [...f.trail, status] };
}

/** Flows newest first, at most `keep` of them. */
export function reduceFlows(flows: readonly CommandFlow[], e: FlowEvent, keep = 10): CommandFlow[] {
  switch (e.kind) {
    case "sent": {
      const flow: CommandFlow = {
          idempotencyKey: e.idempotencyKey,
          machineId: e.machineId,
          type: e.type,
          params: e.params,
          commandId: null,
          status: "pending",
          reason: null,
          trail: ["pending"],
      };
      return [flow, ...flows].slice(0, keep);
    }
    case "command":
      return flows.map((f) =>
        f.idempotencyKey === e.command.idempotencyKey && f.commandId === null ? { ...f, commandId: e.command.commandId } : f,
      );
    case "ack":
      return flows.map((f) => (f.commandId !== null && f.commandId === e.ack.commandId ? advance(f, e.ack.status, e.ack.reason ?? null) : f));
    case "response":
      return flows.map((f) =>
        f.idempotencyKey === e.idempotencyKey
          ? advance({ ...f, commandId: f.commandId ?? e.ack.commandId }, e.ack.status, e.ack.reason ?? null)
          : f,
      );
    case "error":
      return flows.map((f) => (f.idempotencyKey === e.idempotencyKey ? advance(f, "ERROR", e.detail) : f));
  }
}

/** A fresh idempotency key per click (crypto.randomUUID, with a fallback for old runtimes). */
export function newIdempotencyKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `k-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/** The reason a state rule would reject `type` in `state`, if any (the backend decides). */
export function ruleHint(limits: CommandLimitsResponse | null, type: CommandType, state: MachineState | null): string | null {
  if (limits === null || state === null || limits.alwaysAllowed.includes(type)) return null;
  for (const r of limits.rules) {
    if (!r.types.includes(type)) continue;
    if (r.forbiddenStates?.includes(state)) return r.reason;
    if (r.requiredStates && !r.requiredStates.includes(state)) return r.reason;
  }
  return null;
}

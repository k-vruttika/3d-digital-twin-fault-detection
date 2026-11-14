/**
 * Root cause of every STARVED / BLOCKED machine: a port of the backend's
 * `twin_backend/engine/propagation.py` (pure functions).
 *
 * The live dashboard never needs it (the backend sends `line.rootCauses`); it exists so that
 * frontend-only data (the synthetic dev scenes) derives root causes exactly the way the backend
 * does. `rootCause.test.ts` checks it against `__fixtures__/root_causes.json`, which
 * `uv run poe api-docs` generates by running the backend propagation.
 *
 * Rule: RUNNING has no cause; STARVED walks upstream and BLOCKED downstream until a RUNNING
 * neighbour (the bottleneck) or one stopped for another reason (IDLE, WARNING, FAULT,
 * MAINTENANCE); the line ends are SOURCE / SINK. A neighbour that is STALE, OFFLINE or has no
 * state yet is the cause with `unknown: true`. Any other state is its own cause.
 */
import type { LineViewModel, MachineState, RootCauseView } from "../api/types.ts";

export const SOURCE_ID = "SOURCE";
export const SINK_ID = "SINK";

export interface TopologyLike {
  order: readonly string[];
  upstream: Readonly<Record<string, string | null | undefined>>;
  downstream: Readonly<Record<string, string | null | undefined>>;
}

export type StateMap = Readonly<Record<string, MachineState | null | undefined>>;

const LINK_DOWN: readonly (MachineState | null | undefined)[] = ["STALE", "OFFLINE"];
const WAITING: readonly (MachineState | null | undefined)[] = ["STARVED", "BLOCKED"];

export function rootCause(machineId: string, states: StateMap, topo: TopologyLike): RootCauseView {
  const st = states[machineId] ?? null;
  if (st === null) return { cause: null, unknown: true };
  if (st === "RUNNING") return { cause: null, unknown: false };
  if (!WAITING.includes(st)) return { cause: machineId, unknown: LINK_DOWN.includes(st) };
  let current = machineId;
  let depth = 0;
  for (;;) {
    const starved = (states[current] ?? null) === "STARVED";
    const next = (starved ? topo.upstream[current] : topo.downstream[current]) ?? null;
    if (next === null) return { cause: starved ? SOURCE_ID : SINK_ID, unknown: false };
    const nextState = states[next] ?? null;
    if (depth >= topo.order.length || nextState === "RUNNING") return { cause: next, unknown: false };
    if (nextState === null || LINK_DOWN.includes(nextState)) return { cause: next, unknown: true };
    if (!WAITING.includes(nextState)) return { cause: next, unknown: false };
    current = next;
    depth += 1;
  }
}

/** The whole line view: root causes, who each cause impacts, and the line status. */
export function lineView(states: StateMap, topo: TopologyLike): LineViewModel {
  const rootCauses: Record<string, RootCauseView> = {};
  for (const m of topo.order) rootCauses[m] = rootCause(m, states, topo);
  const known = new Set(topo.order);
  const impacted: Record<string, string[]> = {};
  for (const m of topo.order) {
    const rc = rootCauses[m];
    const cause = rc?.cause ?? null;
    if (WAITING.includes(states[m] ?? null) && cause !== null && known.has(cause)) {
      (impacted[cause] ??= []).push(m);
    }
  }
  const values = topo.order.map((m) => states[m] ?? null);
  let status: LineViewModel["status"];
  if (values.some((v) => v === null)) status = "UNKNOWN";
  else if (values.every((v) => v === "RUNNING")) status = "RUNNING";
  else if (values.some((v) => v === "RUNNING")) status = "DEGRADED";
  else status = "STOPPED";
  return { status, rootCauses, impacted };
}

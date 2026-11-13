/**
 * Material-flow topology of the line, from `GET /line` (pure functions).
 *
 * The backend's `topology` (upstream / downstream per machine) is the same one the twin engine
 * uses for root causes, so the frontend never re-derives neighbours differently. The flow order
 * for the layout follows the buffers (a topological sort, config order breaks ties); a line
 * without buffers is laid out in config order (`line_single.yaml`). A cycle falls back to config
 * order with a warning; a branch is kept and reported.
 */
import type { LineBuffer, LineResponse } from "../api/types.ts";

export interface FlowTopology {
  /** Machine ids in layout order. */
  order: string[];
  upstream: ReadonlyMap<string, string | null>;
  downstream: ReadonlyMap<string, string | null>;
  buffers: readonly LineBuffer[];
  warnings: string[];
}

export function flowTopology(line: LineResponse): FlowTopology {
  const configOrder = line.machines.map((m) => m.id);
  const warnings: string[] = [];
  const upstream = new Map<string, string | null>();
  const downstream = new Map<string, string | null>();
  for (const id of configOrder) {
    upstream.set(id, line.topology.upstream[id] ?? null);
    downstream.set(id, line.topology.downstream[id] ?? null);
  }

  const known = new Set(configOrder);
  const buffers = line.buffers.filter((b) => {
    const ok = known.has(b.from) && known.has(b.to);
    if (!ok) warnings.push(`buffer ${b.id} links an unknown machine; it is not shown`);
    return ok;
  });

  let order = configOrder;
  if (buffers.length > 0) {
    const sorted = topologicalOrder(configOrder, buffers);
    if (sorted === null) {
      warnings.push("the buffers form a cycle; machines are laid out in config order");
    } else {
      order = sorted;
    }
    const outDegree = new Map<string, number>();
    const inDegree = new Map<string, number>();
    for (const b of buffers) {
      outDegree.set(b.from, (outDegree.get(b.from) ?? 0) + 1);
      inDegree.set(b.to, (inDegree.get(b.to) ?? 0) + 1);
    }
    for (const [id, n] of outDegree) {
      if (n > 1) warnings.push(`${id} feeds ${String(n)} machines (branch)`);
    }
    for (const [id, n] of inDegree) {
      if (n > 1) warnings.push(`${id} is fed by ${String(n)} machines (merge)`);
    }
  }
  return { order, upstream, downstream, buffers, warnings };
}

/** Kahn's algorithm; ties broken by config order. null if the links contain a cycle. */
export function topologicalOrder(
  ids: readonly string[],
  links: readonly { from: string; to: string }[],
): string[] | null {
  const index = new Map(ids.map((id, i) => [id, i]));
  const inDegree = new Map(ids.map((id) => [id, 0]));
  const next = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const { from, to } of links) {
    if (!index.has(from) || !index.has(to)) continue;
    next.get(from)?.push(to);
    inDegree.set(to, (inDegree.get(to) ?? 0) + 1);
  }
  const byConfig = (a: string, b: string): number => (index.get(a) ?? 0) - (index.get(b) ?? 0);
  const ready = ids.filter((id) => inDegree.get(id) === 0);
  const out: string[] = [];
  while (ready.length > 0) {
    ready.sort(byConfig);
    const id = ready.shift();
    if (id === undefined) break;
    out.push(id);
    for (const to of next.get(id) ?? []) {
      const d = (inDegree.get(to) ?? 0) - 1;
      inDegree.set(to, d);
      if (d === 0) ready.push(to);
    }
  }
  return out.length === ids.length ? out : null;
}

/** The buffer feeding a machine and the one it feeds, if any. */
export function buffersAround(
  topo: FlowTopology,
  machineId: string,
): { input: LineBuffer | null; output: LineBuffer | null } {
  return {
    input: topo.buffers.find((b) => b.to === machineId) ?? null,
    output: topo.buffers.find((b) => b.from === machineId) ?? null,
  };
}

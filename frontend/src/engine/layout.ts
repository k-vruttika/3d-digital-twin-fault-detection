/**
 * Automatic 3D layout of the line (pure functions).
 *
 * Machines are placed along +x in flow order, `gap` metres apart, starting at the origin. After
 * `rowLength` machines the line turns: the next row runs back along -x, `rowSpacing` further in
 * +z (a serpentine), so long lines stay compact. A machine's `layout` override in the line config
 * replaces its position and/or rotation; override positions are world coordinates in the same
 * frame. Buffers sit midway along the link between the machines they connect, turned along it.
 *
 * Links: a link between two machines in line (one row) is straight, from the outlet of one to
 * the inlet of the next. Where the line turns into the next row (or overrides put the machines
 * out of line), the link becomes a transfer path (`linkPath`): out of the outlet, half a gap
 * further, across, and into the inlet. The engine draws it as a transfer track (transfer.ts).
 */
import type { LineResponse, MachineKind } from "../api/types.ts";
import type { Footprint, LayoutConfig } from "./config.ts";
import { flowTopology, type FlowTopology } from "./topology.ts";

export interface Placement {
  id: string;
  kind: MachineKind;
  x: number;
  y: number;
  z: number;
  /** About the vertical axis (radians); 0 = flow along +x. */
  rotationY: number;
  footprint: Footprint;
  row: number;
  overridden: boolean;
}

export interface BufferPlacement {
  id: string;
  from: string;
  to: string;
  capacity: number;
  x: number;
  y: number;
  z: number;
  rotationY: number;
  /** Floor points [x, z] of the link from the outlet of `from` to the inlet of `to`. */
  path: [number, number][];
  /** true when the link is not a straight run within a row (a row turn or an override). */
  turns: boolean;
}

/** Unit vector of the flow direction for a rotation (three.js: +x turns towards -z). */
export function flowDirection(rotationY: number): [number, number] {
  return [Math.cos(rotationY), -Math.sin(rotationY)];
}

const EPS = 1e-6;

/**
 * The floor path [x, z] from the outlet of `a` to the inlet of `b`. Straight when the two are
 * in line facing the same way; otherwise out of the outlet by `gap / 2`, across, and into the
 * inlet from `gap / 2` before it. For a U-turn (opposite directions, the serpentine) both
 * corner points are pushed to the same distance along the flow, so the crossing runs straight
 * across beyond the end of the longer machine.
 */
export function linkPath(a: Placement, b: Placement, gap: number): { path: [number, number][]; turns: boolean } {
  const fa = flowDirection(a.rotationY);
  const fb = flowDirection(b.rotationY);
  const out: [number, number] = [a.x + fa[0] * (a.footprint.length / 2), a.z + fa[1] * (a.footprint.length / 2)];
  const inp: [number, number] = [b.x - fb[0] * (b.footprint.length / 2), b.z - fb[1] * (b.footprint.length / 2)];
  const dot = fa[0] * fb[0] + fa[1] * fb[1];
  const dx = inp[0] - out[0];
  const dz = inp[1] - out[1];
  const cross = fa[0] * dz - fa[1] * dx; // sideways offset of the inlet from the outlet's line
  const ahead = fa[0] * dx + fa[1] * dz;
  if (dot > 1 - EPS && Math.abs(cross) < EPS && ahead >= -EPS) return { path: [out, inp], turns: false };
  const e = gap / 2;
  let p1: [number, number] = [out[0] + fa[0] * e, out[1] + fa[1] * e];
  let p2: [number, number] = [inp[0] - fb[0] * e, inp[1] - fb[1] * e];
  if (dot < -1 + EPS) {
    const along = (p: [number, number]): number => p[0] * fa[0] + p[1] * fa[1];
    const t = Math.max(along(p1), along(p2));
    const push = (p: [number, number]): [number, number] => {
      const k = t - along(p);
      return [p[0] + fa[0] * k, p[1] + fa[1] * k];
    };
    p1 = push(p1);
    p2 = push(p2);
  }
  return { path: [out, p1, p2, inp], turns: true };
}

/** The point halfway along a polyline and the direction (rotationY) of the segment there. */
export function pathMidpoint(path: readonly [number, number][]): { x: number; z: number; rotationY: number } {
  const lengths = path.slice(1).map((p, i) => {
    const q = path[i] ?? p;
    return Math.hypot(p[0] - q[0], p[1] - q[1]);
  });
  const total = lengths.reduce((s, l) => s + l, 0);
  let left = total / 2;
  for (let i = 0; i < lengths.length; i++) {
    const l = lengths[i] ?? 0;
    const a = path[i];
    const b = path[i + 1];
    if (a === undefined || b === undefined) break;
    if (left <= l || i === lengths.length - 1) {
      const k = l > 0 ? left / l : 0;
      return {
        x: a[0] + (b[0] - a[0]) * k,
        z: a[1] + (b[1] - a[1]) * k,
        rotationY: l > 0 ? Math.atan2(-(b[1] - a[1]), b[0] - a[0]) : 0,
      };
    }
    left -= l;
  }
  const only = path[0] ?? [0, 0];
  return { x: only[0], z: only[1], rotationY: 0 };
}

export interface Bounds {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

export interface LineLayout {
  topology: FlowTopology;
  machines: Map<string, Placement>;
  buffers: BufferPlacement[];
  bounds: Bounds;
}

export function footprintOf(kind: string, cfg: LayoutConfig): Footprint {
  return (cfg.footprints as Record<string, Footprint | undefined>)[kind] ?? cfg.defaultFootprint;
}

export function computeLayout(line: LineResponse, cfg: LayoutConfig): LineLayout {
  if (cfg.rowLength < 1) throw new Error("layout.rowLength must be >= 1");
  const topology = flowTopology(line);
  const byId = new Map(line.machines.map((m) => [m.id, m]));
  const machines = new Map<string, Placement>();

  // Rows of machine ids, then x offsets within each row.
  const rows: string[][] = [];
  topology.order.forEach((id, i) => {
    const r = Math.floor(i / cfg.rowLength);
    (rows[r] ??= []).push(id);
  });

  // Row 0 starts with its first machine's back edge at x = 0; every later row starts straight
  // across from the previous row's last machine and runs the other way.
  let turnX = 0;
  rows.forEach((row, r) => {
    const dir = r % 2 === 0 ? 1 : -1;
    let prevX = 0;
    let prevHalf = 0;
    row.forEach((id, j) => {
      const m = byId.get(id);
      if (m === undefined) return;
      const fp = footprintOf(m.kind, cfg);
      const half = fp.length / 2;
      let x: number;
      if (j === 0) x = r === 0 ? half : turnX;
      else x = prevX + dir * (prevHalf + cfg.gap + half);
      machines.set(id, {
        id,
        kind: m.kind,
        x,
        y: 0,
        z: r * cfg.rowSpacing,
        rotationY: dir === 1 ? 0 : Math.PI,
        footprint: fp,
        row: r,
        overridden: false,
      });
      prevX = x;
      prevHalf = half;
    });
    turnX = prevX;
  });

  // Overrides from the line config.
  for (const m of line.machines) {
    const p = machines.get(m.id);
    const o = m.layout;
    if (p === undefined || o === null || o === undefined) continue;
    if (o.position !== null && o.position !== undefined) {
      [p.x, p.y, p.z] = o.position;
      p.overridden = true;
    }
    if (o.rotationDeg !== null && o.rotationDeg !== undefined) {
      p.rotationY = (o.rotationDeg * Math.PI) / 180;
      p.overridden = true;
    }
  }

  const buffers: BufferPlacement[] = [];
  for (const b of topology.buffers) {
    const a = machines.get(b.from);
    const c = machines.get(b.to);
    if (a === undefined || c === undefined) continue;
    const { path, turns } = linkPath(a, c, cfg.gap);
    const mid = pathMidpoint(path);
    buffers.push({
      id: b.id,
      from: b.from,
      to: b.to,
      capacity: b.capacity,
      x: mid.x,
      y: Math.min(a.y, c.y),
      z: mid.z,
      rotationY: mid.rotationY,
      path,
      turns,
    });
  }

  return { topology, machines, buffers, bounds: boundsOf([...machines.values()]) };
}

/** Axis-aligned footprint rectangle of a placement (rotation taken into account). */
export function footprintRect(p: Placement): Bounds {
  const c = Math.abs(Math.cos(p.rotationY));
  const s = Math.abs(Math.sin(p.rotationY));
  const hx = (p.footprint.length * c + p.footprint.width * s) / 2;
  const hz = (p.footprint.length * s + p.footprint.width * c) / 2;
  return { minX: p.x - hx, maxX: p.x + hx, minZ: p.z - hz, maxZ: p.z + hz };
}

export function boundsOf(placements: readonly Placement[]): Bounds {
  if (placements.length === 0) return { minX: 0, maxX: 0, minZ: 0, maxZ: 0 };
  const rects = placements.map(footprintRect);
  return {
    minX: Math.min(...rects.map((r) => r.minX)),
    maxX: Math.max(...rects.map((r) => r.maxX)),
    minZ: Math.min(...rects.map((r) => r.minZ)),
    maxZ: Math.max(...rects.map((r) => r.maxZ)),
  };
}

export function rectsOverlap(a: Bounds, b: Bounds): boolean {
  return a.minX < b.maxX && b.minX < a.maxX && a.minZ < b.maxZ && b.minZ < a.maxZ;
}

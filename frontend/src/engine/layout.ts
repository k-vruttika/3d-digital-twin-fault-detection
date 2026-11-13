/**
 * Automatic 3D layout of the line (pure functions).
 *
 * Machines are placed along +x in flow order, `gap` metres apart, starting at the origin. After
 * `rowLength` machines the line turns: the next row runs back along -x, `rowSpacing` further in
 * +z (a serpentine), so long lines stay compact. A machine's `layout` override in the line config
 * replaces its position and/or rotation; override positions are world coordinates in the same
 * frame. Buffers sit midway between the machines they link, turned along the link.
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
    const dx = c.x - a.x;
    const dz = c.z - a.z;
    buffers.push({
      id: b.id,
      from: b.from,
      to: b.to,
      capacity: b.capacity,
      x: (a.x + c.x) / 2,
      y: Math.min(a.y, c.y),
      z: (a.z + c.z) / 2,
      // rotationY 0 = +x; three.js rotates +x towards -z for positive angles.
      rotationY: dx === 0 && dz === 0 ? 0 : Math.atan2(-dz, dx),
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

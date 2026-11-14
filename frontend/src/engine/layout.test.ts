import { describe, expect, it } from "vitest";

import { cloneLine, createTestEngine, LINE, LINE_SINGLE } from "../test/fakes.ts";
import { DEFAULT_CONFIG, type LayoutConfig } from "./config.ts";
import { computeLayout, footprintRect, linkPath, pathMidpoint, rectsOverlap } from "./layout.ts";
import { generateLine, syntheticMachine } from "./synthetic.ts";

const CFG = DEFAULT_CONFIG.layout;

describe("computeLayout", () => {
  it("lays the full line out along +x in flow order with buffers between neighbours", () => {
    const layout = computeLayout(LINE, CFG);
    const xs = LINE.topology.order.map((id) => layout.machines.get(id)?.x ?? NaN);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
    for (const id of LINE.topology.order) {
      const p = layout.machines.get(id);
      expect(p?.z).toBe(0);
      expect(p?.rotationY).toBe(0);
      expect(p?.overridden).toBe(false);
    }
    // First machine's back edge at x = 0.
    const first = layout.machines.get("CNV-01");
    expect(first && footprintRect(first).minX).toBeCloseTo(0);
    expect(layout.buffers.map((b) => b.id)).toEqual(["B1", "B2", "B3"]);
    for (const b of layout.buffers) {
      const from = layout.machines.get(b.from);
      const to = layout.machines.get(b.to);
      expect(from && to).toBeTruthy();
      if (from === undefined || to === undefined) continue;
      // midway between the outlet and the inlet: the middle of the gap
      expect(b.x).toBeCloseTo((footprintRect(from).maxX + footprintRect(to).minX) / 2);
      // The buffer sits in the gap, not inside either machine.
      expect(b.x).toBeGreaterThan(footprintRect(from).maxX);
      expect(b.x).toBeLessThan(footprintRect(to).minX);
    }
  });

  it("keeps the gap between neighbouring footprints", () => {
    const layout = computeLayout(LINE, CFG);
    const a = layout.machines.get("CNV-01");
    const b = layout.machines.get("CNC-01");
    if (a === undefined || b === undefined) throw new Error("missing placement");
    expect(footprintRect(b).minX - footprintRect(a).maxX).toBeCloseTo(CFG.gap);
  });

  it("handles the single-station line (no buffers)", () => {
    const layout = computeLayout(LINE_SINGLE, CFG);
    expect([...layout.machines.keys()]).toEqual(["CNV-01"]);
    expect(layout.buffers).toEqual([]);
    expect(layout.bounds.maxX - layout.bounds.minX).toBeCloseTo(CFG.footprints.CONVEYOR.length);
  });

  it("adds a machine appended to the config", () => {
    const line = cloneLine();
    line.machines.push(syntheticMachine("CNV-03", "CONVEYOR"));
    line.buffers.push({ id: "B4", from: "CNV-02", to: "CNV-03", capacity: 4 });
    line.topology.order.push("CNV-03");
    line.topology.downstream["CNV-02"] = "CNV-03";
    line.topology.upstream["CNV-03"] = "CNV-02";
    line.topology.downstream["CNV-03"] = null;
    const layout = computeLayout(line, CFG);
    expect(layout.machines.size).toBe(5);
    const prev = layout.machines.get("CNV-02");
    const added = layout.machines.get("CNV-03");
    expect(added && prev && added.x > prev.x).toBe(true);
    expect(layout.buffers.map((b) => b.id)).toContain("B4");
  });

  it("applies position and rotation overrides, and places buffers between the moved ends", () => {
    const line = cloneLine();
    const cnc = line.machines.find((m) => m.id === "CNC-01");
    if (cnc === undefined) throw new Error("no CNC-01");
    cnc.layout = { position: [10, 0, 5], rotationDeg: 90 };
    const insp = line.machines.find((m) => m.id === "INSP-01");
    if (insp === undefined) throw new Error("no INSP-01");
    insp.layout = { position: null, rotationDeg: 180 };
    const layout = computeLayout(line, CFG);
    const p = layout.machines.get("CNC-01");
    expect(p).toMatchObject({ x: 10, y: 0, z: 5, overridden: true });
    expect(p?.rotationY).toBeCloseTo(Math.PI / 2);
    const q = layout.machines.get("INSP-01");
    expect(q?.rotationY).toBeCloseTo(Math.PI);
    expect(q?.z).toBe(0); // position kept
    const b1 = layout.buffers.find((b) => b.id === "B1");
    const cnv = layout.machines.get("CNV-01");
    if (b1 === undefined || cnv === undefined) throw new Error("missing");
    // the moved CNC is out of line: B1 sits halfway along the transfer path between the ends
    expect(b1.turns).toBe(true);
    const mid = pathMidpoint(b1.path);
    expect([b1.x, b1.z]).toEqual([mid.x, mid.z]);
    expect(b1.path[0]).toEqual([footprintRect(cnv).maxX, 0]);
  });

  it("wraps long lines into a serpentine without overlaps (20 machines)", () => {
    const line = generateLine(20);
    const layout = computeLayout(line, CFG);
    expect(layout.machines.size).toBe(20);
    const rows = new Set([...layout.machines.values()].map((p) => p.row));
    expect(rows.size).toBe(Math.ceil(20 / CFG.rowLength));
    const placements = [...layout.machines.values()];
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        const a = placements[i];
        const b = placements[j];
        if (a === undefined || b === undefined) continue;
        expect(rectsOverlap(footprintRect(a), footprintRect(b)), `${a.id} overlaps ${b.id}`).toBe(false);
      }
    }
    // Odd rows run back along -x.
    const row1 = placements.filter((p) => p.row === 1);
    expect(row1.every((p) => Math.abs(p.rotationY - Math.PI) < 1e-9)).toBe(true);
    const xs1 = row1.map((p) => p.x);
    expect(xs1).toEqual([...xs1].sort((a, b) => b - a));
    // The turn: a row starts straight across from the previous row's last machine.
    const lastOfRow0 = placements.filter((p) => p.row === 0).at(-1);
    expect(row1[0]?.x).toBeCloseTo(lastOfRow0?.x ?? NaN);
    expect(layout.buffers).toHaveLength(19);
  });

  it("uses a generic footprint for unknown machine kinds", () => {
    const line = cloneLine();
    const m = line.machines[0];
    if (m === undefined) throw new Error("empty");
    (m as { kind: string }).kind = "ROBOT";
    const layout = computeLayout(line, CFG);
    expect(layout.machines.get(m.id)?.footprint).toEqual(CFG.defaultFootprint);
  });

  it("rejects a row length below 1", () => {
    const cfg: LayoutConfig = { ...CFG, rowLength: 0 };
    expect(() => computeLayout(LINE, cfg)).toThrow(/rowLength/);
  });
});

describe("link paths", () => {
  const cfg = DEFAULT_CONFIG.layout;

  it("same-row links are straight from outlet to inlet, with the buffer in the gap", () => {
    const layout = computeLayout(LINE, cfg);
    for (const b of layout.buffers) {
      expect(b.turns, b.id).toBe(false);
      expect(b.path).toHaveLength(2);
      const a = layout.machines.get(b.from);
      const c = layout.machines.get(b.to);
      if (a === undefined || c === undefined) throw new Error("missing machine");
      expect(b.path[0]?.[0]).toBeCloseTo(a.x + a.footprint.length / 2, 6);
      expect(b.path[1]?.[0]).toBeCloseTo(c.x - c.footprint.length / 2, 6);
      expect(b.x).toBeCloseTo((a.x + a.footprint.length / 2 + c.x - c.footprint.length / 2) / 2, 6);
    }
  });

  it("a 20-machine serpentine gets a transfer path at every row turn, clear of all machines", () => {
    const line = generateLine(20);
    const layout = computeLayout(line, cfg);
    const turning = layout.buffers.filter((b) => b.turns);
    const rows = Math.ceil(20 / cfg.rowLength);
    expect(turning).toHaveLength(rows - 1);
    const rects = [...layout.machines.values()].map(footprintRect);
    for (const b of turning) {
      const a = layout.machines.get(b.from);
      const c = layout.machines.get(b.to);
      if (a === undefined || c === undefined) throw new Error("missing machine");
      expect(a.row).not.toBe(c.row);
      // ends at the machines' outlet / inlet
      expect(b.path[0]?.[1]).toBeCloseTo(a.z, 6);
      expect(b.path.at(-1)?.[1]).toBeCloseTo(c.z, 6);
      // the crossing runs straight across (both corners at the same x), beyond both machines
      const [p1, p2] = [b.path[1], b.path[2]];
      expect(p1?.[0]).toBeCloseTo(p2?.[0] ?? NaN, 6);
      // the buffer tray (and the crossing) overlaps no machine footprint
      const tray = { minX: b.x - 0.6, maxX: b.x + 0.6, minZ: b.z - 0.6, maxZ: b.z + 0.6 };
      for (const r of rects) expect(rectsOverlap(tray, r)).toBe(false);
      expect(Math.abs(b.rotationY)).toBeCloseTo(Math.PI / 2, 6);
    }
  });

  it("linkPath turns for overrides out of line and pathMidpoint halves the length", () => {
    const fp = { length: 2, width: 1 };
    const base = { kind: "CNC" as const, y: 0, footprint: fp, row: 0, overridden: true };
    const a = { ...base, id: "A", x: 0, z: 0, rotationY: 0 };
    const b = { ...base, id: "B", x: 5, z: 3, rotationY: 0 };
    const { path, turns } = linkPath(a, b, 2);
    expect(turns).toBe(true);
    expect(path).toEqual([[1, 0], [2, 0], [3, 3], [4, 3]]);
    expect(pathMidpoint([[0, 0], [4, 0]])).toEqual({ x: 2, z: 0, rotationY: -0 });
  });

  it("the engine draws a transfer track only at row turns", () => {
    const { engine } = createTestEngine({ autoStart: false });
    engine.loadLayout(LINE);
    expect(engine.transferCount).toBe(0);
    engine.loadLayout(generateLine(20));
    expect(engine.transferCount).toBe(Math.ceil(20 / cfg.rowLength) - 1);
    engine.dispose();
    expect(engine.resourceCounts().geometries).toBe(0);
  });
});

import { describe, expect, it } from "vitest";

import { cloneLine, LINE, LINE_SINGLE } from "../test/fakes.ts";
import { DEFAULT_CONFIG, type LayoutConfig } from "./config.ts";
import { computeLayout, footprintRect, rectsOverlap } from "./layout.ts";
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
      expect(b.x).toBeCloseTo((from.x + to.x) / 2);
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
    expect(b1.x).toBeCloseTo((cnv.x + 10) / 2);
    expect(b1.z).toBeCloseTo(2.5);
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

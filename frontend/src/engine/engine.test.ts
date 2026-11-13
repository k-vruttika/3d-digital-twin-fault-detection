/** TwinEngine on a fake renderer: parts, state visuals in the scene graph, buffers, items. */
import { Color } from "three";
import { describe, expect, it } from "vitest";

import type { MachineState, Reading, WsSnapshotData } from "../api/types.ts";
import { cloneLine, createTestEngine, LINE, LINE_SINGLE } from "../test/fakes.ts";
import { DEFAULT_CONFIG } from "./config.ts";
import { generateLine } from "./synthetic.ts";

function reading(machineId: string, state: MachineState, seq: number, speed: number | null = 1000): Reading {
  return {
    schemaVersion: 2,
    machineId,
    seq,
    ts: new Date(Date.parse("2026-01-01T00:00:00Z") + seq * 1000).toISOString(),
    state,
    speed,
    temp: 30,
    vib: 1,
    load: 50,
    current: 10,
    runTime: seq,
    quality: "OK",
  };
}

function snapshotOf(states: Record<string, MachineState>): WsSnapshotData {
  return {
    asOf: "2026-01-01T00:00:05Z",
    machines: Object.fromEntries(
      Object.entries(states).map(([id, state]) => [
        id,
        { machineId: id, state, stateTs: null, reportedState: state, lastReading: reading(id, state, 5), lastStateChange: null },
      ]),
    ),
    line: { status: "DEGRADED", rootCauses: {}, impacted: {} },
    dataQuality: {},
    alerts: [],
  };
}

describe("loadLayout", () => {
  for (const [name, line] of [
    ["line.yaml", LINE],
    ["line_single.yaml", LINE_SINGLE],
  ] as const) {
    it(`every config part of ${name} is an addressable group`, () => {
      const { engine } = createTestEngine({ autoStart: false });
      engine.loadLayout(line);
      for (const m of line.machines) {
        const model = engine.machineModel(m.id);
        expect(model, m.id).toBeDefined();
        for (const part of m.parts) {
          const group = model?.parts.get(part);
          expect(group?.name, `${m.id}.${part}`).toBe(part);
          expect(group?.children.length, `${m.id}.${part} has meshes`).toBeGreaterThan(0);
          expect(model?.partMaterials(part).length, `${m.id}.${part} has its own material`).toBeGreaterThan(0);
        }
      }
      engine.dispose();
    });
  }

  it("creates parts the builders do not know, and machines of unknown kinds", () => {
    const line = cloneLine();
    const cnc = line.machines[1];
    if (cnc === undefined) throw new Error("fixture");
    cnc.parts.push("coolant_pump");
    (line.machines[0] as { kind: string }).kind = "ROBOT";
    const { engine } = createTestEngine({ autoStart: false });
    engine.loadLayout(line);
    expect(engine.machineModel("CNC-01")?.parts.get("coolant_pump")).toBeDefined();
    const robot = engine.machineModel("CNV-01");
    for (const part of line.machines[0]?.parts ?? []) expect(robot?.parts.get(part)).toBeDefined();
    engine.dispose();
  });

  it("builds buffers, items and frames 20 machines", () => {
    const { engine, renderer } = createTestEngine({ autoStart: false });
    engine.loadLayout(generateLine(20));
    expect(engine.currentLayout?.machines.size).toBe(20);
    expect(engine.bufferModel("B19")).toBeDefined();
    expect(renderer.shadowMap.enabled).toBe(false); // above shadowMaxMachines
    engine.loadLayout(LINE);
    expect(renderer.shadowMap.enabled).toBe(true);
    expect(engine.machineModel("M-01")).toBeUndefined(); // old layout gone
    engine.dispose();
  });
});

describe("state visuals in the scene", () => {
  it("maps readings and state changes onto materials, indicators and labels", () => {
    const { engine, clock, labels } = createTestEngine({ autoStart: false });
    engine.loadLayout(LINE);
    engine.applySnapshot(snapshotOf({ "CNV-01": "RUNNING", "CNC-01": "RUNNING", "INSP-01": "RUNNING", "CNV-02": "RUNNING" }));
    engine.tick((clock.now = 0));
    const cnc = engine.machineModel("CNC-01");
    if (cnc === undefined) throw new Error("no CNC");
    expect(cnc.labelText).toBe("RUNNING");
    expect(labels.texts).toContain("RUNNING");

    // RUNNING animates at speed / reference RPM.
    engine.applyReading(reading("CNC-01", "RUNNING", 6, 8000));
    for (let i = 1; i <= 30; i++) engine.tick((clock.now = i * 50));
    expect(cnc.rate).toBeGreaterThan(0.9);

    // OFFLINE: greyed out, stops.
    engine.applyStateChange({ machineId: "CNC-01", state: "OFFLINE", previousState: "STALE", ts: "2026-01-01T00:00:10Z", reason: "no data" });
    for (let i = 31; i <= 90; i++) engine.tick((clock.now = i * 50));
    expect(cnc.rate).toBe(0);
    for (const m of cnc.ownedMaterials()) {
      const c = (m as unknown as { color: Color }).color;
      expect(Math.abs(c.r - c.g) + Math.abs(c.g - c.b), "desaturated").toBeLessThan(0.05);
    }
    expect(cnc.labelText).toBe("OFFLINE");

    // STALE: semi-transparent.
    engine.applyStateChange({ machineId: "CNC-01", state: "STALE", previousState: "RUNNING", ts: "2026-01-01T00:00:08Z", reason: null });
    engine.tick((clock.now += 50));
    const opaque = cnc.ownedMaterials().filter((m) => m.opacity === 1);
    expect(opaque).toEqual([]);
    expect(cnc.ownedMaterials().every((m) => m.transparent)).toBe(true);

    // BLOCKED with root cause: yellow beacon and "BLOCKED by ..." label.
    engine.applyStateChange({ machineId: "CNV-01", state: "BLOCKED", previousState: "RUNNING", ts: "2026-01-01T00:00:09Z", reason: "BLOCKED by CNC-01" });
    engine.applyLine({ status: "DEGRADED", rootCauses: { "CNV-01": { cause: "CNC-01", unknown: false } }, impacted: { "CNC-01": ["CNV-01"] } });
    engine.tick((clock.now += 50));
    const cnv = engine.machineModel("CNV-01");
    expect(cnv?.indicators.beacon).toBe(true);
    expect(cnv?.labelText).toBe("BLOCKED by CNC-01");
    engine.dispose();
  });

  it("WARNING outlines the attributed part; FAULT flashes; data quality and alerts show", () => {
    const { engine, clock } = createTestEngine({ autoStart: false });
    engine.loadLayout(LINE);
    engine.applyReading(reading("CNC-01", "WARNING", 1));
    engine.tick((clock.now = 0));
    const cnc = engine.machineModel("CNC-01");
    if (cnc === undefined) throw new Error("no CNC");
    for (const p of ["spindle", "spindle_motor", "spindle_bearing"]) expect(cnc.outlineVisible(p)).toBe(true);

    engine.setPrediction({
      machineId: "CNC-01",
      ts: "2026-01-01T00:00:01Z",
      horizonS: 30,
      probs: { NORMAL: 0.2, WARNING: 0.5, FAULT: 0.3 },
      calibrated: true,
      conformalSet: ["WARNING"],
      ttfS: 120,
      ttfInterval: [60, 200],
      anomalyScore: 0.4,
      attributions: [{ sensor: "vib", part: "spindle_bearing", weight: 0.7 }],
      reliability: "OK",
      modelVersion: "test",
    });
    engine.tick((clock.now += 16));
    expect(cnc.outlineVisible("spindle_bearing")).toBe(true);
    expect(cnc.outlineVisible("spindle_motor")).toBe(false);

    engine.applyReading(reading("CNC-01", "FAULT", 2, 0));
    let maxEmissive = 0;
    for (let i = 0; i < 40; i++) {
      engine.tick((clock.now += 25));
      const mat = cnc.partMaterials("spindle_bearing")[0]?.material;
      maxEmissive = Math.max(maxEmissive, mat?.emissive.r ?? 0);
      expect(cnc.partMaterials("spindle")[0]?.material.emissive.r).toBe(0);
    }
    expect(maxEmissive).toBeGreaterThan(0.5);
    expect(cnc.outlineVisible("spindle_bearing")).toBe(false);

    engine.applyDataQuality({ machineId: "CNC-01", status: "DEGRADED", samples: 20, latePct: 0, quarantinePct: 9, partialPct: 0, linkDown: false, reasons: ["quarantine"] });
    engine.applyAlert({
      alertId: "a1", machineId: "CNC-01", ts: "2026-01-01T00:00:02Z", severity: "CRITICAL", kind: "machine_fault",
      message: "fault", source: "engine", active: true, acknowledged: false, ackedBy: null, ackedAt: null, rev: 0,
    });
    engine.tick((clock.now += 16));
    expect(cnc.indicators).toMatchObject({ halo: true, marker: true, label: true });
    // Cleared and acknowledged: resolved, the marker goes away.
    engine.applyAlert({
      alertId: "a1", machineId: "CNC-01", ts: "2026-01-01T00:00:03Z", severity: "CRITICAL", kind: "machine_fault",
      message: "fault", source: "engine", active: false, acknowledged: true, ackedBy: "alice", ackedAt: null, rev: 2,
    });
    engine.tick((clock.now += 16));
    expect(cnc.indicators.marker).toBe(false);
    engine.dispose();
  });

  it("infers buffers and items from neighbour states; a lost link shows everything as link lost", () => {
    const { engine, clock } = createTestEngine({ autoStart: false });
    engine.loadLayout(LINE);
    engine.applySnapshot(snapshotOf({ "CNV-01": "BLOCKED", "CNC-01": "IDLE", "INSP-01": "STARVED", "CNV-02": "RUNNING" }));
    engine.tick((clock.now = 0));
    expect(engine.bufferModel("B1")?.status).toBe("FULL");
    expect(engine.bufferModel("B1")?.visibleItems).toBe(10); // FULL: capacity is known
    expect(engine.bufferModel("B2")?.status).toBe("EMPTY");
    expect(engine.bufferModel("B2")?.visibleItems).toBe(0);
    expect(engine.bufferModel("B3")?.status).toBe("FLOWING");
    expect(engine.itemFlow?.visibleCount("CNV-02")).toBeGreaterThan(0);

    engine.setConnected(false);
    engine.tick((clock.now += 16));
    expect(engine.machineModel("CNV-02")?.labelText).toBe("LINK LOST");
    expect(engine.bufferModel("B3")?.status).toBe("UNKNOWN");
    expect(engine.itemFlow?.visibleCount("CNV-02")).toBe(0);
    engine.setConnected(true);
    engine.tick((clock.now += 16));
    expect(engine.machineModel("CNV-02")?.labelText).toBe("RUNNING");
    engine.dispose();
  });

  it("ignores data for unknown machines and stale readings", () => {
    const { engine, clock } = createTestEngine({ autoStart: false });
    engine.loadLayout(LINE);
    engine.applyReading(reading("NOPE", "FAULT", 1));
    engine.applyReading(reading("CNC-01", "RUNNING", 5));
    engine.applyReading(reading("CNC-01", "FAULT", 3)); // older: ignored
    engine.tick((clock.now = 0));
    expect(engine.machineModel("CNC-01")?.labelText).toBe("RUNNING");
    engine.dispose();
  });

  it("labels can be turned off in the config", () => {
    const { engine, clock, labels } = createTestEngine({
      autoStart: false,
      config: { ...DEFAULT_CONFIG, labels: { ...DEFAULT_CONFIG.labels, enabled: false } },
    });
    engine.loadLayout(LINE);
    engine.applyReading(reading("CNC-01", "RUNNING", 1));
    engine.tick((clock.now = 0));
    expect(engine.machineModel("CNC-01")?.indicators.label).toBe(false);
    expect(labels.texts).toEqual([]);
    engine.dispose();
  });
});

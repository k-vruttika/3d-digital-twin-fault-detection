/** Label declutter policy, on-screen sizing and its use in the engine. */
import { Vector3 } from "three";
import { describe, expect, it } from "vitest";

import type { MachineState, Reading } from "../api/types.ts";
import { createTestEngine, LINE } from "../test/fakes.ts";
import { DEFAULT_CONFIG } from "./config.ts";
import {
  bufferLabelVisible,
  labelPixelHeight,
  labelScreenScale,
  machineLabelVisible,
  type LabelFlags,
} from "./labelPolicy.ts";

const AUTO: LabelFlags = { mode: "auto", hovered: false, selected: false };
const NORMAL: MachineState[] = ["RUNNING", "IDLE"];
const S = DEFAULT_CONFIG.labels;

describe("machineLabelVisible", () => {
  it("hides RUNNING and IDLE by default and shows every other state", () => {
    expect(machineLabelVisible("RUNNING", true, NORMAL, AUTO)).toBe(false);
    expect(machineLabelVisible("IDLE", true, NORMAL, AUTO)).toBe(false);
    for (const st of ["WARNING", "FAULT", "STARVED", "BLOCKED", "MAINTENANCE", "STALE", "OFFLINE"] as const) {
      expect(machineLabelVisible(st, true, NORMAL, AUTO), st).toBe(true);
    }
  });
  it("shows NO DATA and LINK LOST", () => {
    expect(machineLabelVisible(null, true, NORMAL, AUTO)).toBe(true);
    expect(machineLabelVisible("RUNNING", false, NORMAL, AUTO)).toBe(true);
  });
  it("shows a normal label when hovered, selected or in mode all", () => {
    expect(machineLabelVisible("RUNNING", true, NORMAL, { ...AUTO, hovered: true })).toBe(true);
    expect(machineLabelVisible("RUNNING", true, NORMAL, { ...AUTO, selected: true })).toBe(true);
    expect(machineLabelVisible("RUNNING", true, NORMAL, { ...AUTO, mode: "all" })).toBe(true);
  });
});

describe("bufferLabelVisible", () => {
  it("shows FULL / EMPTY / UNKNOWN, hides FLOWING unless hovered or mode all", () => {
    for (const st of ["FULL", "EMPTY", "UNKNOWN"] as const) {
      expect(bufferLabelVisible(st, S.bufferStatuses, AUTO), st).toBe(true);
    }
    expect(bufferLabelVisible("FLOWING", S.bufferStatuses, AUTO)).toBe(false);
    expect(bufferLabelVisible("FLOWING", S.bufferStatuses, { ...AUTO, hovered: true })).toBe(true);
    expect(bufferLabelVisible("FLOWING", S.bufferStatuses, { ...AUTO, mode: "all" })).toBe(true);
  });
});

describe("label sizing", () => {
  it("keeps a constant pixel height on normal viewports and never goes below the minimum", () => {
    expect(labelPixelHeight(720, S)).toBe(S.pixelHeight);
    expect(labelPixelHeight(1440, S)).toBe(S.pixelHeight);
    expect(labelPixelHeight(360, S)).toBe(Math.max(S.minPixelHeight, S.pixelHeight / 2));
    expect(labelPixelHeight(10, S)).toBe(S.minPixelHeight);
  });
  it("world height grows linearly with distance, so the on-screen size stays constant", () => {
    const a = labelScreenScale(10, 45, 720, S, false);
    const b = labelScreenScale(20, 45, 720, S, false);
    expect(b.worldHeight / a.worldHeight).toBeCloseTo(2, 6);
    // projected height in px = worldHeight / (2 d tan(fov/2)) * viewport
    const px = (a.worldHeight / (2 * 10 * Math.tan((45 * Math.PI) / 360))) * 720;
    expect(px).toBeCloseTo(S.pixelHeight, 6);
  });
  it("fades between fadeStart and hideDistance; hovered / selected never fade", () => {
    expect(labelScreenScale(S.fadeStart, 45, 720, S, false).opacity).toBe(1);
    const mid = (S.fadeStart + S.hideDistance) / 2;
    expect(labelScreenScale(mid, 45, 720, S, false).opacity).toBeCloseTo(0.5, 6);
    expect(labelScreenScale(S.hideDistance + 1, 45, 720, S, false).opacity).toBe(0);
    expect(labelScreenScale(S.hideDistance + 1, 45, 720, S, true).opacity).toBe(1);
  });
});

function reading(machineId: string, state: MachineState, seq = 1): Reading {
  return {
    schemaVersion: 2,
    machineId,
    seq,
    ts: new Date(Date.parse("2026-01-01T00:00:00Z") + seq * 1000).toISOString(),
    state,
    speed: 1000,
    temp: 30,
    vib: 1,
    load: 50,
    current: 10,
    runTime: seq,
    quality: "OK",
  };
}

describe("engine labels", () => {
  function setup(states: Record<string, MachineState>) {
    const t = createTestEngine({ autoStart: false });
    t.engine.loadLayout(LINE);
    for (const [id, st] of Object.entries(states)) t.engine.applyReading(reading(id, st));
    // close enough that no label fades
    t.engine.focus("CNC-01", 6);
    t.engine.tick(16);
    return t;
  }

  it("shows labels only for non-normal machines and FULL/EMPTY/UNKNOWN buffers by default", () => {
    const { engine } = setup({ "CNV-01": "BLOCKED", "CNC-01": "FAULT", "INSP-01": "STARVED", "CNV-02": "RUNNING" });
    expect(engine.machineModel("CNC-01")?.indicators.label).toBe(true);
    expect(engine.machineModel("CNV-01")?.indicators.label).toBe(true);
    expect(engine.machineModel("CNV-02")?.indicators.label).toBe(false);
    expect(engine.bufferModel("B1")?.status).toBe("FULL");
    expect(engine.bufferModel("B1")?.labelVisible).toBe(true);
    engine.dispose();
  });

  it("hides FLOWING buffer labels and RUNNING machine labels until selected or mode all", () => {
    const { engine } = setup({ "CNV-01": "RUNNING", "CNC-01": "RUNNING", "INSP-01": "RUNNING", "CNV-02": "RUNNING" });
    expect(engine.machineModel("CNC-01")?.indicators.label).toBe(false);
    expect(engine.bufferModel("B1")?.status).toBe("FLOWING");
    expect(engine.bufferModel("B1")?.labelVisible).toBe(false);
    engine.select("CNC-01");
    engine.tick(32);
    expect(engine.machineModel("CNC-01")?.indicators.label).toBe(true);
    expect(engine.machineModel("CNV-02")?.indicators.label).toBe(false);
    engine.setLabelMode("all");
    engine.tick(48);
    expect(engine.machineModel("CNV-02")?.indicators.label).toBe(true);
    expect(engine.bufferModel("B1")?.labelVisible).toBe(true);
    engine.dispose();
  });

  it("scales a label so its on-screen height is constant", () => {
    const { engine } = setup({ "CNC-01": "FAULT" });
    const model = engine.machineModel("CNC-01");
    const label = model?.root.children.find((c) => c.type === "Sprite" && c.visible);
    expect(label).toBeDefined();
    if (label === undefined) return;
    const pos = label.getWorldPosition(new Vector3());
    const d = pos.distanceTo(engine.camera.position);
    const px = (label.scale.y / (2 * d * Math.tan((engine.camera.fov * Math.PI) / 360))) * 600;
    expect(px).toBeCloseTo(labelPixelHeight(600, S), 4);
    engine.dispose();
  });
});

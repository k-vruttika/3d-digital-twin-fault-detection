import { describe, expect, it } from "vitest";

import type { MachineState } from "../api/types.ts";
import { DEFAULT_CONFIG, type VisualConfig } from "./config.ts";
import {
  animationRate,
  atRiskParts,
  bufferStatus,
  highestActiveSeverity,
  machineVisual,
  smoothRate,
  stateLabel,
  type VisualInput,
} from "./visuals.ts";

const CFG = DEFAULT_CONFIG;
const PARTS = ["spindle", "spindle_motor", "spindle_bearing"];
const ALL_STATES: MachineState[] = [
  "RUNNING",
  "IDLE",
  "WARNING",
  "FAULT",
  "STARVED",
  "BLOCKED",
  "MAINTENANCE",
  "OFFLINE",
  "STALE",
];

function visual(state: MachineState | null, extra: Partial<VisualInput> = {}, cfg: VisualConfig = CFG) {
  return machineVisual({ state, parts: PARTS, ...extra }, cfg);
}

describe("machineVisual: one row per state", () => {
  const expected: Record<MachineState, Partial<ReturnType<typeof visual>>> = {
    RUNNING: { tone: "running", animate: true, opacity: 1, grey: 0, beacon: "none" },
    IDLE: { tone: "neutral", animate: false, opacity: 1, grey: 0, beacon: "none" },
    WARNING: { tone: "warning", animate: true, outlineParts: PARTS, outlineColor: CFG.colors.warning },
    FAULT: { tone: "fault", animate: false, flashParts: PARTS, flashColor: CFG.colors.fault },
    STARVED: { tone: "waiting", animate: false, beacon: "waiting" },
    BLOCKED: { tone: "waiting", animate: false, beacon: "waiting" },
    MAINTENANCE: { tone: "maintenance", animate: false, beacon: "maintenance", tint: CFG.colors.maintenance },
    STALE: { tone: "stale", animate: false, opacity: CFG.staleOpacity, grey: 0 },
    OFFLINE: { tone: "offline", animate: false, opacity: 1, grey: CFG.offlineGrey },
  };
  for (const state of ALL_STATES) {
    it(state, () => {
      const v = visual(state);
      expect(v).toMatchObject(expected[state]);
      expect(v.label?.text).toBe(state);
      expect(v.label?.tone).toBe(v.tone);
      if (state !== "WARNING") expect(v.outlineParts).toEqual([]);
      if (state !== "FAULT") expect(v.flashParts).toEqual([]);
      if (!["STARVED", "BLOCKED", "MAINTENANCE"].includes(state)) expect(v.beacon).toBe("none");
    });
  }

  it("no state yet: neutral, labelled NO DATA", () => {
    expect(visual(null)).toMatchObject({ tone: "neutral", animate: false, label: { text: "NO DATA" } });
  });
});

describe("machineVisual: overlays", () => {
  it("DEGRADED data quality adds the halo for any state", () => {
    for (const state of ALL_STATES) {
      expect(visual(state, { dataQuality: "DEGRADED" }).dqBadge).toBe(true);
      expect(visual(state, { dataQuality: "OK" }).dqBadge).toBe(false);
    }
  });

  it("the alert marker shows the highest active severity", () => {
    const alerts = [
      { severity: "WARNING" as const, active: true },
      { severity: "CRITICAL" as const, active: false },
      { severity: "INFO" as const, active: true },
    ];
    expect(visual("RUNNING", { alerts }).alertMarker).toBe("WARNING");
    expect(visual("RUNNING", { alerts: [] }).alertMarker).toBeNull();
    expect(highestActiveSeverity([{ severity: "CRITICAL", active: true }, ...alerts])).toBe("CRITICAL");
  });

  it("a lost link looks STALE, but OFFLINE stays grey", () => {
    expect(visual("RUNNING", { connected: false })).toMatchObject({
      tone: "stale",
      animate: false,
      opacity: CFG.staleOpacity,
      label: { text: "LINK LOST" },
    });
    expect(visual("OFFLINE", { connected: false })).toMatchObject({ tone: "offline", label: { text: "OFFLINE" } });
  });

  it("labels can be turned off", () => {
    const off: VisualConfig = { ...CFG, labels: { ...CFG.labels, enabled: false } };
    for (const state of ALL_STATES) expect(visual(state, {}, off).label).toBeNull();
  });
});

describe("at-risk part", () => {
  it("uses the top attribution naming a machine part", () => {
    const attributions = [
      { part: "spindle_motor", weight: 0.2 },
      { part: "spindle_bearing", weight: -0.6 },
      { part: "not_a_part", weight: 0.9 },
    ];
    expect(atRiskParts(attributions, PARTS)).toEqual(["spindle_bearing"]);
    expect(visual("WARNING", { attributions }).outlineParts).toEqual(["spindle_bearing"]);
    expect(visual("FAULT", { attributions }).flashParts).toEqual(["spindle_bearing"]);
  });

  it("falls back to the whole machine without a usable attribution", () => {
    expect(atRiskParts(null, PARTS)).toEqual(PARTS);
    expect(atRiskParts([{ part: "elsewhere", weight: 1 }], PARTS)).toEqual(PARTS);
  });
});

describe("stateLabel", () => {
  it("names the root cause of STARVED / BLOCKED", () => {
    expect(stateLabel({ state: "BLOCKED", parts: [], rootCause: { cause: "CNC-01", unknown: false } })).toBe(
      "BLOCKED by CNC-01",
    );
    expect(stateLabel({ state: "STARVED", parts: [], rootCause: { cause: "SOURCE" } })).toBe("STARVED by SOURCE");
    expect(stateLabel({ state: "STARVED", parts: [], rootCause: { cause: "CNC-01", unknown: true } })).toBe(
      "STARVED by CNC-01 (unknown)",
    );
  });

  it("falls back to the StateChange reason, then to the bare state", () => {
    expect(stateLabel({ state: "BLOCKED", parts: [], reason: "BLOCKED by INSP-01" })).toBe("BLOCKED by INSP-01");
    expect(stateLabel({ state: "BLOCKED", parts: [], reason: "something else" })).toBe("BLOCKED");
    expect(stateLabel({ state: "BLOCKED", parts: [], rootCause: { cause: null } })).toBe("BLOCKED");
  });
});

describe("animation rate", () => {
  it("is speed / reference RPM, clamped, null when missing", () => {
    expect(animationRate(1450, "CONVEYOR", CFG)).toBeCloseTo(1);
    expect(animationRate(4000, "CNC", CFG)).toBeCloseTo(0.5);
    expect(animationRate(1e9, "CNC", CFG)).toBe(CFG.maxRate);
    expect(animationRate(-5, "CNC", CFG)).toBe(0);
    expect(animationRate(null, "CNC", CFG)).toBeNull();
    expect(animationRate(undefined, "CNC", CFG)).toBeNull();
    expect(animationRate(Number.NaN, "CNC", CFG)).toBeNull();
    expect(animationRate(1000, "ROBOT", CFG)).toBeCloseTo(1000 / CFG.defaultReferenceRpm);
  });

  it("smooths towards the target", () => {
    const r = smoothRate(0, 1, 0.1, CFG.rateResponse);
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(1);
    expect(smoothRate(0, 1, 100, CFG.rateResponse)).toBeCloseTo(1);
    expect(smoothRate(0.5, 1, 0, CFG.rateResponse)).toBe(0.5);
  });
});

describe("bufferStatus", () => {
  const cases: [MachineState | null, MachineState | null, string][] = [
    ["BLOCKED", "IDLE", "FULL"],
    ["BLOCKED", "STARVED", "FULL"],
    ["RUNNING", "STARVED", "EMPTY"],
    ["FAULT", "STARVED", "EMPTY"],
    ["RUNNING", "RUNNING", "FLOWING"],
    ["WARNING", "BLOCKED", "FLOWING"],
    ["STALE", "RUNNING", "UNKNOWN"],
    ["RUNNING", "OFFLINE", "UNKNOWN"],
    [null, "RUNNING", "UNKNOWN"],
    ["BLOCKED", "STALE", "UNKNOWN"],
  ];
  for (const [from, to, status] of cases) {
    it(`${String(from)} -> ${String(to)}: ${status}`, () => {
      const s = bufferStatus("A", from, "B", to);
      expect(s.status).toBe(status);
      expect(s.why.length).toBeGreaterThan(10);
    });
  }

  it("explains the inference with the machine ids", () => {
    expect(bufferStatus("CNV-01", "BLOCKED", "CNC-01", "IDLE").why).toContain("CNV-01 is BLOCKED");
    expect(bufferStatus("CNV-01", "RUNNING", "CNC-01", "STARVED").why).toContain("CNC-01 is STARVED");
    expect(bufferStatus("CNV-01", "RUNNING", "CNC-01", "OFFLINE").why).toContain("CNC-01 has no live data");
  });
});

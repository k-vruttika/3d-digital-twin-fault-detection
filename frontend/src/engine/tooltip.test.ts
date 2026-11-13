import { describe, expect, it } from "vitest";

import type { Reading } from "../api/types.ts";
import { LINE } from "../test/fakes.ts";
import { DEFAULT_CONFIG } from "./config.ts";
import { bufferTooltip, formatAge, formatSensor, machineTooltip, MISSING } from "./tooltip.ts";

const CNC = LINE.machines[1];
if (CNC === undefined) throw new Error("fixture without CNC");

const READING: Reading = {
  schemaVersion: 2,
  machineId: "CNC-01",
  seq: 5,
  ts: "2026-01-01T00:00:08Z",
  state: "RUNNING",
  speed: 7949.97,
  temp: 30.94,
  vib: null,
  load: 33.2,
  current: 10.01,
  runTime: 8,
  quality: "SUSPECT",
};

describe("formatSensor", () => {
  it("formats with the catalog unit and per-sensor decimals", () => {
    expect(formatSensor("speed", 7949.97, LINE.sensors.speed, DEFAULT_CONFIG)).toBe("7950 RPM");
    expect(formatSensor("temp", 30.94, LINE.sensors.temp, DEFAULT_CONFIG)).toBe("30.9 °C");
    expect(formatSensor("vib", 1.234, LINE.sensors.vib, DEFAULT_CONFIG)).toBe("1.23 mm/s");
  });

  it("shows – for missing values", () => {
    for (const v of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(formatSensor("temp", v, LINE.sensors.temp, DEFAULT_CONFIG)).toBe(MISSING);
    }
    expect(MISSING).toBe("–");
  });
});

describe("machineTooltip", () => {
  it("lists every sensor, quality and reading age; missing values as –", () => {
    const t = machineTooltip(
      {
        machine: CNC,
        stateLabel: "RUNNING",
        reading: READING,
        sensors: LINE.sensors,
        dataQuality: null,
        alerts: [],
        nowMs: Date.parse("2026-01-01T00:00:10Z"),
      },
      DEFAULT_CONFIG,
    );
    expect(t.title).toBe("CNC-01 · CNC machining station");
    expect(t.subtitle).toBe("RUNNING");
    const rows = Object.fromEntries(t.rows.map((r) => [r.label, r.value]));
    expect(rows).toEqual({
      speed: "7950 RPM",
      temp: "30.9 °C",
      vib: "–",
      load: "33.2 %",
      current: "10.0 A",
      quality: "SUSPECT",
      reading: "2 s ago",
    });
    expect(t.notes).toEqual([]);
  });

  it("shows – everywhere before the first reading, plus DQ reasons and active alerts", () => {
    const t = machineTooltip(
      {
        machine: CNC,
        stateLabel: "NO DATA",
        reading: null,
        sensors: LINE.sensors,
        dataQuality: {
          machineId: "CNC-01",
          status: "DEGRADED",
          samples: 30,
          latePct: 0,
          quarantinePct: 12,
          partialPct: 0,
          linkDown: false,
          reasons: ["quarantine 12.0 % > 5.0 %"],
        },
        alerts: [
          {
            alertId: "a",
            machineId: "CNC-01",
            ts: "2026-01-01T00:00:00Z",
            severity: "CRITICAL",
            kind: "machine_fault",
            message: "CNC-01 is in FAULT",
            source: "engine",
            active: true,
            acknowledged: false,
            ackedBy: null,
            ackedAt: null,
            rev: 0,
          },
          {
            alertId: "b",
            machineId: "CNC-01",
            ts: "2026-01-01T00:00:00Z",
            severity: "WARNING",
            kind: "machine_warning",
            message: "cleared",
            source: "engine",
            active: false,
            acknowledged: false,
            ackedBy: null,
            ackedAt: null,
            rev: 1,
          },
        ],
        nowMs: 0,
      },
      DEFAULT_CONFIG,
    );
    expect(t.rows.every((r) => r.value === MISSING)).toBe(true);
    expect(t.notes).toEqual(["Data quality DEGRADED: quarantine 12.0 % > 5.0 %", "CRITICAL: CNC-01 is in FAULT"]);
  });
});

describe("formatAge / bufferTooltip", () => {
  it("formats ages", () => {
    const now = Date.parse("2026-01-01T01:00:00Z");
    expect(formatAge("2026-01-01T00:59:30Z", now)).toBe("30 s ago");
    expect(formatAge("2026-01-01T00:30:00Z", now)).toBe("30 min ago");
    expect(formatAge("2025-12-31T22:00:00Z", now)).toBe("3.0 h ago");
    expect(formatAge(null, now)).toBe(MISSING);
    expect(formatAge("garbage", now)).toBe(MISSING);
  });

  it("explains a buffer's inferred status and never shows a count", () => {
    const t = bufferTooltip(
      { id: "B1", from: "CNV-01", to: "CNC-01", capacity: 10 },
      { status: "FLOWING", why: "parts are passing" },
    );
    expect(t.subtitle).toBe("FLOWING");
    expect(t.notes[0]).toBe("parts are passing");
    expect(t.notes[1]).toMatch(/no buffer level/);
    expect(t.rows).toEqual([{ label: "capacity", value: "10 parts" }]);
  });
});

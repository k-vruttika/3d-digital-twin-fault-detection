import type { Chart } from "chart.js";
import { describe, expect, it } from "vitest";

import { alert } from "../../test/dashboard.tsx";
import { alertMarkers, annotationPlugin, gapRuns, markerKindLabel, type AnnotationStyle, type ChartMarker } from "./annotations.ts";
import type { ChartPoint } from "./series.ts";

const T0 = Date.parse("2026-01-01T00:00:00Z");

describe("alertMarkers", () => {
  it("marks each alert of the machine once, at its raise time, in its kind's colour", () => {
    const raise = alert({ alertId: "f", machineId: "CNC-01", ts: "2026-01-01T00:00:10Z", kind: "machine_fault" });
    const acked = { ...raise, rev: 1, acknowledged: true, ts: "2026-01-01T00:00:40Z" };
    const unusual = alert({ alertId: "u", machineId: "CNC-01", ts: "2026-01-01T00:00:20Z", kind: "unusual_behaviour", severity: "WARNING" });
    const predicted = alert({ alertId: "p", machineId: "CNC-01", ts: "2026-01-01T00:00:30Z", kind: "predicted_fault", severity: "WARNING" });
    const other = alert({ alertId: "o", machineId: "CNV-01", ts: "2026-01-01T00:00:15Z" });
    const events = new Map([
      ["f", [raise, acked]],
      ["o", [other]],
    ]);
    const m = alertMarkers([acked, unusual, predicted, other], events, "CNC-01");
    expect(m.map((x) => x.alertId)).toEqual(["f", "u", "p"]);
    expect(m[0]).toMatchObject({ x: T0 + 10_000, tone: "fault", dashed: false, label: "FAULT 00:00:10" });
    expect(m[1]).toMatchObject({ tone: "unusual", dashed: false, label: "UNUSUAL BEHAVIOUR 00:00:20" });
    expect(m[2]).toMatchObject({ tone: "fault", dashed: true, label: "PREDICTED FAULT 00:00:30" });
  });

  it("labels kinds in plain words", () => {
    expect(markerKindLabel("machine_warning")).toBe("WARNING");
    expect(markerKindLabel("line_stoppage")).toBe("LINE STOPPAGE");
    expect(markerKindLabel("model_drift")).toBe("MODEL DRIFT");
  });
});

describe("gapRuns", () => {
  it("spans each run of nulls from the last value before to the first after", () => {
    const pts: ChartPoint[] = [
      { x: 0, y: 1 },
      { x: 10, y: null },
      { x: 20, y: null },
      { x: 30, y: 2 },
      { x: 40, y: null },
      { x: 50, y: 3 },
    ];
    expect(gapRuns(pts)).toEqual([
      [0, 30],
      [30, 50],
    ]);
    expect(gapRuns([{ x: 0, y: 1 }])).toEqual([]);
  });
});

/** A canvas context that records the calls the plugin makes. */
function recordingChart(points: ChartPoint[], range: [number, number]) {
  const calls: { op: string; args: unknown[]; fill?: unknown; dash?: number[] }[] = [];
  let fillStyle: unknown = "";
  let dash: number[] = [];
  const ctx = new Proxy(
    {},
    {
      get(_t, prop: string) {
        if (prop === "measureText") return (s: string) => ({ width: s.length * 6 });
        if (prop === "setLineDash")
          return (d: number[]) => {
            dash = d;
          };
        if (prop === "fillStyle") return fillStyle;
        return (...args: unknown[]) => {
          calls.push({ op: prop, args, fill: fillStyle, dash });
        };
      },
      set(_t, prop: string, v: unknown) {
        if (prop === "fillStyle") fillStyle = v;
        return true;
      },
    },
  );
  const [min, max] = range;
  const px = (v: number) => 100 + ((v - min) / (max - min)) * 400;
  const chart = {
    ctx,
    chartArea: { left: 100, right: 500, top: 10, bottom: 110 },
    scales: { x: { min, max, getPixelForValue: px } },
    data: { datasets: [{ data: points }] },
  } as unknown as Chart;
  return { chart, calls };
}

const STYLE: AnnotationStyle = {
  hatch: "hatch",
  noDataText: "nodata-text",
  font: "Barlow",
  tones: { fault: "fault-fill", warning: "warning-fill", maintenance: "m-fill", unusual: "unusual-fill", degraded: "d-fill" },
  inks: { fault: "fault-ink", warning: "warning-ink", maintenance: "m-ink", unusual: "unusual-ink", degraded: "d-ink" },
};

describe("annotationPlugin", () => {
  it("hatches a gap and labels it NO DATA only when it is wide enough", () => {
    const wide = recordingChart(
      [
        { x: 0, y: 1 },
        { x: 50, y: null },
        { x: 100, y: 1 },
      ],
      [0, 100],
    );
    const plugin = annotationPlugin(() => ({ markers: [], style: STYLE }));
    (plugin.beforeDatasetsDraw as (c: Chart) => void)(wide.chart);
    expect(wide.calls.some((c) => c.op === "fillText" && c.args[0] === "NO DATA")).toBe(true);

    const narrow = recordingChart(
      [
        { x: 0, y: 1 },
        { x: 1, y: null },
        { x: 2, y: 1 },
      ],
      [0, 100],
    );
    (plugin.beforeDatasetsDraw as (c: Chart) => void)(narrow.chart);
    expect(narrow.calls.some((c) => c.op === "stroke")).toBe(true);
    expect(narrow.calls.some((c) => c.op === "fillText")).toBe(false);
  });

  it("draws markers inside the range only: dashed when predicted, label in the status ink", () => {
    const markers: ChartMarker[] = [
      { x: 20, tone: "fault", dashed: false, label: "FAULT 00:00:20", alertId: "a" },
      { x: 80, tone: "warning", dashed: true, label: "PREDICTED WARNING 00:01:20", alertId: "b" },
      { x: 500, tone: "unusual", dashed: false, label: "OUT OF RANGE", alertId: "c" },
    ];
    const { chart, calls } = recordingChart([{ x: 0, y: 1 }], [0, 100]);
    (annotationPlugin(() => ({ markers, style: STYLE })).afterDatasetsDraw as (c: Chart) => void)(chart);
    const labels = calls.filter((c) => c.op === "fillText");
    expect(labels.map((c) => c.args[0])).toEqual(["FAULT 00:00:20", "PREDICTED WARNING 00:01:20"]);
    expect(labels.map((c) => c.fill)).toEqual(["fault-ink", "warning-ink"]);
    const strokes = calls.filter((c) => c.op === "stroke");
    expect(strokes.map((c) => c.dash)).toEqual([[], [4, 3]]);
  });
});

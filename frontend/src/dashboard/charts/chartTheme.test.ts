import { describe, expect, it } from "vitest";

import { chartTheme, withAlpha } from "./chartTheme.ts";
import { lastValueIndex, lineChartConfig } from "./lineChart.ts";

describe("chart styling helpers (DESIGN.md §8)", () => {
  it("withAlpha turns a token hex into rgba and leaves other colours alone", () => {
    expect(withAlpha("#ECEEF0", 0.85)).toBe("rgba(236, 238, 240, 0.85)");
    expect(withAlpha("gray", 0.5)).toBe("gray");
  });

  it("lastValueIndex finds the newest real value, skipping trailing gaps", () => {
    expect(lastValueIndex([{ x: 1, y: 2 }, { x: 2, y: null }, { x: 3, y: 4 }, { x: 4, y: null }])).toBe(2);
    expect(lastValueIndex([{ x: 1, y: null }])).toBe(-1);
    expect(lastValueIndex([])).toBe(-1);
  });

  it("sensor charts: gaps stay gaps, horizontal grid only, the axis keeps its unit's case", () => {
    const cfg = lineChartConfig({ label: "vib", yTitle: "vib (mm/s)", points: [{ x: 1, y: 1 }, { x: 2, y: null }] }, chartTheme());
    for (const ds of cfg.data.datasets) expect((ds as { spanGaps?: boolean }).spanGaps).toBe(false);
    const scales = cfg.options?.scales as Record<string, { grid?: { display?: boolean }; title?: { text?: string } }>;
    expect(scales.x?.grid?.display).toBe(false);
    expect(scales.y?.title?.text).toBe("vib (mm/s)");
  });
});

import { describe, expect, it } from "vitest";

import { HUD_GAP_PX, hudInsets, type Box } from "./hudInsets.ts";

const canvas: Box = { top: 100, bottom: 700, left: 0, right: 1400 };

describe("hudInsets", () => {
  it("reserves the annunciator above and the line strip below, plus a gap", () => {
    const ann: Box = { top: 116, bottom: 186, left: 500, right: 900 };
    const strip: Box = { top: 600, bottom: 692, left: 8, right: 1392 };
    expect(hudInsets(canvas, [ann], [strip])).toEqual({ top: 86 + HUD_GAP_PX, right: 0, bottom: 100 + HUD_GAP_PX, left: 0 });
  });

  it("ignores a HUD element that does not cover the canvas (the docked strip on narrow screens)", () => {
    const docked: Box = { top: 708, bottom: 800, left: 0, right: 1400 };
    expect(hudInsets(canvas, [], [docked])).toEqual({ top: 0, right: 0, bottom: 0, left: 0 });
  });

  it("always leaves the engine at least a third of the canvas", () => {
    const tall: Box = { top: 100, bottom: 500, left: 0, right: 1400 };
    const low: Box = { top: 300, bottom: 700, left: 0, right: 1400 };
    const i = hudInsets(canvas, [tall], [low]);
    expect(i.top + i.bottom).toBeLessThanOrEqual(400);
  });
});

/** The synthetic states scene only shows combinations the simulator can produce. */
import { describe, expect, it } from "vitest";

import type { MachineState } from "../api/types.ts";
import { DEFAULT_CONFIG } from "./config.ts";
import { computeLayout, footprintRect, rectsOverlap } from "./layout.ts";
import { STATE_SCENE, STATE_SEGMENTS, stateSceneLine, stateSceneLineView } from "./synthetic.ts";

/** States in which the simulator's machine is moving parts (twin_sim/line.py state priority). */
const MOVING: readonly MachineState[] = ["RUNNING", "WARNING"];

describe("states scene", () => {
  const line = stateSceneLine();
  const view = stateSceneLineView();
  const stateOf = new Map(STATE_SCENE.map((s) => [s.id, s.state]));

  it("covers every machine state the twin shows", () => {
    const states = new Set(STATE_SCENE.map((s) => s.state));
    for (const st of ["RUNNING", "WARNING", "FAULT", "STARVED", "BLOCKED", "MAINTENANCE", "STALE", "OFFLINE"] as const) {
      expect(states.has(st), st).toBe(true);
    }
  });

  it("never blames a moving (RUNNING / WARNING) machine for STARVED / BLOCKED", () => {
    for (const s of STATE_SCENE) {
      if (s.state !== "STARVED" && s.state !== "BLOCKED") continue;
      const cause = view.rootCauses[s.id]?.cause ?? null;
      expect(cause, s.id).not.toBeNull();
      expect(MOVING.includes(stateOf.get(cause ?? "") ?? "IDLE"), `${s.id} caused by ${String(cause)}`).toBe(false);
    }
  });

  it("a STARVED machine's feeder is stopped, a BLOCKED machine's receiver is stopped", () => {
    for (const s of STATE_SCENE) {
      if (s.state === "STARVED") {
        const up = line.topology.upstream[s.id] ?? null;
        expect(up, s.id).not.toBeNull();
        expect(MOVING.includes(stateOf.get(up ?? "") ?? "RUNNING"), s.id).toBe(false);
      }
      if (s.state === "BLOCKED") {
        const down = line.topology.downstream[s.id] ?? null;
        expect(down, s.id).not.toBeNull();
        expect(MOVING.includes(stateOf.get(down ?? "") ?? "RUNNING"), s.id).toBe(false);
      }
    }
  });

  it("the WARNING machine's neighbours run (WARNING does not stop the flow)", () => {
    const warn = STATE_SCENE.find((s) => s.state === "WARNING");
    if (warn === undefined) throw new Error("no WARNING machine");
    for (const n of [line.topology.upstream[warn.id], line.topology.downstream[warn.id]]) {
      expect(stateOf.get(n ?? ""), String(n)).toBe("RUNNING");
    }
  });

  it("puts each segment in its own row without overlaps; unlinked segments get no buffer", () => {
    const layout = computeLayout(line, DEFAULT_CONFIG.layout);
    const rects = [...layout.machines.values()].map(footprintRect);
    rects.forEach((a, i) => {
      rects.slice(i + 1).forEach((b) => {
        expect(rectsOverlap(a, b)).toBe(false);
      });
    });
    const unlinked = STATE_SEGMENTS.filter((s) => !s.linked).flatMap((s) => s.machines.map((m) => m.id));
    expect(line.buffers.some((b) => unlinked.includes(b.from) || unlinked.includes(b.to))).toBe(false);
    expect(view.rootCauses["S-STARVE"]).toEqual({ cause: "S-FAULT", unknown: false });
    expect(view.rootCauses["S-BLOCK2"]).toEqual({ cause: "S-MAINT", unknown: false });
  });
});

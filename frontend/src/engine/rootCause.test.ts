/** The TS root-cause port equals the backend propagation on every generated case. */
import { describe, expect, it } from "vitest";

import type { LineViewModel, MachineState } from "../api/types.ts";
import fixture from "./__fixtures__/root_causes.json";
import { lineView, rootCause, type TopologyLike } from "./rootCause.ts";

interface Case {
  topology: TopologyLike;
  states: Record<string, MachineState | null>;
  expected: LineViewModel;
}

const cases = (fixture as unknown as { cases: Case[] }).cases;

describe("rootCause port", () => {
  it("has a fixture with many cases (regenerate with uv run poe api-docs)", () => {
    expect(cases.length).toBeGreaterThan(100);
  });

  it("matches the backend propagation on every fixture case", () => {
    for (const [i, c] of cases.entries()) {
      expect(lineView(c.states, c.topology), `case ${String(i)}: ${JSON.stringify(c.states)}`).toEqual(c.expected);
    }
  });

  it("names a WARNING neighbour that stopped the flow, and walks chains", () => {
    const topo: TopologyLike = {
      order: ["A", "B", "C"],
      upstream: { A: null, B: "A", C: "B" },
      downstream: { A: "B", B: "C", C: null },
    };
    expect(rootCause("C", { A: "IDLE", B: "STARVED", C: "STARVED" }, topo)).toEqual({ cause: "A", unknown: false });
    expect(rootCause("A", { A: "STARVED", B: "RUNNING", C: "RUNNING" }, topo)).toEqual({ cause: "SOURCE", unknown: false });
    expect(rootCause("A", { A: "BLOCKED", B: "OFFLINE", C: "RUNNING" }, topo)).toEqual({ cause: "B", unknown: true });
  });
});

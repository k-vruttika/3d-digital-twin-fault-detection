import { describe, expect, it } from "vitest";

import { cloneLine, LINE, LINE_SINGLE } from "../test/fakes.ts";
import { buffersAround, flowTopology, topologicalOrder } from "./topology.ts";

describe("flowTopology", () => {
  it("follows the buffers of the full line and keeps the backend's neighbours", () => {
    const topo = flowTopology(LINE);
    expect(topo.order).toEqual(["CNV-01", "CNC-01", "INSP-01", "CNV-02"]);
    expect(topo.upstream.get("CNC-01")).toBe("CNV-01");
    expect(topo.downstream.get("CNV-02")).toBeNull();
    expect(topo.upstream.get("CNV-01")).toBeNull();
    expect(topo.warnings).toEqual([]);
    expect(buffersAround(topo, "CNC-01")).toMatchObject({ input: { id: "B1" }, output: { id: "B2" } });
    expect(buffersAround(topo, "CNV-01").input).toBeNull();
  });

  it("uses config order for a line without buffers", () => {
    const topo = flowTopology(LINE_SINGLE);
    expect(topo.order).toEqual(["CNV-01"]);
    expect(topo.buffers).toEqual([]);
  });

  it("orders by the buffers even when the config lists machines out of flow order", () => {
    const line = cloneLine();
    line.machines.reverse();
    expect(flowTopology(line).order).toEqual(["CNV-01", "CNC-01", "INSP-01", "CNV-02"]);
  });

  it("falls back to config order on a cycle", () => {
    const line = cloneLine();
    line.buffers.push({ id: "B9", from: "CNV-02", to: "CNV-01", capacity: 1 });
    const topo = flowTopology(line);
    expect(topo.order).toEqual(line.machines.map((m) => m.id));
    expect(topo.warnings.join(" ")).toMatch(/cycle/);
  });

  it("reports branches and merges, and drops buffers to unknown machines", () => {
    const line = cloneLine();
    line.buffers.push({ id: "BX", from: "CNV-01", to: "INSP-01", capacity: 2 });
    line.buffers.push({ id: "BY", from: "CNV-01", to: "NOPE", capacity: 2 });
    const topo = flowTopology(line);
    expect(topo.warnings.some((w) => w.includes("CNV-01 feeds 2"))).toBe(true);
    expect(topo.warnings.some((w) => w.includes("INSP-01 is fed by 2"))).toBe(true);
    expect(topo.warnings.some((w) => w.includes("BY"))).toBe(true);
    expect(topo.buffers.map((b) => b.id)).not.toContain("BY");
    expect(topo.order[0]).toBe("CNV-01");
  });
});

describe("topologicalOrder", () => {
  it("breaks ties by config order", () => {
    expect(topologicalOrder(["a", "b", "c"], [{ from: "c", to: "b" }])).toEqual(["a", "c", "b"]);
  });

  it("returns null for a cycle", () => {
    expect(
      topologicalOrder(["a", "b"], [
        { from: "a", to: "b" },
        { from: "b", to: "a" },
      ]),
    ).toBeNull();
  });
});

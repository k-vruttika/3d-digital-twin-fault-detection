// @vitest-environment jsdom
/**
 * Phase 11 Part C: the What-if panel (ADR 0031).
 * Pure helpers, the panel (form, ghost states, dashed projection lines, outcome text), the
 * 409 / 422 messages, and "apply as command" (operator only, after a confirmation).
 */
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { MachineProjection, MachineState, ProjectionPoint, WhatIfResponse } from "../api/types.ts";
import { OPERATOR, VIEWER, pathOf, renderDashboard, type Route } from "../test/dashboard.tsx";
import { DASH } from "./charts/projectionConfig.ts";
import { applyBody, buildPayload, commandText, ghostRows, hasData, outcomeLines, projectionSeries } from "./state/whatif.ts";

afterEach(() => {
  cleanup();
});

const AS_OF = "2026-01-01T00:01:00Z";
const IDS = ["CNV-01", "CNC-01", "INSP-01", "CNV-02"];

function pt(tS: number, state: MachineState, over: Partial<ProjectionPoint> = {}): ProjectionPoint {
  return { tS, ts: new Date(Date.parse(AS_OF) + tS * 1000).toISOString(), state, vib: 1 + tS / 100, temp: 60, speed: 8000, load: 50, current: 10, ...over };
}

function machine(
  state: MachineState,
  endState: MachineState,
  over: Partial<{ firstFaultS: number; firstAlarmS: number }> = {},
): MachineProjection {
  return {
    points: [pt(1, state), pt(2, state, { vib: null }), pt(3, endState)],
    ghost: { "10": state, "30": endState, end: endState },
    ...over,
  };
}

function response(over: Partial<WhatIfResponse> = {}): WhatIfResponse {
  const hist = (s: MachineState): ProjectionPoint[] => [pt(-1, s), pt(0, s)];
  return {
    asOf: AS_OF,
    machineId: "CNC-01",
    type: "STOP",
    params: {},
    horizonS: 60,
    ghostHorizonsS: [10, 30],
    history: Object.fromEntries(IDS.map((id) => [id, hist("RUNNING")])),
    baseline: { machines: Object.fromEntries(IDS.map((id) => [id, machine("RUNNING", "RUNNING")])) },
    scenario: {
      machines: {
        ...Object.fromEntries(IDS.map((id) => [id, machine("RUNNING", "RUNNING")])),
        "CNC-01": machine("IDLE", "IDLE"),
        "INSP-01": machine("RUNNING", "STARVED", { firstFaultS: 45 }),
      },
    },
    elapsedMs: 900,
    modelNote: "Live inference is off: the projection shows the simulator only, without predictions.",
    ...over,
  };
}

const SOURCES = { active: "mock", capabilities: ["commands", "history", "scenario", "whatif"], configured: ["mock"] };
const sourcesRoute: Route = (c) => (pathOf(c) === "/sources" ? { body: SOURCES } : undefined);
const whatifRoute =
  (res: { status?: number; body: unknown }): Route =>
  (c) =>
    pathOf(c) === "/whatif" && c.method === "POST" ? res : undefined;

// ---------------------------------------------------------------------------- pure helpers
describe("what-if helpers", () => {
  it("builds the request and asks for a number where the change needs one", () => {
    expect(buildPayload("CNC-01", "STOP", "", 120)).toEqual({ ok: true, payload: { machineId: "CNC-01", type: "STOP", params: {}, horizonS: 120 } });
    expect(buildPayload("CNC-01", "SET_SPEED", "3000", 60)).toEqual({
      ok: true,
      payload: { machineId: "CNC-01", type: "SET_SPEED", params: { speed: 3000 }, horizonS: 60 },
    });
    expect(buildPayload("CNC-01", "SET_LOAD", "", 60).ok).toBe(false);
    expect(buildPayload("CNC-01", "SET_SPEED", "abc", 60).ok).toBe(false);
    expect(buildPayload(null, "STOP", "", 60).ok).toBe(false);
  });

  it("series: history solid, projections start at the last history point", () => {
    const s = projectionSeries(response(), "CNC-01", "vib");
    expect(s.history.map((p) => p.x)).toEqual([-1, 0]);
    expect(s.baseline[0]).toEqual(s.history[1]);
    expect(s.scenario.map((p) => p.x)).toEqual([0, 1, 2, 3]);
    expect(s.scenario[2]?.y).toBeNull(); // a missing value is a gap, never a zero
    expect(hasData(s)).toBe(true);
    expect(hasData(projectionSeries(response(), "CNC-01", "pFault"))).toBe(false); // no model
  });

  it("ghost rows: real state now, projected states of both, and what differs", () => {
    const rows = ghostRows(response(), IDS);
    const cnc = rows.find((r) => r.machineId === "CNC-01");
    expect(cnc?.now).toBe("RUNNING");
    expect(cnc?.cells.map((c) => [c.horizon, c.baseline, c.scenario, c.differs])).toEqual([
      ["10", "RUNNING", "IDLE", true],
      ["30", "RUNNING", "IDLE", true],
      ["end", "RUNNING", "IDLE", true],
    ]);
    expect(rows.find((r) => r.machineId === "CNV-01")?.changed).toBe(false);
    expect(rows.find((r) => r.machineId === "INSP-01")?.cells[1]?.scenario).toBe("STARVED");
  });

  it("outcome text names the model only when there is one", () => {
    const plain = outcomeLines(response(), "INSP-01");
    expect(plain.join(" ")).toContain("Simulator FAULT: without the change not within the run, with it at +45 s");
    expect(plain.join(" ")).not.toContain("Model");
    const m = machine("RUNNING", "RUNNING", { firstAlarmS: 20 });
    const withModel = response({ model: "linetwin-gru v1", primaryHorizonS: 30, scenario: { machines: { ...response().scenario.machines, "CNC-01": m } } });
    expect(outcomeLines(withModel, "CNC-01").join(" ")).toContain("alarm at +20 s");
  });

  it("apply body keeps numeric params only; text names the change", () => {
    const r = response({ type: "SET_SPEED", params: { speed: 3000 } });
    expect(applyBody(r)).toEqual({ machineId: "CNC-01", type: "SET_SPEED", params: { speed: 3000 } });
    expect(commandText(r)).toBe("SET_SPEED (speed 3000) on CNC-01");
  });
});

// ---------------------------------------------------------------------------- the panel
async function open(h: ReturnType<typeof renderDashboard>): Promise<void> {
  fireEvent.click(screen.getByTestId("tab-whatif"));
  await waitFor(() => {
    expect(h.model.sources.get()?.capabilities).toContain("whatif");
  });
}

describe("what-if panel", () => {
  it("runs a projection: ghost states, dashed lines, outcome, and sends nothing to a machine", async () => {
    const h = renderDashboard({ auth: OPERATOR, routes: [sourcesRoute, whatifRoute({ body: response() })] });
    await open(h);
    fireEvent.change(screen.getByTestId("whatif-machine"), { target: { value: "CNC-01" } });
    fireEvent.change(screen.getByTestId("whatif-horizon"), { target: { value: "60" } });
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-result")).toBeTruthy();
    });
    const post = h.calls.find((c) => pathOf(c) === "/whatif");
    expect(post?.body).toEqual({ machineId: "CNC-01", type: "STOP", params: {}, horizonS: 60 });
    expect(h.calls.some((c) => pathOf(c) === "/commands" && c.method === "POST")).toBe(false);
    // ghost states: "no change → with change"
    const cell = screen.getByTestId("whatif-ghost-CNC-01-30");
    expect(cell.textContent).toContain("RUNNING");
    expect(cell.textContent).toContain("IDLE");
    expect(within(cell).getAllByText(/RUNNING|IDLE/).every((e) => e.classList.contains("ghost"))).toBe(true);
    expect(screen.getByTestId("whatif-ghost-CNC-01-end")).toBeTruthy();
    expect(screen.getByTestId("whatif-outcome").textContent).toContain("Simulator FAULT");
    expect(screen.getByTestId("whatif-model-note").textContent).toContain("without predictions");
    expect(screen.queryByTestId("whatif-field-pFault")).toBeNull(); // no model: no probability chart
    // the projections are dashed, the measured history is solid
    await waitFor(() => {
      expect(h.charts.length).toBeGreaterThan(0);
    });
    const ds = h.charts[h.charts.length - 1]?.config.data.datasets ?? [];
    expect(ds.map((d) => (d as { borderDash?: number[] }).borderDash)).toEqual([undefined, DASH, DASH]);
    // another signal and another machine
    fireEvent.click(screen.getByTestId("whatif-field-temp"));
    fireEvent.click(screen.getByTestId("whatif-row-INSP-01"));
    expect(screen.getByTestId("whatif-row-INSP-01").className).toContain("selected");
  });

  it("shows a model's probabilities and alarm lines when the response has a model", async () => {
    const res = response({ model: "linetwin-gru v1", primaryHorizonS: 30, modelNote: null });
    res.scenario.machines["CNC-01"] = { ...machine("RUNNING", "FAULT", { firstAlarmS: 12 }), points: [pt(1, "RUNNING", { pFault: 0.4, pWarning: 0.2, gate: "predict" })] };
    const h = renderDashboard({ auth: OPERATOR, routes: [sourcesRoute, whatifRoute({ body: res })] });
    await open(h);
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-model").textContent).toContain("linetwin-gru v1");
    });
    expect(screen.getByTestId("whatif-outcome").textContent).toContain("alarm at +12 s");
    fireEvent.click(screen.getByTestId("whatif-field-pFault"));
    expect(screen.getByTestId("whatif-chart")).toBeTruthy();
  });

  it("needs a number for set speed and sends it", async () => {
    const h = renderDashboard({ auth: OPERATOR, routes: [sourcesRoute, whatifRoute({ body: response({ type: "SET_SPEED", params: { speed: 3000 } }) })] });
    await open(h);
    fireEvent.change(screen.getByTestId("whatif-type"), { target: { value: "SET_SPEED" } });
    fireEvent.click(screen.getByTestId("whatif-run"));
    expect(screen.getByTestId("whatif-error").textContent).toContain("needs a number");
    expect(h.calls.some((c) => pathOf(c) === "/whatif")).toBe(false);
    fireEvent.change(screen.getByTestId("whatif-param"), { target: { value: "3000" } });
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(h.calls.find((c) => pathOf(c) === "/whatif")?.body).toMatchObject({ type: "SET_SPEED", params: { speed: 3000 } });
    });
  });

  it("explains a refusal (422) and a source without the capability (409)", async () => {
    const refused = renderDashboard({
      auth: OPERATOR,
      routes: [sourcesRoute, whatifRoute({ status: 422, body: { detail: "speed 99999 RPM is outside [1, 12000] RPM for a CNC machine" } })],
    });
    await open(refused);
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-error").textContent).toBe("Refused: speed 99999 RPM is outside [1, 12000] RPM for a CNC machine");
    });
    expect(screen.queryByTestId("whatif-result")).toBeNull();
    cleanup();
    const unsupported = renderDashboard({
      auth: OPERATOR,
      routes: [sourcesRoute, whatifRoute({ status: 409, body: { detail: "the active source (file adapter) does not support what-if projections" } })],
    });
    await open(unsupported);
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-error").textContent).toContain("Not available: the active source (file adapter)");
    });
  });

  it("is disabled, with the reason, when the active source cannot fork a simulation", async () => {
    const file: Route = (c) => (pathOf(c) === "/sources" ? { body: { active: "file", capabilities: [], configured: ["file"] } } : undefined);
    const h = renderDashboard({ auth: OPERATOR, routes: [file] });
    fireEvent.click(screen.getByTestId("tab-whatif"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-blocked").textContent).toContain("the active source (file)");
    });
    expect(screen.getByTestId<HTMLButtonElement>("whatif-run").disabled).toBe(true);
    expect(h.calls.some((c) => pathOf(c) === "/whatif")).toBe(false);
  });

  it("applies the change as a command only after a confirmation (operator)", async () => {
    const ack = { commandId: "c1", status: "EXECUTED", ts: AS_OF };
    const commands: Route = (c) => (pathOf(c) === "/commands" && c.method === "POST" ? { body: ack } : undefined);
    const h = renderDashboard({ auth: OPERATOR, routes: [sourcesRoute, whatifRoute({ body: response() }), commands] });
    await open(h);
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-apply")).toBeTruthy();
    });
    fireEvent.click(screen.getByTestId("whatif-apply"));
    expect(screen.getByTestId("whatif-confirm-text").textContent).toContain("STOP on CNC-01");
    expect(h.calls.some((c) => pathOf(c) === "/commands" && c.method === "POST")).toBe(false); // not yet
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(h.calls.some((c) => pathOf(c) === "/commands" && c.method === "POST")).toBe(false);
    fireEvent.click(screen.getByTestId("whatif-apply"));
    fireEvent.click(screen.getByRole("button", { name: "Send command" }));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-applied").textContent).toContain("Command EXECUTED");
    });
    const sent = h.calls.find((c) => pathOf(c) === "/commands" && c.method === "POST");
    expect(sent?.body).toMatchObject({ machineId: "CNC-01", type: "STOP", params: {} });
    expect((sent?.body as { idempotencyKey: string }).idempotencyKey.length).toBeGreaterThan(8);
  });

  it("a viewer can project but not apply", async () => {
    const h = renderDashboard({ auth: VIEWER, routes: [sourcesRoute, whatifRoute({ body: response() })] });
    await open(h);
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("whatif-result")).toBeTruthy();
    });
    expect(screen.queryByTestId("whatif-apply")).toBeNull();
    expect(screen.getByTestId("whatif-apply-need-operator")).toBeTruthy();
  });

  it("does not run in replay mode", async () => {
    const h = renderDashboard({ auth: OPERATOR, routes: [sourcesRoute] });
    await open(h);
    act(() => {
      h.model.setMode("replay");
    });
    expect(screen.getByTestId("whatif-blocked").textContent).toContain("replay");
    expect(screen.getByTestId<HTMLButtonElement>("whatif-run").disabled).toBe(true);
  });
});

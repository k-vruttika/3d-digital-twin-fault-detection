// @vitest-environment jsdom
/** Commands, scenarios, sources, replay and the Phase 11 placeholders. */
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Command, CommandAck, CommandLimitsResponse, SourcesResponse } from "../api/types.ts";
import { OPERATOR, pathOf, renderDashboard, snapshot, VIEWER, type Route } from "../test/dashboard.tsx";
import { reduceFlows, ruleHint, type CommandFlow } from "./state/commandFlow.ts";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const LIMITS: CommandLimitsResponse = {
  limits: {
    CONVEYOR: { speedRpm: { min: 1, max: 1750 }, loadPct: { min: 0, max: 120 } },
    CNC: { speedRpm: { min: 1, max: 12000 }, loadPct: { min: 0, max: 120 } },
    INSPECTION: { speedRpm: { min: 1, max: 900 }, loadPct: { min: 0, max: 120 } },
  },
  rules: [{ types: ["START"], forbiddenStates: ["FAULT"], requiredStates: null, reason: "the machine is faulted; RESET it first" }],
  alwaysAllowed: ["STOP"],
};

const limitsRoute: Route = (c) => (pathOf(c) === "/commands/limits" ? { body: LIMITS } : undefined);
const emptyAudit: Route = (c) => (pathOf(c) === "/commands" && c.method === "GET" ? { body: { commands: [] } } : undefined);

function ack(commandId: string, status: CommandAck["status"], reason: string | null = null): CommandAck {
  return { commandId, status, reason, ts: "2026-01-01T00:00:06Z" };
}

function command(commandId: string, idempotencyKey: string): Command {
  return { commandId, idempotencyKey, machineId: "CNC-01", type: "STOP", params: {}, issuedBy: "alice", ts: "2026-01-01T00:00:06Z" };
}

describe("command flow reducer", () => {
  const sent = reduceFlows([], { kind: "sent", idempotencyKey: "k1", machineId: "CNC-01", type: "STOP", params: {} });

  it("pending -> ACCEPTED (stream) -> EXECUTED (response)", () => {
    let f: CommandFlow[] = sent;
    f = reduceFlows(f, { kind: "command", command: command("c1", "k1") });
    f = reduceFlows(f, { kind: "ack", ack: ack("c1", "ACCEPTED") });
    expect(f[0]?.status).toBe("ACCEPTED");
    f = reduceFlows(f, { kind: "response", idempotencyKey: "k1", ack: ack("c1", "EXECUTED") });
    expect(f[0]?.trail).toEqual(["pending", "ACCEPTED", "EXECUTED"]);
    // a late ACCEPTED never reopens a final status
    f = reduceFlows(f, { kind: "ack", ack: ack("c1", "ACCEPTED") });
    expect(f[0]?.status).toBe("EXECUTED");
  });

  it("REJECTED keeps its reason; TIMEOUT and HTTP errors are final", () => {
    expect(reduceFlows(sent, { kind: "response", idempotencyKey: "k1", ack: ack("c1", "REJECTED", "rule") })[0]).toMatchObject({
      status: "REJECTED",
      reason: "rule",
    });
    expect(reduceFlows(sent, { kind: "response", idempotencyKey: "k1", ack: ack("c1", "TIMEOUT") })[0]?.status).toBe("TIMEOUT");
    expect(reduceFlows(sent, { kind: "error", idempotencyKey: "k1", detail: "HTTP 409: reused" })[0]).toMatchObject({
      status: "ERROR",
      reason: "HTTP 409: reused",
    });
  });

  it("ruleHint names the rule that would reject a command; STOP never", () => {
    expect(ruleHint(LIMITS, "START", "FAULT")).toBe("the machine is faulted; RESET it first");
    expect(ruleHint(LIMITS, "START", "IDLE")).toBeNull();
    expect(ruleHint(LIMITS, "STOP", "FAULT")).toBeNull();
  });
});

describe("command panel", () => {
  it("viewers see no command controls", () => {
    renderDashboard({ auth: VIEWER });
    act(() => {
      screen.getByTestId("machine-item-CNC-01").click();
    });
    expect(screen.queryByTestId("command-panel")).toBeNull();
    expect(screen.getByTestId("commands-need-operator")).toBeTruthy();
  });

  it("STOP asks for confirmation, sends a fresh idempotency key and shows pending -> ACCEPTED -> EXECUTED", async () => {
    let resolvePost: ((a: CommandAck) => void) | null = null;
    const h = renderDashboard({ routes: [limitsRoute, emptyAudit] });
    // hold the POST open so the intermediate states are visible
    const rest = h.services.session.rest.get();
    const posted: unknown[] = [];
    vi.spyOn(rest, "postCommand").mockImplementation((body) => {
      posted.push(body);
      return new Promise<CommandAck>((res) => {
        resolvePost = res;
      });
    });
    act(() => {
      h.model.select("CNC-01");
    });
    fireEvent.click(screen.getByTestId("cmd-stop"));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Stop CNC-01?")).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("button", { name: "STOP" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByTestId("command-status").textContent).toBe("PENDING");
    const body = posted[0] as { idempotencyKey: string; type: string; machineId: string };
    expect(body).toMatchObject({ type: "STOP", machineId: "CNC-01" });
    expect(body.idempotencyKey.length).toBeGreaterThan(8);

    act(() => {
      h.model.applyCommand(command("c9", body.idempotencyKey));
      h.model.applyAck(ack("c9", "ACCEPTED"));
    });
    expect(screen.getByTestId("command-status").textContent).toBe("ACCEPTED");
    await act(async () => {
      resolvePost?.(ack("c9", "EXECUTED"));
      await Promise.resolve();
    });
    expect(screen.getByTestId("command-status").textContent).toBe("EXECUTED");

    // a second click gets a different idempotency key
    fireEvent.click(screen.getByTestId("cmd-stop"));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "STOP" }));
    const second = posted[1] as { idempotencyKey: string };
    expect(second.idempotencyKey).not.toBe(body.idempotencyKey);
  });

  it("REJECTED shows the reason; Escape cancels the STOP dialog", async () => {
    const h = renderDashboard({
      routes: [
        limitsRoute,
        emptyAudit,
        (c) => (c.method === "POST" && pathOf(c) === "/commands" ? { body: ack("c1", "REJECTED", "the machine is faulted; RESET it first") } : undefined),
      ],
    });
    act(() => {
      h.model.select("CNC-01");
    });
    fireEvent.click(screen.getByTestId("cmd-stop"));
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(h.calls.some((c) => c.method === "POST")).toBe(false);
    fireEvent.click(screen.getByTestId("cmd-START"));
    await waitFor(() => {
      expect(screen.getByTestId("command-status").textContent).toBe("REJECTED");
    });
    expect(screen.getByTestId("command-reason").textContent).toContain("RESET it first");
  });

  it("enforces the backend's limits for the machine's kind", async () => {
    const h = renderDashboard({ routes: [limitsRoute, emptyAudit] });
    act(() => {
      h.model.select("INSP-01");
    });
    await waitFor(() => {
      expect(screen.getByText("RPM (1–900)")).toBeTruthy();
    });
    const input = screen.getByTestId("input-speed");
    fireEvent.change(input, { target: { value: "1200" } });
    expect(screen.getByTestId<HTMLButtonElement>("cmd-SET_SPEED").disabled).toBe(true);
    expect(screen.getByRole("alert").textContent).toContain("outside the allowed range");
    fireEvent.change(input, { target: { value: "600" } });
    expect(screen.getByTestId<HTMLButtonElement>("cmd-SET_SPEED").disabled).toBe(false);
  });

  it("disables commands while the stream is not connected", () => {
    const h = renderDashboard({ routes: [limitsRoute, emptyAudit], connected: false });
    act(() => {
      h.model.select("CNC-01");
    });
    expect(screen.getByTestId<HTMLButtonElement>("cmd-stop").disabled).toBe(true);
    expect(screen.getByTestId<HTMLButtonElement>("cmd-START").disabled).toBe(true);
  });
});

function sources(capabilities: NonNullable<SourcesResponse["capabilities"]>): SourcesResponse {
  return {
    active: "mock",
    capabilities,
    health: { kind: "mock", status: "CONNECTED", connected: true, messages: 10, reconnects: 0, lastMessageAt: null, lastError: null },
    configured: ["file", "mock"],
    switches: 0,
    ingest: {
      messagesIn: 100,
      published: 98,
      late: 2,
      latePct: 2.04,
      quarantined: 1,
      quarantinedByReason: { schema: 1 },
      partial: 3,
      partialPct: 3.06,
      duplicates: 0,
      latency: { count: 98, meanS: 0.004, p95S: 0.01 },
    },
  };
}

const scenarioOptions: Route = (c) =>
  pathOf(c) === "/scenario/options"
    ? {
        body: {
          faults: [
            { faultType: "bearing_wear", kinds: ["CONVEYOR", "CNC", "INSPECTION"], gradual: true, heldOut: false },
            { faultType: "spindle_imbalance", kinds: ["CNC"], gradual: true, heldOut: false },
            { faultType: "belt_jam", kinds: ["CONVEYOR", "INSPECTION"], gradual: false, heldOut: false },
          ],
          progressionS: { min: 720, max: 1500 },
        },
      }
    : undefined;
const emptyScenarios: Route = (c) => (pathOf(c) === "/scenario" && c.method === "GET" ? { body: { requests: [] } } : undefined);

describe("sources and scenario panels", () => {
  it("shows ingestion metrics; the scenario tab only with the SCENARIO capability", async () => {
    const h = renderDashboard({ routes: [(c) => (pathOf(c) === "/sources" ? { body: sources(["commands"]) } : undefined)] });
    await waitFor(() => {
      expect(h.model.sources.get()).not.toBeNull();
    });
    expect(screen.queryByTestId("tab-scenario")).toBeNull();
    fireEvent.click(screen.getByTestId("tab-sources"));
    const stats = screen.getByTestId("ingest-stats");
    expect(stats.textContent).toContain("2.0 %");
    expect(stats.textContent).toContain("schema 1");
    act(() => {
      h.model.sources.set(sources(["commands", "scenario"]));
    });
    expect(screen.getByTestId("tab-scenario")).toBeTruthy();
  });

  it("scenario panel is labelled simulation only, filters faults by kind and posts the request", async () => {
    let body: unknown = null;
    const h = renderDashboard({
      routes: [
        scenarioOptions,
        emptyScenarios,
        (c) => (pathOf(c) === "/sources" ? { body: sources(["commands", "scenario"]) } : undefined),
        (c) => {
          if (c.method !== "POST" || pathOf(c) !== "/scenario") return undefined;
          body = c.body;
          return {
            body: {
              request: { requestId: "r1", action: "INJECT_FAULT", machineId: "CNC-01", faultType: "spindle_imbalance", requestedBy: "alice", source: "api", ts: "2026-01-01T00:00:06Z" },
              acks: [{ requestId: "r1", status: "EXECUTED", reason: null, ts: "2026-01-01T00:00:06Z" }],
            },
          };
        },
      ],
    });
    act(() => {
      h.model.select("CNC-01");
    });
    await waitFor(() => {
      expect(screen.getByTestId("tab-scenario")).toBeTruthy();
    });
    fireEvent.click(screen.getByTestId("tab-scenario"));
    expect(screen.getByRole("note").textContent).toContain("SIMULATION ONLY");
    await waitFor(() => {
      expect(within(screen.getByTestId("scenario-fault")).getAllByRole("option").map((o) => o.textContent)).toEqual([
        "bearing_wear",
        "spindle_imbalance",
      ]); // belt_jam does not apply to a CNC
    });
    fireEvent.change(screen.getByTestId("scenario-fault"), { target: { value: "spindle_imbalance" } });
    fireEvent.change(screen.getByTestId("scenario-progression"), { target: { value: "800" } });
    fireEvent.click(screen.getByTestId("scenario-inject"));
    await waitFor(() => {
      expect(screen.getByTestId("scenario-result").textContent).toContain("EXECUTED");
    });
    expect(body).toEqual({ action: "INJECT_FAULT", machineId: "CNC-01", faultType: "spindle_imbalance", progressionS: 800 });
  });

  it("viewers cannot inject", async () => {
    renderDashboard({
      auth: VIEWER,
      routes: [scenarioOptions, emptyScenarios, (c) => (pathOf(c) === "/sources" ? { body: sources(["scenario"]) } : undefined)],
    });
    await waitFor(() => {
      expect(screen.getByTestId("tab-scenario")).toBeTruthy();
    });
    fireEvent.click(screen.getByTestId("tab-scenario"));
    expect(screen.queryByTestId("scenario-inject")).toBeNull();
    expect(screen.getByText("Injecting needs an operator API key.")).toBeTruthy();
  });
});

const T0 = Date.parse("2026-01-01T00:00:00Z");
const replayRoutes: Route[] = [
  (c) => (pathOf(c) === "/replay/range" ? { body: { start: "2026-01-01T00:00:00Z", end: "2026-01-01T00:10:00Z" } } : undefined),
  (c) => {
    if (pathOf(c) !== "/replay") return undefined;
    const t = new URL(c.url).searchParams.get("t") ?? "";
    const faulted = Date.parse(t) >= T0 + 300_000;
    const s = snapshot({ "CNV-01": "RUNNING", "CNC-01": faulted ? "FAULT" : "RUNNING", "INSP-01": "RUNNING", "CNV-02": "RUNNING" });
    return { body: { asOf: t, machines: s.machines, predictions: {}, line: s.line, alerts: [] } };
  },
];

describe("replay", () => {
  it("shows the REPLAY banner, applies the scrubbed state to the engine and returns to live", async () => {
    const h = renderDashboard({ routes: replayRoutes });
    await waitFor(() => {
      expect(h.engine.calls).toContain("loadLayout");
    });
    expect(screen.queryByTestId("replay-banner")).toBeNull();
    act(() => {
      h.model.applyDataQuality({
        machineId: "CNC-01",
        status: "DEGRADED",
        samples: 10,
        latePct: 0,
        quarantinePct: 0,
        partialPct: 0,
        linkDown: true,
        reasons: ["data link down (STALE/OFFLINE)"],
      });
    });
    fireEvent.click(screen.getByTestId("enter-replay"));
    await waitFor(() => {
      expect(screen.getByTestId("replay-banner")).toBeTruthy();
    });
    expect(h.engine.mode).toBe("replay");
    await waitFor(() => {
      expect(screen.getByTestId("replay-scrubber")).toBeTruthy();
    });
    fireEvent.change(screen.getByTestId("replay-scrubber"), { target: { value: String(T0 + 400_000) } });
    await waitFor(() => {
      expect(h.model.machines.get()["CNC-01"]?.state).toBe("FAULT");
    });
    expect(screen.getByTestId("replay-time").textContent).toBe("2026-01-01 00:06:40 UTC");
    expect(within(screen.getByTestId("machine-item-CNC-01")).getByText("FAULT")).toBeTruthy();
    // data quality is not replayable: the live report is not shown as if it were history
    expect(h.model.machines.get()["CNC-01"]?.dataQuality).toBeNull();
    // commands are disabled in replay
    act(() => {
      h.model.select("CNC-01");
    });
    expect(screen.getByTestId("details-panel").textContent).toContain("Not replayed");
    expect(screen.getByTestId<HTMLButtonElement>("cmd-stop").disabled).toBe(true);
    // live data is ignored in replay
    expect(h.model.mode).toBe("replay");

    fireEvent.click(screen.getByTestId("exit-replay"));
    expect(screen.queryByTestId("replay-banner")).toBeNull();
    expect(h.engine.mode).toBe("live");
    expect(h.resubscribed.count).toBe(1); // a fresh snapshot replaces the replayed state
  });

  it("play advances event time at the chosen speed with one request in flight", async () => {
    vi.useFakeTimers();
    const h = renderDashboard({ routes: replayRoutes });
    const rest = h.services.session.rest.get();
    const pending: (() => void)[] = [];
    const getReplay = vi.spyOn(rest, "getReplay").mockImplementation((t) => {
      return new Promise((resolve) => {
        const s = snapshot({ "CNC-01": "RUNNING" });
        pending.push(() => {
          resolve({ asOf: t, machines: s.machines, predictions: {}, line: s.line, alerts: [] });
        });
      });
    });
    await act(async () => {
      await h.replay.enter();
    });
    expect(getReplay).toHaveBeenCalledTimes(1); // the initial seek
    act(() => {
      h.replay.seek(T0);
      h.replay.setSpeed(20);
      h.replay.play();
    });
    // three ticks while the first request is still open: no new requests
    act(() => {
      vi.advanceTimersByTime(h.model.cfg.replayTickMs * 3);
    });
    expect(getReplay).toHaveBeenCalledTimes(1);
    expect(h.replay.state.get().t).toBe(T0 + 3 * 20 * h.model.cfg.replayTickMs);
    // finishing it sends exactly one request, for the newest target
    await act(async () => {
      pending.shift()?.();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(getReplay).toHaveBeenCalledTimes(2);
    expect(getReplay.mock.calls[1]?.[0]).toBe(new Date(T0 + 3 * 20 * h.model.cfg.replayTickMs).toISOString());
    act(() => {
      h.replay.pause();
    });
    h.replay.dispose();
  });
});

describe("placeholders", () => {
  const notImplemented: Route = (c) =>
    pathOf(c) === "/predict" || pathOf(c) === "/whatif"
      ? { status: 501, body: { detail: "available from Phase 11 (inference service); not implemented yet" } }
      : undefined;

  it("prediction and what-if handle the 501 gracefully", async () => {
    renderDashboard({ auth: OPERATOR, routes: [notImplemented] });
    fireEvent.click(screen.getByTestId("tab-prediction"));
    await waitFor(() => {
      expect(screen.getByTestId("placeholder-501").textContent).toContain("available in Phase 11");
    });
    fireEvent.click(screen.getByTestId("tab-whatif"));
    fireEvent.click(screen.getByTestId("whatif-run"));
    await waitFor(() => {
      expect(screen.getByTestId("placeholder-501").textContent).toContain("What-if: available in Phase 11");
    });
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

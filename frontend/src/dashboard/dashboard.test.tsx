// @vitest-environment jsdom
/** Dashboard components: role gating, connection banner, alert states, details, charts. */
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ConnectionBanner } from "./components/ConnectionBanner.tsx";
import { DashboardContext } from "./context.ts";
import { LiveCharts } from "./charts/LiveCharts.tsx";
import {
  alert,
  ANONYMOUS,
  OPERATOR,
  pathOf,
  reading,
  renderDashboard,
  VIEWER,
} from "../test/dashboard.tsx";

afterEach(() => {
  cleanup();
});

describe("layout and selection", () => {
  it("lists every machine with its state badge and shows details of the selected one", async () => {
    const h = renderDashboard({ states: { "CNV-01": "BLOCKED", "CNC-01": "FAULT", "INSP-01": "STARVED", "CNV-02": "STARVED" } });
    act(() => {
      h.model.applyLine({
        status: "STOPPED",
        rootCauses: { "CNV-01": { cause: "CNC-01", unknown: false }, "INSP-01": { cause: "CNC-01", unknown: false } },
        impacted: {},
      });
    });
    const list = screen.getByTestId("machine-list");
    expect(within(list).getAllByRole("option")).toHaveLength(4);
    expect(within(screen.getByTestId("machine-item-CNC-01")).getByText("FAULT")).toBeTruthy();
    expect(within(screen.getByTestId("machine-item-CNV-01")).getByText("BLOCKED by CNC-01")).toBeTruthy();

    fireEvent.click(screen.getByTestId("machine-item-CNC-01"));
    const details = await screen.findByTestId("details-panel");
    expect(within(details).getByTestId("details-state-label").textContent).toBe("FAULT");
    // readings with units from the /line catalog
    const table = within(details).getByTestId("details-readings");
    expect(table.textContent).toContain("RPM");
    expect(table.textContent).toContain("°C");
    // the 3D view follows the selection
    await waitFor(() => {
      expect(h.engine.selected).toBe("CNC-01");
    });
  });

  it("keyboard: arrow keys move the selection; a 3D click selects in the list", async () => {
    const h = renderDashboard();
    const list = screen.getByTestId("machine-list");
    fireEvent.keyDown(list, { key: "ArrowDown" });
    expect(h.model.selection.get()).toBe("CNV-01");
    fireEvent.keyDown(list, { key: "ArrowDown" });
    expect(h.model.selection.get()).toBe("CNC-01");
    fireEvent.keyDown(list, { key: "End" });
    expect(h.model.selection.get()).toBe("CNV-02");
    await waitFor(() => {
      expect(h.engine.calls).toContain("loadLayout");
    });
    act(() => {
      h.engine.click("INSP-01");
    });
    expect(screen.getByTestId("machine-item-INSP-01").getAttribute("aria-selected")).toBe("true");
  });

  it("shows a missing reading value as a dash, never as 0", () => {
    const h = renderDashboard();
    act(() => {
      h.model.applyReading(reading("CNC-01", "RUNNING", 9, { vib: null }));
      h.model.select("CNC-01");
    });
    const row = within(screen.getByTestId("details-readings")).getByText("vib").closest("tr");
    expect(row?.textContent).toContain("–");
    expect(row?.textContent).not.toMatch(/\b0\.00\b/);
  });
});

describe("role gating", () => {
  const withAlert = [alert({ alertId: "a1", machineId: "CNC-01" })];

  it("operator: Acknowledge is offered", () => {
    renderDashboard({ auth: OPERATOR, alerts: withAlert });
    fireEvent.click(screen.getByTestId("tab-alerts"));
    expect(screen.getByRole("button", { name: "Acknowledge" })).toBeTruthy();
    expect(screen.getByTestId("role").textContent).toBe("operator (alice)");
  });

  it("viewer and anonymous: no Acknowledge, with an explanation", () => {
    for (const auth of [VIEWER, ANONYMOUS]) {
      renderDashboard({ auth, alerts: withAlert });
      fireEvent.click(screen.getByTestId("tab-alerts"));
      expect(screen.queryByRole("button", { name: "Acknowledge" })).toBeNull();
      expect(screen.getByText("Acknowledging needs an operator key.")).toBeTruthy();
      cleanup();
    }
    renderDashboard({ auth: ANONYMOUS });
    expect(screen.getByTestId("role").textContent).toBe("anonymous viewer");
  });
});

describe("connection banner", () => {
  it("is hidden while connected and says reconnecting, then link lost", () => {
    let now = 1_000_000;
    const h = renderDashboard({ now: () => now, alerts: [alert({ alertId: "a1", machineId: "CNC-01" })] });
    expect(screen.queryByTestId("connection-banner")).toBeNull();
    expect(screen.getByTestId("connection-pill").textContent).toContain("Connected");

    act(() => {
      h.model.setStreamStatus("reconnecting");
    });
    expect(screen.getByTestId("connection-banner").textContent).toContain("Reconnecting");
    fireEvent.click(screen.getByTestId("tab-alerts"));
    // operator buttons are disabled while not connected
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Acknowledge" }).disabled).toBe(true);

    now += h.model.cfg.linkLostAfterMs + 1;
    act(() => {
      h.model.refreshConnection();
    });
    expect(screen.getByTestId("connection-banner").textContent).toContain("Link lost");
    expect(screen.getByTestId("connection-pill").textContent).toContain("Link lost");

    act(() => {
      h.model.setStreamStatus("live");
    });
    expect(screen.queryByTestId("connection-banner")).toBeNull();
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Acknowledge" }).disabled).toBe(false);
    h.model.dispose();
  });

  it("says the key was rejected on unauthorized", () => {
    const h = renderDashboard({
      ui: (s) => (
        <DashboardContext.Provider value={s}>
          <ConnectionBanner />
        </DashboardContext.Provider>
      ),
    });
    act(() => {
      h.model.setStreamStatus("unauthorized");
    });
    expect(screen.getByTestId("connection-banner").textContent).toContain("refused the API key");
  });
});

describe("alerts panel", () => {
  const alerts = [
    alert({ alertId: "u", machineId: "CNC-01", severity: "CRITICAL" }),
    alert({ alertId: "k", machineId: "INSP-01", severity: "WARNING", kind: "machine_warning", acknowledged: true, ackedBy: "alice", rev: 1 }),
    alert({ alertId: "c", machineId: "CNV-01", severity: "WARNING", kind: "line_stoppage", active: false, rev: 1 }),
    alert({ alertId: "r", machineId: "CNV-02", active: false, acknowledged: true, rev: 2 }), // resolved: hidden
  ];

  it("groups alerts by ISA-18.2 state and hides resolved ones", () => {
    renderDashboard({ alerts });
    fireEvent.click(screen.getByTestId("tab-alerts"));
    expect(within(screen.getByTestId("alert-group-activeUnacked")).getByTestId("alert-u")).toBeTruthy();
    expect(within(screen.getByTestId("alert-group-activeAcked")).getByTestId("alert-k")).toBeTruthy();
    expect(within(screen.getByTestId("alert-group-clearedUnacked")).getByTestId("alert-c")).toBeTruthy();
    expect(screen.queryByTestId("alert-r")).toBeNull();
  });

  it("filters by machine and severity", () => {
    renderDashboard({ alerts });
    fireEvent.click(screen.getByTestId("tab-alerts"));
    const panel = screen.getByTestId("alerts-panel");
    fireEvent.change(within(panel).getByLabelText("Severity"), { target: { value: "WARNING" } });
    expect(screen.queryByTestId("alert-u")).toBeNull();
    expect(screen.getByTestId("alert-k")).toBeTruthy();
    fireEvent.change(within(panel).getByLabelText("Machine"), { target: { value: "CNV-01" } });
    expect(screen.queryByTestId("alert-k")).toBeNull();
    expect(screen.getByTestId("alert-c")).toBeTruthy();
  });

  it("acknowledge posts to the API and moves the alert to the acknowledged group", async () => {
    const acked = alert({ alertId: "u", machineId: "CNC-01", acknowledged: true, ackedBy: "alice", rev: 1 });
    const h = renderDashboard({
      alerts,
      routes: [(c) => (c.method === "POST" && pathOf(c) === "/alerts/u/ack" ? { body: acked } : undefined)],
    });
    fireEvent.click(screen.getByTestId("tab-alerts"));
    const row = screen.getByTestId("alert-u");
    fireEvent.click(within(row).getByRole("button", { name: "Acknowledge" }));
    await waitFor(() => {
      expect(within(screen.getByTestId("alert-group-activeAcked")).getByTestId("alert-u")).toBeTruthy();
    });
    const post = h.calls.find((c) => c.method === "POST");
    expect(post && pathOf(post)).toBe("/alerts/u/ack");
    expect(post?.headers["X-API-Key"]).toBe(undefined); // the harness REST client has no key
  });

  it("clicking a machine in an alert selects it", () => {
    const h = renderDashboard({ alerts });
    fireEvent.click(screen.getByTestId("tab-alerts"));
    fireEvent.click(within(screen.getByTestId("alert-k")).getByRole("button", { name: "INSP-01" }));
    expect(h.model.selection.get()).toBe("INSP-01");
  });
});

describe("live charts", () => {
  it("draws a missing sensor value as a gap (null, spanGaps false), never as zero", async () => {
    const h = renderDashboard({
      ui: (s) => (
        <DashboardContext.Provider value={s}>
          <LiveCharts />
        </DashboardContext.Provider>
      ),
    });
    act(() => {
      h.model.select("CNC-01");
    });
    await waitFor(() => {
      expect(h.charts.length).toBeGreaterThanOrEqual(5);
    });
    act(() => {
      h.model.applyReading(reading("CNC-01", "RUNNING", 10));
      h.model.applyReading(reading("CNC-01", "RUNNING", 11, { vib: null }));
      h.model.applyReading(reading("CNC-01", "RUNNING", 12));
    });
    await waitFor(() => {
      const vib = h.charts.find((c) => c.config.data.datasets.some((d) => d.label === "vib"));
      const ds = vib?.data.datasets.find((d) => d.label === "vib");
      expect(ds?.data.length).toBe(3);
    });
    const vib = h.charts.find((c) => c.config.data.datasets.some((d) => d.label === "vib"));
    const ds = vib?.data.datasets.find((d) => d.label === "vib");
    expect(ds?.spanGaps).toBe(false);
    expect(ds?.data.map((p) => p.y)).toEqual([1, null, 1]);
    // the axis has the unit
    const y = vib?.options.scales?.y as { title?: { text?: string } } | undefined;
    expect(y?.title?.text).toBe("vib (mm/s)");
    // history prefill asked for the selected machine, raw
    const hist = h.calls.find((c) => pathOf(c) === "/history");
    expect(hist?.url).toContain("machines=CNC-01");
    expect(hist?.url).toContain("resolution=raw");
  });
});

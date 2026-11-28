// @vitest-environment jsdom
/** The redesigned shell (DESIGN.md §5): annunciator, line strip keyboard, inspector drawer state. */
import { act, cleanup, fireEvent, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { alert, renderDashboard } from "../test/dashboard.tsx";

afterEach(() => {
  cleanup();
});

describe("annunciator", () => {
  it("is a quiet line when nothing needs attention", () => {
    renderDashboard();
    const ann = screen.getByTestId("annunciator");
    expect(ann.textContent).toContain("No active alerts");
    expect(screen.queryByTestId("annunciator-open")).toBeNull();
  });

  it("shows the most urgent alert: active and unacknowledged before acknowledged, CRITICAL first", () => {
    renderDashboard({
      alerts: [
        alert({ alertId: "w", machineId: "CNV-01", severity: "WARNING", message: "CNV-01 is WARNING", acknowledged: true }),
        alert({ alertId: "c", machineId: "INSP-01", severity: "CRITICAL", message: "INSP-01 is FAULT" }),
      ],
    });
    const ann = screen.getByTestId("annunciator");
    expect(ann.getAttribute("data-unacked")).toBe("true");
    expect(ann.textContent).toContain("INSP-01 is FAULT");
    expect(ann.textContent).toContain("+1 more");
    // an active, unacknowledged alarm is the only solid badge (ISA-18.2)
    expect(within(ann).getByText("CRITICAL").className).toContain("solid");
    const steps = within(ann).getByRole("list", { name: "Alert lifecycle" });
    const done = [...steps.querySelectorAll("li")].map((li) => li.getAttribute("data-done"));
    expect(done).toEqual(["true", "false", "false"]);
  });

  it("keeps the unusual-behaviour purple for an autoencoder alarm", () => {
    renderDashboard({
      alerts: [alert({ alertId: "u", machineId: "CNC-01", severity: "WARNING", kind: "unusual_behaviour", message: "CNC-01: unusual behaviour, cause unknown" })],
    });
    const badge = within(screen.getByTestId("annunciator")).getByText("WARNING");
    expect(badge.className).toContain("kind-unusual");
  });

  it("opens the Alerts tab and expands a collapsed console", () => {
    renderDashboard({ alerts: [alert({ alertId: "c", machineId: "CNC-01" })] });
    fireEvent.click(screen.getByTestId("console-toggle"));
    expect(screen.getByTestId("console").getAttribute("data-collapsed")).toBe("true");
    fireEvent.click(screen.getByTestId("annunciator-open"));
    expect(screen.getByTestId("tab-alerts").getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("console").getAttribute("data-collapsed")).toBe("false");
  });
});

describe("line strip", () => {
  it("moves the selection with ArrowRight / ArrowLeft as well as Up / Down", () => {
    const h = renderDashboard();
    const list = screen.getByTestId("machine-list");
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(h.model.selection.get()).toBe("CNV-01");
    fireEvent.keyDown(list, { key: "ArrowRight" });
    expect(h.model.selection.get()).toBe("CNC-01");
    fireEvent.keyDown(list, { key: "ArrowLeft" });
    expect(h.model.selection.get()).toBe("CNV-01");
    fireEvent.keyDown(list, { key: "Escape" });
    expect(h.model.selection.get()).toBeNull();
  });

  it("shows what the data says about a buffer, never a count", () => {
    renderDashboard({ states: { "CNV-01": "BLOCKED", "CNC-01": "FAULT", "INSP-01": "STARVED", "CNV-02": "STARVED" } });
    const first = screen.getByTestId("machine-item-CNV-01");
    expect(first.querySelector(".strip-flow")?.getAttribute("data-buffer")).toBe("FULL");
    const second = screen.getByTestId("machine-item-CNC-01");
    expect(second.querySelector(".strip-flow")?.getAttribute("data-buffer")).toBe("EMPTY");
  });
});

describe("inspector", () => {
  it("keeps the state label text exact while showing the cause on its own line", () => {
    const h = renderDashboard({ states: { "CNV-01": "BLOCKED", "CNC-01": "FAULT", "INSP-01": "RUNNING", "CNV-02": "RUNNING" } });
    act(() => {
      h.model.applyLine({ status: "STOPPED", rootCauses: { "CNV-01": { cause: "CNC-01", unknown: false } }, impacted: {} });
    });
    fireEvent.click(screen.getByTestId("machine-item-CNV-01"));
    const label = screen.getByTestId("details-state-label");
    expect(label.textContent).toBe("BLOCKED by CNC-01");
    expect(label.querySelector(".state-cause")?.textContent).toBe(" by CNC-01");
    expect(h.model.selection.get()).toBe("CNV-01");
  });

  it("flags the layout while a machine is selected (the drawer opens on narrow screens)", () => {
    const h = renderDashboard();
    const layout = screen.getByTestId("layout");
    expect(layout.getAttribute("data-selected")).toBe("false");
    fireEvent.click(screen.getByTestId("machine-item-CNC-01"));
    expect(layout.getAttribute("data-selected")).toBe("true");
    fireEvent.click(screen.getByTestId("inspector-close"));
    expect(h.model.selection.get()).toBeNull();
    expect(layout.getAttribute("data-selected")).toBe("false");
  });
});

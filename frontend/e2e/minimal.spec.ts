/**
 * Minimal mode (file source + SQLite, no Docker): the dashboard loads, lists the machines,
 * shows a selected machine's details and chart, and a replay scrub changes the 3D state.
 */
import { expect, test } from "@playwright/test";

import { engineLabel, moments, openDashboard, scrubTo } from "./helpers.ts";

test("dashboard on minimal mode: machines, details, chart, replay scrub", async ({ page }) => {
  const m = moments();
  await openDashboard(page);

  // every machine of the line is listed with a state badge
  const items = page.getByTestId("machine-list").getByRole("option");
  await expect(items).toHaveCount(4);
  for (const id of ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]) {
    await expect(page.getByTestId(`machine-item-${id}`).locator(".badge")).toBeVisible();
  }

  // selecting a machine shows its details and live charts (prefilled from /history)
  await page.getByTestId(`machine-item-${m.fault.machine}`).click();
  const details = page.getByTestId("details-panel");
  await expect(details).toContainText(m.fault.machine);
  await expect(page.getByTestId("details-readings")).toContainText("°C");
  await expect(page.getByTestId("live-chart-temp").locator("canvas")).toBeVisible();

  // the history tab draws a chart too
  await page.getByTestId("tab-history").click();
  await expect(page.getByTestId("history-chart").locator("canvas")).toBeVisible();

  // replay: the 3D scene follows the scrubbed time
  await page.getByTestId("enter-replay").click();
  await expect(page.getByTestId("replay-banner")).toBeVisible();
  await expect(page.getByTestId("replay-scrubber")).toBeVisible();
  await scrubTo(page, m.normal.ts);
  await expect.poll(() => engineLabel(page, m.normal.machine)).toBe("RUNNING");
  await scrubTo(page, m.fault.ts);
  await expect.poll(() => engineLabel(page, m.fault.machine)).toBe("FAULT");
  await expect(page.getByTestId(`machine-item-${m.fault.machine}`).locator(".badge")).toHaveText("FAULT");
  await scrubTo(page, m.waiting.ts);
  await expect.poll(() => engineLabel(page, m.waiting.machine)).toMatch(new RegExp(`^${m.waiting.state} by `));

  // one click back to live
  await page.getByTestId("exit-replay").click();
  await expect(page.getByTestId("replay-banner")).toHaveCount(0);
});

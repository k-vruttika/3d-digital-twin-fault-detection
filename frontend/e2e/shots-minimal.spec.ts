/**
 * Dashboard screenshots from minimal mode (fixed seeded file, --no-loop). Deterministic
 * states come from replay seeks to moments found in the file (twin_backend.e2e_stack).
 * Run by `python -m twin_backend.e2e_stack screenshots`.
 */
import { expect, test, type Page } from "@playwright/test";

import { engineLabel, expectChartInsidePanel, moments, openDashboard, scrubTo } from "./helpers.ts";

function shotsDir(): string {
  const dir = process.env.E2E_SHOTS_DIR;
  if (dir === undefined) throw new Error("E2E_SHOTS_DIR is not set");
  return dir;
}

async function shot(page: Page, name: string): Promise<void> {
  await page.waitForTimeout(700); // a few frames for the 3D view and charts
  await page.screenshot({ path: `${shotsDir()}/${name}.png` });
}

async function setTheme(page: Page, theme: "light" | "dark"): Promise<void> {
  const current = await page.evaluate(() => document.documentElement.dataset.theme);
  if (current !== theme) await page.getByTestId("theme-toggle").click();
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(theme);
}

async function enterReplayAt(page: Page, iso: string, machine: string, state: RegExp): Promise<void> {
  if ((await page.getByTestId("replay-banner").count()) === 0) await page.getByTestId("enter-replay").click();
  await expect(page.getByTestId("replay-scrubber")).toBeVisible();
  await scrubTo(page, iso);
  await expect.poll(() => engineLabel(page, machine)).toMatch(state);
}

for (const width of [1280, 1920]) {
  test(`overview at ${String(width)} px, dark and light`, async ({ page }) => {
    const m = moments();
    await page.setViewportSize({ width, height: width === 1280 ? 800 : 1080 });
    await openDashboard(page);
    await enterReplayAt(page, m.waiting.ts, m.waiting.machine, new RegExp(`^${m.waiting.state}`));
    // the history chart fits its panel at this width (axis labels visible)
    await page.getByTestId(`machine-item-${m.fault.machine}`).click();
    await page.getByTestId("tab-history").click();
    await expectChartInsidePanel(page, "history-chart");
    await page.getByTestId(`machine-item-${m.fault.machine}`).press("Escape");
    await page.getByTestId("tab-alerts").click();
    await setTheme(page, "dark");
    await shot(page, `overview_dark_${String(width)}`);
    await setTheme(page, "light");
    await shot(page, `overview_light_${String(width)}`);
  });
}

test("machine details, alerts and replay mode", async ({ page }) => {
  const m = moments();
  await openDashboard(page);
  await setTheme(page, "dark");

  // details of the machine that is about to fault, at its first WARNING, with its history
  await enterReplayAt(page, m.warning.ts, m.warning.machine, /^WARNING$/);
  await page.getByTestId(`machine-item-${m.warning.machine}`).click();
  await page.getByTestId("tab-history").click();
  await expectChartInsidePanel(page, "history-chart");
  await shot(page, "machine_details");

  // the alerts panel once the fault has stopped its neighbours
  await scrubTo(page, m.waiting.ts);
  await expect.poll(() => engineLabel(page, m.waiting.machine)).toMatch(new RegExp(`^${m.waiting.state}`));
  await page.getByTestId("tab-alerts").click();
  await shot(page, "alerts");

  // replay mode at the fault, mid-playback controls visible
  await scrubTo(page, m.fault.ts);
  await expect.poll(() => engineLabel(page, m.fault.machine)).toBe("FAULT");
  await page.getByTestId(`machine-item-${m.fault.machine}`).click();
  await page.getByTestId("tab-history").click();
  await shot(page, "replay_mode");
});

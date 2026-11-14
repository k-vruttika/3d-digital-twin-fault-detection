/**
 * Dashboard screenshots from dev mode (mock source, TimescaleDB): scenario injection, a STOP
 * command result, the line_stoppage alert and the sources panel. Run by
 * `python -m twin_backend.e2e_stack screenshots`.
 */
import { expect, test, type Page } from "@playwright/test";

import { expectBadge, openDashboard } from "./helpers.ts";

const MACHINE = "CNC-01";

function env(name: string): string {
  const v = process.env[name];
  if (v === undefined) throw new Error(`${name} is not set`);
  return v;
}

async function shot(page: Page, name: string): Promise<void> {
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${env("E2E_SHOTS_DIR")}/${name}.png` });
}

test("scenario, command result, line stoppage alert, sources", async ({ page }) => {
  await openDashboard(page, env("E2E_OPERATOR_KEY"));
  if ((await page.evaluate(() => document.documentElement.dataset.theme)) !== "dark") await page.getByTestId("theme-toggle").click();
  for (const id of ["CNV-01", MACHINE, "INSP-01", "CNV-02"]) await expectBadge(page, id, ["RUNNING"]);
  await page.getByTestId(`machine-item-${MACHINE}`).click();

  await page.getByTestId("tab-sources").click();
  await expect(page.getByTestId("ingest-stats")).toBeVisible();
  await shot(page, "sources");

  await page.getByTestId("tab-scenario").click();
  await page.getByTestId("scenario-machine").selectOption(MACHINE);
  await page.getByTestId("scenario-fault").selectOption("bearing_wear");
  await page.getByTestId("scenario-inject").click();
  await expect(page.getByTestId("scenario-result")).toContainText("EXECUTED");
  await expectBadge(page, MACHINE, ["WARNING"], 90_000);
  await shot(page, "scenario_warning");

  await expectBadge(page, MACHINE, ["FAULT"], 90_000);
  await page.getByTestId("cmd-stop").click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await shot(page, "command_confirm_stop");
  await page.getByRole("dialog").getByRole("button", { name: "STOP" }).click();
  await expect(page.getByTestId("command-status")).toHaveText("EXECUTED", { timeout: 30_000 });
  await expectBadge(page, "CNV-01", ["BLOCKED"]);
  await expectBadge(page, "INSP-01", ["STARVED"]);
  await page.getByTestId("tab-alerts").click();
  await expect(page.getByTestId("alert-group-activeUnacked")).toContainText("line_stoppage", { timeout: 60_000 });
  await shot(page, "command_result");
});

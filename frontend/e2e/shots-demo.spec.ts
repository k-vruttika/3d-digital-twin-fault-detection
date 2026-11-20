/**
 * What-if and drift screenshots from dev mode with live inference (Phase 11 Part C, ADR 0031):
 * mock source, TimescaleDB, the deployed bundles, the simulator in real time.
 *
 * 1. What-if: project "stop CNC-01" for 60 s on the healthy line. Ghost states, dashed projection
 *    lines and the outcome text; "apply as command" asks first (the dialog is cancelled: nothing
 *    is sent to the machine).
 * 2. Drift: SET_LOAD 120 % on CNC-01 (above every trained load). The `model_drift` alert raises,
 *    the banner appears and the machine's predictions turn DEGRADED.
 *
 * Run by `python -m twin_backend.e2e_stack screenshots` (needs `up` and `deploy-models`).
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

test("what-if projection, then load drift", async ({ page }) => {
  test.setTimeout(1_200_000); // real time: the drift needs its window and sustain time
  await openDashboard(page, env("E2E_OPERATOR_KEY"));
  if ((await page.evaluate(() => document.documentElement.dataset.theme)) !== "dark")
    await page.getByTestId("theme-toggle").click();
  for (const id of ["CNV-01", MACHINE, "INSP-01", "CNV-02"]) await expectBadge(page, id, ["RUNNING"]);
  await page.getByTestId(`machine-item-${MACHINE}`).click();

  // The healthy line, predicted after its warm-up.
  await page.getByTestId("tab-prediction").click();
  await expect(page.getByTestId(`prediction-cell-${MACHINE}-30`)).toContainText("NORMAL", { timeout: 120_000 });

  // 1. What-if: stop the CNC for 60 s.
  await page.getByTestId("tab-whatif").click();
  await expect(page.getByTestId("whatif-run")).toBeEnabled({ timeout: 60_000 });
  await page.getByTestId("whatif-machine").selectOption(MACHINE);
  await page.getByTestId("whatif-type").selectOption("STOP");
  await page.getByTestId("whatif-horizon").selectOption("60");
  await page.getByTestId("whatif-run").click();
  await expect(page.getByTestId("whatif-result")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId(`whatif-ghost-${MACHINE}-end`)).toContainText("IDLE");
  await expect(page.getByTestId("whatif-ghost-INSP-01-end")).toContainText("STARVED");
  await expect(page.getByTestId("whatif-model")).toBeVisible(); // scored by the deployed model
  await expect(page.getByTestId("whatif-chart")).toBeVisible();
  await shot(page, "whatif_projection");
  // the projection chart (measured values solid, projections dashed) is below the ghost table
  await page.getByTestId("whatif-field-vib").click();
  await page.getByTestId("whatif-chart").scrollIntoViewIfNeeded();
  await shot(page, "whatif_chart");
  await page.getByTestId("whatif-field-pFault").click();
  await page.getByTestId("whatif-chart").scrollIntoViewIfNeeded();
  await shot(page, "whatif_probability");
  // apply as command asks first; cancelling sends nothing
  await page.getByTestId("whatif-apply").click();
  await expect(page.getByTestId("whatif-confirm-text")).toContainText(`STOP on ${MACHINE}`);
  await shot(page, "whatif_apply_confirm");
  await page.getByRole("button", { name: "Cancel" }).click();
  await expectBadge(page, MACHINE, ["RUNNING"], 5_000);

  // 2. Drift: a load above anything trained.
  await page.getByTestId("input-load").fill("120");
  await page.getByTestId("cmd-SET_LOAD").click();
  await expect(page.getByTestId("command-status")).toHaveText("EXECUTED", { timeout: 30_000 });
  await expect(page.getByTestId("drift-banner")).toBeVisible({ timeout: 600_000 });
  await expect(page.getByTestId(`drift-note-${MACHINE}`)).toContainText("load PSI");
  await page.getByTestId("tab-prediction").click();
  await expect(page.getByTestId(`prediction-reliability-${MACHINE}`)).toContainText("DEGRADED", { timeout: 60_000 });
  await shot(page, "prediction_drift");
  await page.getByTestId("tab-alerts").click();
  await expect(page.getByTestId("alert-group-activeUnacked")).toContainText("model drift");
  await shot(page, "alerts_drift");
});

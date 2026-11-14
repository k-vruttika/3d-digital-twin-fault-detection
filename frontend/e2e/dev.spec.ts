/**
 * Dev mode with the mock source (simulator publisher + TimescaleDB): inject a fault, watch
 * WARNING then FAULT and its alert, STOP the machine, see its neighbours BLOCKED / STARVED and
 * the line_stoppage alert, acknowledge it, restart the machine (maintenance clears the fault,
 * the simulator then runs it again), and replay back to watch the fault build up.
 */
import { expect, test, type Page } from "@playwright/test";

import { apiUrl, engineLabel, expectBadge, openDashboard, scrubTo } from "./helpers.ts";

const MACHINE = "CNC-01";

function operatorKey(): string {
  const key = process.env.E2E_OPERATOR_KEY;
  if (key === undefined) throw new Error("E2E_OPERATOR_KEY is not set");
  return key;
}

async function alertTs(page: Page, kind: string): Promise<string> {
  const res = await page.request.get(`${apiUrl()}/alerts/history`, {
    params: { from: "2000-01-01T00:00:00Z", to: "2100-01-01T00:00:00Z", machine: MACHINE },
    headers: { "X-API-Key": operatorKey() },
  });
  expect(res.ok()).toBe(true);
  const body = (await res.json()) as { alerts: { kind: string; ts: string; rev: number }[] };
  const first = body.alerts.filter((a) => a.kind === kind).sort((a, b) => a.rev - b.rev)[0];
  if (first === undefined) throw new Error(`no ${kind} alert for ${MACHINE}`);
  return first.ts;
}

test("inject a fault, stop, acknowledge, restart, replay the build-up", async ({ page }) => {
  await openDashboard(page, operatorKey());
  for (const id of ["CNV-01", MACHINE, "INSP-01", "CNV-02"]) await expectBadge(page, id, ["RUNNING"]);

  // inject a fault (simulation only) on the CNC
  await page.getByTestId(`machine-item-${MACHINE}`).click();
  await page.getByTestId("tab-scenario").click();
  await expect(page.getByRole("note")).toContainText("SIMULATION ONLY");
  await page.getByTestId("scenario-machine").selectOption(MACHINE);
  await page.getByTestId("scenario-fault").selectOption("bearing_wear");
  await page.getByTestId("scenario-inject").click();
  await expect(page.getByTestId("scenario-result")).toContainText("EXECUTED");

  // the fault builds up: WARNING first, then FAULT, with its alert
  await expectBadge(page, MACHINE, ["WARNING"], 90_000);
  await expect.poll(() => engineLabel(page, MACHINE)).toBe("WARNING");
  await expectBadge(page, MACHINE, ["FAULT"], 90_000);
  await page.getByTestId("tab-alerts").click();
  await expect(page.getByTestId("alert-group-activeUnacked")).toContainText("machine_fault");

  // STOP (always allowed) with confirmation; the result follows the ack
  await page.getByTestId("cmd-stop").click();
  await page.getByRole("dialog").getByRole("button", { name: "STOP" }).click();
  await expect(page.getByTestId("command-status")).toHaveText(/^(EXECUTED|ACCEPTED)$/, { timeout: 30_000 });
  await expect(page.getByTestId("command-status")).toHaveText("EXECUTED", { timeout: 30_000 });

  // the line stops around the CNC: upstream BLOCKED, downstream STARVED, and line_stoppage
  await expectBadge(page, "CNV-01", ["BLOCKED"]);
  await expectBadge(page, "INSP-01", ["STARVED"]);
  await expect(page.getByTestId("machine-item-CNV-01")).toContainText(`BLOCKED by ${MACHINE}`);
  const unacked = page.getByTestId("alert-group-activeUnacked");
  const stoppage = unacked.locator("tr", { hasText: "line_stoppage" });
  await expect(stoppage).toBeVisible({ timeout: 60_000 });

  // acknowledge it: it moves to "active, acknowledged"
  await stoppage.getByRole("button", { name: "Acknowledge" }).click();
  await expect(page.getByTestId("alert-group-activeAcked")).toContainText("line_stoppage");

  // restart: maintenance clears the fault and the wear; afterwards the simulator runs it again
  await page.getByTestId("cmd-SCHEDULE_MAINTENANCE").click();
  await expect(page.getByTestId("command-status")).toHaveText("EXECUTED", { timeout: 30_000 });
  await expectBadge(page, MACHINE, ["MAINTENANCE"]);
  await expectBadge(page, MACHINE, ["RUNNING"], 90_000);
  await expectBadge(page, "CNV-01", ["RUNNING"], 90_000);
  await expectBadge(page, "INSP-01", ["RUNNING"], 90_000);

  // replay: scrub back and watch the fault build up in the 3D scene
  const warnTs = await alertTs(page, "machine_warning");
  const faultTs = await alertTs(page, "machine_fault");
  expect(Date.parse(warnTs)).toBeLessThan(Date.parse(faultTs));
  await page.getByTestId("enter-replay").click();
  await expect(page.getByTestId("replay-banner")).toBeVisible();
  await expect(page.getByTestId("replay-scrubber")).toBeVisible();
  await scrubTo(page, new Date(Date.parse(warnTs) - 5_000).toISOString());
  await expect.poll(() => engineLabel(page, MACHINE)).toBe("RUNNING");
  await scrubTo(page, warnTs);
  await expect.poll(() => engineLabel(page, MACHINE)).toBe("WARNING");
  await scrubTo(page, faultTs);
  await expect.poll(() => engineLabel(page, MACHINE)).toBe("FAULT");
  // commands are disabled in replay
  await expect(page.getByTestId("cmd-stop")).toBeDisabled();
  await page.getByTestId("exit-replay").click();
  await expectBadge(page, MACHINE, ["RUNNING"]);
});

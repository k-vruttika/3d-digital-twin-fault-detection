/**
 * Prediction panel screenshots from dev mode with live inference (Phase 11, ADR 0030): mock source,
 * TimescaleDB, the deployed bundles, the simulator in real time (see e2e_stack.PREDICT_SIM_SPEED).
 * bearing_wear is injected on CNC-01; the spec checks, in the running app:
 *
 * 1. the autoencoder's "unusual behaviour" alarm while the supervised model still predicts NORMAL:
 *    a purple outline on every part of the machine and no part highlight;
 * 2. the supervised predicted alarm with the bearing highlighted (from the model's attributions),
 *    before the simulator itself reports WARNING where the timing allows.
 *
 * Run by `python -m twin_backend.e2e_stack screenshots` (needs `up` and `deploy-models`).
 */
import { expect, test, type Page } from "@playwright/test";

import { badgeOf, expectBadge, openDashboard } from "./helpers.ts";

const MACHINE = "CNC-01";
const FAULT = "bearing_wear";
const BEARING = "spindle_bearing";

function env(name: string): string {
  const v = process.env[name];
  if (v === undefined) throw new Error(`${name} is not set`);
  return v;
}

async function shot(page: Page, name: string): Promise<void> {
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${env("E2E_SHOTS_DIR")}/${name}.png` });
}

interface Visual {
  unusual: boolean;
  outlineParts: string[];
  outlineColor: number | null;
  parts: string[];
  outlined: string[];
  highlights: [string, number][];
}

/** What the 3D engine shows for a machine (dev-build hook): outline, its colour, highlights. */
async function visual(page: Page, id: string): Promise<Visual | null> {
  return page.evaluate((machineId) => {
    interface Model {
      spec: { unusual: boolean; outlineParts: string[]; outlineColor: number | null } | null;
      parts: Map<string, unknown>;
      highlightMap: ReadonlyMap<string, number>;
      outlineVisible(part: string): boolean;
    }
    const hook = (window as unknown as { __linetwin?: { engine: { machineModel(id: string): Model | undefined } } })
      .__linetwin;
    const m = hook?.engine.machineModel(machineId);
    if (m === undefined || m.spec === null) return null;
    const parts = [...m.parts.keys()];
    return {
      unusual: m.spec.unusual,
      outlineParts: m.spec.outlineParts,
      outlineColor: m.spec.outlineColor,
      parts,
      outlined: parts.filter((p) => m.outlineVisible(p)),
      highlights: [...m.highlightMap.entries()],
    };
  }, id);
}

async function engineColors(page: Page): Promise<{ unusual: number; warning: number }> {
  return page.evaluate(() => {
    const hook = (
      window as unknown as { __linetwin?: { engine: { cfg: { colors: { unusual: number; warning: number } } } } }
    ).__linetwin;
    if (hook === undefined) throw new Error("no engine hook");
    return { unusual: hook.engine.cfg.colors.unusual, warning: hook.engine.cfg.colors.warning };
  });
}

/** A close-up of the machine in the 3D view (camera moved for the picture only). */
async function closeUp(page: Page, name: string): Promise<void> {
  await page.evaluate((id) => {
    const hook = (window as unknown as { __linetwin?: { engine: { focus(id: string, distance?: number): void } } })
      .__linetwin;
    hook?.engine.focus(id, 4.5);
  }, MACHINE);
  await page.waitForTimeout(1500);
  await page.getByTestId("twin-view").screenshot({ path: `${env("E2E_SHOTS_DIR")}/${name}.png` });
}

test("live predictions: unusual behaviour, then the predicted alarm with the bearing highlighted", async ({
  page,
}) => {
  test.setTimeout(1_200_000); // real time: the fault takes about 9 minutes to reach WARNING
  await openDashboard(page, env("E2E_OPERATOR_KEY"));
  if ((await page.evaluate(() => document.documentElement.dataset.theme)) !== "dark")
    await page.getByTestId("theme-toggle").click();
  for (const id of ["CNV-01", MACHINE, "INSP-01", "CNV-02"]) await expectBadge(page, id, ["RUNNING"]);
  await page.getByTestId(`machine-item-${MACHINE}`).click();
  const colors = await engineColors(page);

  // Healthy line: every machine predicted after its warm-up.
  await page.getByTestId("tab-prediction").click();
  await expect(page.getByTestId("prediction-model")).toContainText("ready");
  await expect(page.getByTestId(`prediction-cell-${MACHINE}-30`)).toContainText("NORMAL", { timeout: 120_000 });
  await expect(page.getByTestId("prediction-alarm")).toHaveText("off");
  await shot(page, "prediction_normal");

  // Inject the fault (the shortest training progression, so the model sees a familiar rate).
  await page.getByTestId("tab-scenario").click();
  await page.getByTestId("scenario-machine").selectOption(MACHINE);
  await page.getByTestId("scenario-fault").selectOption(FAULT);
  await page.getByTestId("scenario-progression").fill(env("E2E_PROGRESSION_S"));
  await page.getByTestId("scenario-inject").click();
  await expect(page.getByTestId("scenario-result")).toContainText("EXECUTED");
  await page.getByTestId("tab-prediction").click();

  // 1. Unusual behaviour (autoencoder) before any supervised alarm: whole machine, purple, no part.
  await expect
    .poll(async () => (await visual(page, MACHINE))?.unusual ?? false, {
      timeout: 420_000,
      message: "the unusual_behaviour alert should reach the 3D view",
    })
    .toBe(true);
  const unusual = await visual(page, MACHINE);
  expect(unusual).not.toBeNull();
  if (unusual === null) return;
  expect(await badgeOf(page, MACHINE)).toBe("RUNNING");
  expect(unusual.outlineColor).toBe(colors.unusual);
  expect([...unusual.outlined].sort()).toEqual([...unusual.parts].sort());
  expect(unusual.highlights).toEqual([]);
  await expect(page.getByTestId("prediction-alarm")).toHaveText("off");
  await expect(page.getByTestId("prediction-anomaly")).toContainText("cause unknown");
  await shot(page, "prediction_unusual");
  await page.getByTestId("tab-alerts").click();
  await expect(page.getByTestId("alert-group-activeUnacked")).toContainText("unusual behaviour");
  await shot(page, "alerts_unusual");
  await closeUp(page, "twin_unusual_closeup");

  // 2. The supervised alarm (the evaluated rule) with the bearing highlighted.
  await page.getByTestId("tab-prediction").click();
  await expect(page.getByTestId("prediction-alarm")).toHaveText("on", { timeout: 420_000 });
  const stateAtAlarm = await badgeOf(page, MACHINE);
  // Attributions arrive beside the prediction path and are attached only while recent
  // (ADR 0030), so the highlight can drop out between explanations: shoot while it is on.
  const highlighted = async (): Promise<boolean> => {
    const v = await visual(page, MACHINE);
    if (v === null || v.highlights.length === 0) return false;
    return [...v.highlights].sort((a, b) => b[1] - a[1])[0]?.[0] === BEARING;
  };
  // How steadily the highlight holds during the alarm (reported, not asserted).
  let on = 0;
  const samples = 40;
  for (let i = 0; i < samples; i++) {
    if (await highlighted()) on++;
    await page.waitForTimeout(250);
  }
  test.info().annotations.push({ type: "bearing-highlight-share", description: `${String(on)}/${String(samples)}` });
  console.log(`bearing highlighted in ${String(on)}/${String(samples)} samples over 10 s of the alarm`);
  let captured = false;
  for (let attempt = 0; attempt < 20 && !captured; attempt++) {
    await expect
      .poll(highlighted, { timeout: 60_000, message: "the bearing should be highlighted (strongest part)" })
      .toBe(true);
    const v = await visual(page, MACHINE);
    expect(v?.unusual).toBe(false); // superseded by the supervised alarm
    await expect(page.getByTestId("prediction-attributions")).toContainText(BEARING);
    await page.screenshot({ path: `${env("E2E_SHOTS_DIR")}/prediction_alarm.png` });
    captured = (await highlighted()) && ((await page.getByTestId("prediction-attributions").textContent()) ?? "").includes(BEARING);
  }
  expect(captured, "a screenshot with the bearing highlighted").toBe(true);
  await page.evaluate((id) => {
    const hook = (window as unknown as { __linetwin?: { engine: { focus(id: string, distance?: number): void } } })
      .__linetwin;
    hook?.engine.focus(id, 4.5);
  }, MACHINE);
  await page.waitForTimeout(1500);
  captured = false;
  for (let attempt = 0; attempt < 20 && !captured; attempt++) {
    await expect.poll(highlighted, { timeout: 60_000 }).toBe(true);
    await page.getByTestId("twin-view").screenshot({ path: `${env("E2E_SHOTS_DIR")}/twin_alarm_closeup.png` });
    captured = await highlighted();
  }
  expect(captured, "a close-up with the bearing highlighted").toBe(true);
  await page.getByTestId("tab-alerts").click();
  await expect(page.getByTestId("alert-group-activeUnacked")).toContainText("predicted");
  await shot(page, "alerts_predicted");

  // The simulator's own WARNING: the at-risk part is outlined in the warning colour.
  await expectBadge(page, MACHINE, ["WARNING"], 300_000);
  await expect
    .poll(async () => (await visual(page, MACHINE))?.outlined ?? [], { timeout: 30_000 })
    .toContain(BEARING);
  expect((await visual(page, MACHINE))?.outlineColor).toBe(colors.warning);
  await page.getByTestId("tab-prediction").click();
  await shot(page, "prediction_warning");
  test.info().annotations.push({ type: "state-at-alarm", description: stateAtAlarm });
  console.log(`twin state when the predicted alarm was first seen: ${stateAtAlarm}`);
});

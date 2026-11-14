/** Shared helpers for the dashboard end-to-end specs. */
import { expect, type Page } from "@playwright/test";

export interface Moment {
  machine: string;
  state: string;
  ts: string;
}

/** Key moments of the replay file (found by twin_backend.e2e_stack.key_moments). */
export function moments(): Record<"normal" | "warning" | "fault" | "waiting", Moment> {
  const raw = process.env.E2E_MOMENTS;
  if (raw === undefined) throw new Error("E2E_MOMENTS is not set (run through pytest / e2e_stack)");
  return JSON.parse(raw) as Record<"normal" | "warning" | "fault" | "waiting", Moment>;
}

export function apiUrl(): string {
  const url = process.env.E2E_API_URL;
  if (url === undefined) throw new Error("E2E_API_URL is not set");
  return url;
}

/** The 3D engine's label text for a machine (what the scene shows), via the dev-build hook. */
export async function engineLabel(page: Page, machineId: string): Promise<string | null> {
  return page.evaluate((id) => {
    const hook = (window as unknown as { __linetwin?: { engine: { machineModel(id: string): { labelText: string | null } | undefined } } })
      .__linetwin;
    return hook?.engine.machineModel(id)?.labelText ?? null;
  }, machineId);
}

export async function waitForEngine(page: Page): Promise<void> {
  await page.waitForFunction(() => (window as unknown as { __linetwin?: unknown }).__linetwin !== undefined, null, {
    timeout: 60_000,
  });
}

/** Move the replay scrubber to `iso` (Playwright's fill sets a range input like a drag does). */
export async function scrubTo(page: Page, iso: string): Promise<void> {
  await page.getByTestId("replay-scrubber").fill(String(Date.parse(iso)));
}

export async function badgeOf(page: Page, machineId: string): Promise<string> {
  return (await page.getByTestId(`machine-item-${machineId}`).locator(".badge").first().textContent()) ?? "";
}

export async function expectBadge(page: Page, machineId: string, states: string[], timeout = 60_000): Promise<void> {
  await expect
    .poll(async () => badgeOf(page, machineId), { timeout, message: `${machineId} should become ${states.join(" or ")}` })
    .toMatch(new RegExp(`^(${states.join("|")})$`));
}

/** Open the dashboard; with `operatorKey` (a backend with keys) enter it first. */
export async function openDashboard(page: Page, operatorKey?: string): Promise<void> {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  if (operatorKey !== undefined) {
    // without a key the backend refuses: the dashboard says so and waits for one
    await expect(page.getByTestId("role")).toContainText("no key", { timeout: 30_000 });
    await applyKey(page, operatorKey);
  }
  await expect(page.getByTestId("machine-list")).toBeVisible({ timeout: 60_000 });
  await waitForEngine(page);
  expect(errors, "no page errors while loading").toEqual([]);
}

export async function applyKey(page: Page, key: string): Promise<void> {
  await page.locator("#api-key").fill(key);
  await page.getByRole("button", { name: "Apply" }).click();
  await expect(page.getByTestId("role")).toContainText("operator", { timeout: 30_000 });
  await expect(page.getByTestId("connection-pill")).toContainText("Connected", { timeout: 30_000 });
}

/** The chart (canvas, with its axis labels) lies fully inside the bottom panel's visible area. */
export async function expectChartInsidePanel(page: Page, testId: string): Promise<void> {
  const canvas = page.getByTestId(testId).locator("canvas");
  await expect(canvas).toBeVisible();
  const panel = await page.locator("section.bottom").boundingBox();
  const box = await canvas.boundingBox();
  if (panel === null || box === null) throw new Error("no layout box");
  expect(box.height, `${testId} has room to draw`).toBeGreaterThanOrEqual(120);
  expect(box.y + box.height, `${testId} bottom (axis labels) inside the panel`).toBeLessThanOrEqual(panel.y + panel.height + 0.5);
  expect(box.x + box.width).toBeLessThanOrEqual(panel.x + panel.width + 0.5);
}

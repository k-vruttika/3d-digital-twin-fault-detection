// Screenshots of the 3D twin dev page in headless Chromium (software WebGL).
//
// Run with `uv run poe screenshots`. Starts Vite through its Node API on a free port, opens
// dev.html in synthetic mode with time frozen (deterministic), and writes PNGs to
// experiments/results/frontend_screenshots/. Cross-platform: plain Node, no shell.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { createServer } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const outDir = resolve(root, "..", "experiments", "results", "frontend_screenshots");
const VIEWPORT = { width: 1280, height: 720 };
const FREEZE_S = 10;

/** One state per screenshot (ids from src/engine/synthetic.ts STATE_SCENE). */
const STATE_SHOTS = [
  ["state_warning", "S-WARN"],
  ["state_fault", "S-FAULT"],
  ["state_blocked", "S-BLOCK"],
  ["state_starved", "S-STARVE"],
  ["state_maintenance", "S-MAINT"],
  ["state_stale", "S-STALE"],
  ["state_offline", "S-OFF"],
];

async function main() {
  mkdirSync(outDir, { recursive: true });
  const server = await createServer({
    root,
    configFile: resolve(root, "vite.config.ts"),
    logLevel: "warn",
    server: { host: "127.0.0.1", port: 5199, strictPort: false },
  });
  await server.listen();
  const base = server.resolvedUrls?.local[0] ?? "http://127.0.0.1:5199/";
  const browser = await chromium.launch({
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
  });
  const written = [];
  const problems = [];
  try {
    const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: 1 });
    page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
    page.on("console", (m) => {
      if (m.type() === "error") problems.push(`console error: ${m.text()}`);
    });

    const open = async (query) => {
      await page.goto(`${base}dev.html?${query}&freeze=${FREEZE_S}`, { waitUntil: "load" });
      await page.waitForFunction(() => window.__linetwinDev !== undefined, null, { timeout: 30_000 });
      await page.waitForTimeout(800); // a few frames
    };
    const shot = async (name) => {
      const path = resolve(outDir, `${name}.png`);
      await page.screenshot({ path });
      written.push(name);
    };

    await open("synthetic=line");
    const webgl = await page.evaluate(() => {
      const c = document.querySelector("#twin canvas");
      return c instanceof HTMLCanvasElement && (c.getContext("webgl2") !== null || c.getContext("webgl") !== null);
    });
    if (!webgl) throw new Error("no WebGL context in headless Chromium (software rendering unavailable)");
    await shot("line_running");

    // Hover tooltip over the CNC.
    const pos = await page.evaluate(() => window.__linetwinDev.screenPosition("CNC-01"));
    await page.mouse.move(pos.x, pos.y);
    await page.waitForTimeout(400);
    await shot("tooltip_cnc");
    await page.mouse.move(5, 5);

    await open("synthetic=states");
    await shot("states_overview");
    for (const [name, id] of STATE_SHOTS) {
      await page.evaluate((m) => window.__linetwinDev.focus(m, 8), id);
      await page.waitForTimeout(300);
      await shot(name);
    }

    await open("synthetic=single");
    await shot("line_single");

    await open("synthetic=20");
    await shot("line_20_machines");
  } finally {
    await browser.close();
    await server.close();
  }
  const summary = { viewport: VIEWPORT, freezeS: FREEZE_S, screenshots: written, problems };
  writeFileSync(resolve(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`wrote ${String(written.length)} screenshots to ${outDir}`);
  if (problems.length > 0) {
    console.error(problems.join("\n"));
    process.exitCode = 1;
  }
}

main().catch((e) => {
  console.error(String(e?.stack ?? e));
  process.exit(1);
});

/**
 * Playwright end-to-end tests of the dashboard (run by pytest through twin_backend.e2e_stack,
 * which starts the backend and Vite and passes their URLs in the environment).
 *
 * Retries are 0 everywhere (locally and in CI): a flaky test must show up as a failure. Traces
 * and screenshots are kept for failed tests.
 */
import { defineConfig } from "@playwright/test";

const outputDir = process.env.E2E_OUTPUT_DIR ?? "test-results";

export default defineConfig({
  testDir: "e2e",
  outputDir,
  retries: 0,
  workers: 1,
  fullyParallel: false,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL,
    browserName: "chromium",
    viewport: { width: 1280, height: 800 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: {
      // Software WebGL: works headless on CI machines without a GPU.
      args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
    },
  },
  projects: [
    { name: "minimal", testMatch: "minimal.spec.ts" },
    { name: "dev", testMatch: "dev.spec.ts" },
    { name: "shots-minimal", testMatch: "shots-minimal.spec.ts" },
    { name: "shots-dev", testMatch: "shots-dev.spec.ts" },
  ],
});

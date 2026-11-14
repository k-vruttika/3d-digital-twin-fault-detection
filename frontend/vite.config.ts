import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },
  build: {
    rolldownOptions: {
      // index.html: the (future) React dashboard; dev.html: the full-screen 3D twin dev page.
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        dev: resolve(import.meta.dirname, "dev.html"),
      },
      output: {
        // three.js and Chart.js get their own chunks; the engine and the charts are loaded with
        // dynamic import() by the pages, so neither is in the initial bundle.
        codeSplitting: {
          groups: [
            // three ships as two modules (three.core.js + three.module.js); maxSize keeps them
            // in separate chunks, each under the 500 kB warning limit.
            { name: "three", test: /node_modules[\\/]three[\\/]/, maxSize: 450_000 },
            { name: "chartjs", test: /node_modules[\\/]chart\.js[\\/]/ },
          ],
        },
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});

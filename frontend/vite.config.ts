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
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});

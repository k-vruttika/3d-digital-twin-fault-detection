/**
 * Full-screen dev page for the 3D twin (no dashboard): http://127.0.0.1:5173/dev.html
 *
 * Live mode (default): GET /line, then the WebSocket stream of the running backend
 * (`uv run poe minimal` or `uv run poe dev`).
 *
 * URL options: `fps` shows the FPS counter. `synthetic=<n|line|single|states>` runs without a
 * backend on generated data (see synthetic.ts); `freeze=<s>` freezes synthetic time (screenshots).
 */
import { bindStream } from "../api/bind.ts";
import { ApiError, RestClient } from "../api/rest.ts";
import type { LineResponse } from "../api/types.ts";
import { StreamClient, streamUrl } from "../api/ws.ts";
import { TwinEngine } from "../engine/TwinEngine.ts";
import { startSynthetic } from "./syntheticMode.ts";

const params = new URLSearchParams(window.location.search);
const container = document.getElementById("twin");
const statusEl = document.getElementById("status");
const selectionEl = document.getElementById("selection");
if (container === null || statusEl === null || selectionEl === null) {
  throw new Error("dev.html is missing #twin, #status or #selection");
}

function setStatus(status: string, text: string): void {
  if (statusEl === null) return;
  statusEl.dataset.status = status;
  statusEl.textContent = text;
}

const synthetic = params.get("synthetic");
const freeze = params.get("freeze");
const frozenAt = freeze === null ? null : Number(freeze) * 1000;
const engine = new TwinEngine(container, {
  fps: params.has("fps"),
  ...(frozenAt === null ? {} : { now: () => frozenAt, wallClock: () => Date.parse("2026-01-01T00:00:10Z") }),
});

engine.on("select", ({ machineId }) => {
  selectionEl.hidden = machineId === null;
  selectionEl.textContent = machineId === null ? "" : `selected: ${machineId}`;
});

const cleanups: (() => void)[] = [];

if (synthetic !== null) {
  cleanups.push(startSynthetic(engine, synthetic, { frozen: frozenAt !== null, setStatus }));
} else {
  void startLive();
}

async function startLive(): Promise<void> {
  const baseUrl = import.meta.env.VITE_LINETWIN_API_URL ?? "http://127.0.0.1:8000";
  const apiKey = import.meta.env.VITE_LINETWIN_API_KEY;
  const rest = new RestClient({ baseUrl, apiKey });

  let line: LineResponse | null = null;
  let waitMs = 500;
  while (line === null) {
    try {
      setStatus("connecting", `loading layout from ${baseUrl} …`);
      line = await rest.getLine();
    } catch (error) {
      if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        setStatus(
          "unauthorized",
          `${baseUrl} needs an API key: set VITE_LINETWIN_API_KEY in frontend/.env.local (HTTP ${String(error.status)}); retrying`,
        );
      } else {
        setStatus("waiting", `backend not reachable at ${baseUrl} (${String(error)}); retrying`);
      }
      await new Promise((r) => setTimeout(r, waitMs));
      waitMs = Math.min(waitMs * 2, 10_000);
    }
  }
  engine.loadLayout(line);
  const client = new StreamClient({
    url: streamUrl(baseUrl),
    apiKey,
    heartbeatMs: 10_000,
    heartbeatTimeoutMs: 5_000,
  });
  cleanups.push(
    bindStream(client, engine, {
      fetchLine: () => rest.getLine(),
      initialLine: line,
      onError: (error) => {
        setStatus("error", `update failed: ${String(error)}`);
      },
    }),
  );
  cleanups.push(
    client.on("status", (status, detail) => {
      const lineId = line.lineId;
      const text = status === "live" ? `live · ${lineId}` : `${status}${detail ? ` · ${detail}` : ""}`;
      setStatus(status, text);
    }),
  );
  cleanups.push(() => {
    client.close();
  });
  client.connect();
}

// Expose a small hook for the screenshot script (dev server only).
const devHook = {
  engine,
  screenPosition: (id: string) => engine.screenPosition(id),
  focus: (id: string, distance?: number) => {
    engine.focus(id, distance);
  },
};
(window as unknown as { __linetwinDev?: typeof devHook }).__linetwinDev = devHook;

const onResize = (): void => {
  engine.resize();
};
window.addEventListener("resize", onResize);
cleanups.push(() => {
  window.removeEventListener("resize", onResize);
});

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    for (const c of cleanups) c();
    engine.dispose();
  });
}

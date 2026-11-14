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
import type { TwinEngine } from "../engine/TwinEngine.ts";

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
const cleanups: (() => void)[] = [];

setStatus("connecting", "loading the 3D engine …");
void start(container, selectionEl);

/** three.js and the engine load as their own chunk (dynamic import), after the page shell. */
async function start(twin: HTMLElement, selection: HTMLElement): Promise<void> {
  const [{ TwinEngine }, { startSynthetic }] = await Promise.all([
    import("../engine/TwinEngine.ts"),
    import("./syntheticMode.ts"),
  ]);
  const engine = new TwinEngine(twin, {
    fps: params.has("fps"),
    ...(frozenAt === null ? {} : { now: () => frozenAt, wallClock: () => Date.parse("2026-01-01T00:00:10Z") }),
  });
  engine.on("select", ({ machineId }) => {
    selection.hidden = machineId === null;
    selection.textContent = machineId === null ? "" : `selected: ${machineId}`;
  });
  exposeDevHook(engine);
  const onResize = (): void => {
    engine.resize();
  };
  window.addEventListener("resize", onResize);
  cleanups.push(() => {
    window.removeEventListener("resize", onResize);
    engine.dispose();
  });
  if (synthetic !== null) {
    cleanups.push(startSynthetic(engine, synthetic, { frozen: frozenAt !== null, setStatus }));
  } else {
    await startLive(engine);
  }
}

async function startLive(engine: TwinEngine): Promise<void> {
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

/** A small hook for the screenshot script. */
function exposeDevHook(engine: TwinEngine): void {
  const devHook = {
    engine,
    screenPosition: (id: string) => engine.screenPosition(id),
    focus: (id: string, distance?: number) => {
      engine.focus(id, distance);
    },
  };
  (window as unknown as { __linetwinDev?: typeof devHook }).__linetwinDev = devHook;
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    for (const c of cleanups) c();
  });
}

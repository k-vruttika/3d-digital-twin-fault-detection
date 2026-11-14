/**
 * Wires a StreamClient to a twin engine.
 *
 * - Every snapshot (first connect and each reconnect) is applied in full. After a reconnect the
 *   layout is fetched again first (`GET /line`) and reloaded only if it changed, so a backend
 *   restarted with another line config shows the new line.
 * - Stream messages go to the matching engine method, in arrival order (a reconnect's layout
 *   fetch holds later messages back until it is done).
 * - In replay mode live data is ignored; the engine shows what the replay applies.
 * - The engine is told when the link is up (`live`) or down (anything else).
 */
import type {
  Alert,
  Command,
  CommandAck,
  DataQualityReport,
  LineResponse,
  LineViewModel,
  Prediction,
  Reading,
  StateChange,
  WsSnapshotData,
} from "./types.ts";
import type { StreamClient, StreamMessage } from "./ws.ts";

/** What bindStream needs from the engine (TwinEngine implements it). */
export interface EngineSink {
  readonly mode: "live" | "replay";
  loadLayout(line: LineResponse): void;
  applySnapshot(snapshot: WsSnapshotData): void;
  applyReading(reading: Reading): void;
  applyStateChange(change: StateChange): void;
  applyDataQuality(report: DataQualityReport): void;
  applyAlert(alert: Alert): void;
  applyLine(view: LineViewModel): void;
  setPrediction(prediction: Prediction): void;
  setConnected(connected: boolean): void;
  /** Command audit events (the dashboard's command panel; the 3D scene ignores them). */
  applyCommand?(command: Command): void;
  applyAck?(ack: CommandAck): void;
}

export interface BindOptions {
  /** Fetches the layout (RestClient.getLine). */
  fetchLine: () => Promise<LineResponse>;
  /** The layout already loaded into the engine, if any. */
  initialLine?: LineResponse | null;
  onError?: (error: unknown) => void;
}

export function layoutKey(line: LineResponse): string {
  return JSON.stringify(line);
}

export function routeMessage(engine: EngineSink, msg: StreamMessage): void {
  switch (msg.type) {
    case "reading":
      engine.applyReading(msg.data);
      return;
    case "state":
      engine.applyStateChange(msg.data);
      return;
    case "line":
      engine.applyLine(msg.data);
      return;
    case "dataQuality":
      engine.applyDataQuality(msg.data);
      return;
    case "alert":
      engine.applyAlert(msg.data);
      return;
    case "prediction":
      engine.setPrediction(msg.data);
      return;
    case "command":
      engine.applyCommand?.(msg.data);
      return;
    case "ack":
      engine.applyAck?.(msg.data);
      return;
    default:
      // dropped / error: nothing to show
      return;
  }
}

/** Returns an unbind function (removes the client listeners; does not close the client). */
export function bindStream(client: StreamClient, engine: EngineSink, opts: BindOptions): () => void {
  let currentKey = opts.initialLine ? layoutKey(opts.initialLine) : null;
  let chain: Promise<void> = Promise.resolve();
  let unbound = false;

  const enqueue = (task: () => Promise<void> | void): void => {
    chain = chain
      .then(async () => {
        if (!unbound) await task();
      })
      .catch((error: unknown) => {
        opts.onError?.(error);
      });
  };

  const offSnapshot = client.on("snapshot", (data, info) => {
    enqueue(async () => {
      if (info.reconnect || currentKey === null) {
        const line = await opts.fetchLine();
        const key = layoutKey(line);
        if (key !== currentKey) {
          engine.loadLayout(line);
          currentKey = key;
        }
      }
      engine.setConnected(true);
      if (engine.mode === "live") engine.applySnapshot(data);
    });
  });
  const offMessage = client.on("message", (msg) => {
    enqueue(() => {
      if (engine.mode === "live") routeMessage(engine, msg);
    });
  });
  const offStatus = client.on("status", (status) => {
    if (status !== "live") {
      enqueue(() => {
        engine.setConnected(false);
      });
    }
  });

  return () => {
    unbound = true;
    offSnapshot();
    offMessage();
    offStatus();
  };
}

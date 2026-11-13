import { describe, expect, it } from "vitest";

import { bindStream, routeMessage, type EngineSink } from "./bind.ts";
import type { LineResponse, WsSnapshotData } from "./types.ts";
import type { StreamClient } from "./ws.ts";

type Handler = (...args: never[]) => void;

/** A StreamClient stand-in whose events the test fires by hand. */
class FakeClient {
  readonly handlers = new Map<string, Set<Handler>>();
  on(event: string, cb: Handler): () => void {
    let set = this.handlers.get(event);
    if (set === undefined) {
      set = new Set();
      this.handlers.set(event, set);
    }
    set.add(cb);
    return () => set.delete(cb);
  }
  emit(event: string, ...args: unknown[]): void {
    for (const cb of this.handlers.get(event) ?? []) (cb as (...a: unknown[]) => void)(...args);
  }
  get listenerCount(): number {
    let n = 0;
    for (const s of this.handlers.values()) n += s.size;
    return n;
  }
}

class RecordingEngine implements EngineSink {
  mode: "live" | "replay" = "live";
  readonly calls: string[] = [];
  loadLayout(line: LineResponse): void {
    this.calls.push(`layout:${line.lineId}`);
  }
  applySnapshot(s: WsSnapshotData): void {
    this.calls.push(`snapshot:${String(s.asOf)}`);
  }
  applyReading(): void {
    this.calls.push("reading");
  }
  applyStateChange(): void {
    this.calls.push("state");
  }
  applyDataQuality(): void {
    this.calls.push("dq");
  }
  applyAlert(): void {
    this.calls.push("alert");
  }
  applyLine(): void {
    this.calls.push("line");
  }
  setPrediction(): void {
    this.calls.push("prediction");
  }
  setConnected(c: boolean): void {
    this.calls.push(`connected:${String(c)}`);
  }
}

const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
const line = (id: string): LineResponse =>
  ({ lineId: id, sensors: {}, machines: [], buffers: [], topology: { order: [], upstream: {}, downstream: {} } });
const snap = (asOf: string): WsSnapshotData => ({
  asOf,
  machines: {},
  line: { status: "RUNNING", rootCauses: {}, impacted: {} },
  dataQuality: {},
  alerts: [],
});

describe("bindStream", () => {
  it("applies the snapshot, then messages in order; reloads the layout only when it changed", async () => {
    const client = new FakeClient();
    const engine = new RecordingEngine();
    let served = line("line1");
    let fetches = 0;
    const unbind = bindStream(client as unknown as StreamClient, engine, {
      fetchLine: () => {
        fetches += 1;
        return Promise.resolve(served);
      },
      initialLine: line("line1"),
    });
    client.emit("snapshot", snap("t0"), { reconnect: false });
    client.emit("message", { type: "reading", data: {} });
    await tick();
    expect(fetches).toBe(0); // first snapshot: the layout passed in is current
    expect(engine.calls).toEqual(["connected:true", "snapshot:t0", "reading"]);

    // Reconnect with the same layout: fetched, not reloaded.
    engine.calls.length = 0;
    client.emit("status", "reconnecting");
    client.emit("snapshot", snap("t1"), { reconnect: true });
    client.emit("message", { type: "state", data: {} });
    await tick();
    expect(fetches).toBe(1);
    expect(engine.calls).toEqual(["connected:false", "connected:true", "snapshot:t1", "state"]);

    // Reconnect to a backend with another line: reloaded before the snapshot.
    engine.calls.length = 0;
    served = line("line1_single");
    client.emit("snapshot", snap("t2"), { reconnect: true });
    await tick();
    expect(engine.calls).toEqual(["layout:line1_single", "connected:true", "snapshot:t2"]);

    unbind();
    expect(client.listenerCount).toBe(0);
  });

  it("ignores live data in replay mode", async () => {
    const client = new FakeClient();
    const engine = new RecordingEngine();
    engine.mode = "replay";
    bindStream(client as unknown as StreamClient, engine, {
      fetchLine: () => Promise.resolve(line("l")),
      initialLine: line("l"),
    });
    client.emit("snapshot", snap("t0"), { reconnect: false });
    client.emit("message", { type: "reading", data: {} });
    await tick();
    expect(engine.calls).toEqual(["connected:true"]);
  });

  it("reports layout fetch errors and keeps going", async () => {
    const client = new FakeClient();
    const engine = new RecordingEngine();
    const errors: unknown[] = [];
    bindStream(client as unknown as StreamClient, engine, {
      fetchLine: () => Promise.reject(new Error("down")),
      onError: (e) => errors.push(e),
    });
    client.emit("snapshot", snap("t0"), { reconnect: false });
    client.emit("message", { type: "alert", data: {} });
    await tick();
    expect(errors).toHaveLength(1);
    expect(engine.calls).toEqual(["alert"]);
  });
});

describe("routeMessage", () => {
  it("sends each message type to its engine method", () => {
    const engine = new RecordingEngine();
    for (const type of ["reading", "state", "line", "dataQuality", "alert", "prediction", "command", "ack", "dropped", "error"]) {
      routeMessage(engine, { type, data: {} } as never);
    }
    expect(engine.calls).toEqual(["reading", "state", "line", "dq", "alert", "prediction"]);
  });
});

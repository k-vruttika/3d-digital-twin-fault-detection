import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Backoff, mulberry32 } from "./backoff.ts";
import type { WsServerMessage, WsSnapshotData } from "./types.ts";
import {
  CLOSE_TOO_SLOW,
  CLOSE_UNAUTHORIZED,
  StreamClient,
  streamUrl,
  type SocketLike,
  type StreamMessage,
  type StreamStatus,
} from "./ws.ts";

class FakeSocket implements SocketLike {
  readonly sent: Record<string, unknown>[] = [];
  closed: { code?: number | undefined; reason?: string | undefined } | null = null;
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string): void {
    this.sent.push(JSON.parse(data) as Record<string, unknown>);
  }

  close(code?: number, reason?: string): void {
    this.closed = { code, reason };
  }

  // server side
  open(): void {
    this.onopen?.({});
  }

  receive(msg: WsServerMessage | Record<string, unknown>): void {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }

  serverClose(code: number, reason = ""): void {
    this.onclose?.({ code, reason });
  }
}

const SNAPSHOT: WsSnapshotData = {
  asOf: "2026-01-01T00:00:00Z",
  machines: {},
  line: { status: "RUNNING", rootCauses: {}, impacted: {} },
  dataQuality: {},
  alerts: [],
};

function setup(opts: Partial<ConstructorParameters<typeof StreamClient>[0]> = {}) {
  const sockets: FakeSocket[] = [];
  const client = new StreamClient({
    url: "ws://127.0.0.1:8000/stream",
    socketFactory: (url) => {
      const s = new FakeSocket(url);
      sockets.push(s);
      return s;
    },
    backoff: { baseMs: 100, factor: 2, maxMs: 1000, jitter: 0, seed: 1 },
    ...opts,
  });
  const snapshots: { data: WsSnapshotData; reconnect: boolean }[] = [];
  const messages: StreamMessage[] = [];
  const statuses: StreamStatus[] = [];
  client.on("snapshot", (data, info) => snapshots.push({ data, reconnect: info.reconnect }));
  client.on("message", (m) => messages.push(m));
  client.on("status", (s) => statuses.push(s));
  const last = (): FakeSocket => {
    const s = sockets.at(-1);
    if (s === undefined) throw new Error("no socket");
    return s;
  };
  /** Server side of a successful handshake on the newest socket. */
  const handshake = (snapshot: WsSnapshotData = SNAPSHOT): void => {
    const s = last();
    s.open();
    s.receive({ type: "auth", data: { principal: "victor", role: "viewer" } });
    s.receive({ type: "subscribed", data: { machines: null, channels: ["readings"] } });
    s.receive({ type: "snapshot", data: snapshot });
  };
  return { client, sockets, snapshots, messages, statuses, last, handshake };
}

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("StreamClient handshake", () => {
  it("authenticates with the first frame, subscribes only after the auth reply", () => {
    const t = setup({ apiKey: "key-1", machines: ["CNC-01"], channels: ["readings", "states"] });
    t.client.connect();
    const s = t.last();
    expect(s.url).toBe("ws://127.0.0.1:8000/stream");
    expect(s.sent).toEqual([]);
    s.open();
    expect(s.sent).toEqual([{ type: "auth", apiKey: "key-1" }]);
    expect(t.client.status).toBe("authenticating");
    s.receive({ type: "auth", data: { principal: "victor", role: "viewer" } });
    expect(s.sent[1]).toEqual({ type: "subscribe", machines: ["CNC-01"], channels: ["readings", "states"] });
    s.receive({ type: "subscribed", data: { machines: ["CNC-01"], channels: ["readings", "states"] } });
    s.receive({ type: "snapshot", data: SNAPSHOT });
    expect(t.snapshots).toEqual([{ data: SNAPSHOT, reconnect: false }]);
    expect(t.client.status).toBe("live");
  });

  it("sends auth without a key for anonymous read", () => {
    const t = setup();
    t.client.connect();
    t.last().open();
    expect(t.last().sent[0]).toEqual({ type: "auth" });
  });

  it("delivers stream messages but not control frames", () => {
    const t = setup();
    t.client.connect();
    t.handshake();
    const reading = {
      type: "reading",
      data: { schemaVersion: 2, machineId: "CNC-01", seq: 1, ts: "2026-01-01T00:00:01Z", state: "RUNNING", speed: 1, temp: 1, vib: 1, load: 1, current: 1, runTime: 1, quality: "OK" },
    } as const;
    t.last().receive(reading);
    t.last().receive({ type: "pong", data: {} });
    t.last().receive({ type: "dropped", data: { count: 3 } });
    t.last().onmessage?.({ data: "not json" });
    t.last().onmessage?.({ data: 42 });
    expect(t.messages.map((m) => m.type)).toEqual(["reading", "dropped"]);
  });
});

describe("StreamClient reconnect", () => {
  it("reconnects with exponential backoff, re-subscribes and re-applies a fresh snapshot", () => {
    const t = setup({ machines: ["CNC-01"] });
    t.client.connect();
    t.handshake();
    t.client.subscribe(["INSP-01"], ["states"]); // selection changed while live
    expect(t.last().sent.at(-1)).toEqual({ type: "subscribe", machines: ["INSP-01"], channels: ["states"] });

    t.last().serverClose(1006);
    expect(t.client.status).toBe("reconnecting");
    // Three failed attempts: 100, 200, 400 ms.
    for (const delay of [100, 200, 400]) {
      const before = t.sockets.length;
      vi.advanceTimersByTime(delay - 1);
      expect(t.sockets.length).toBe(before);
      vi.advanceTimersByTime(1);
      expect(t.sockets.length).toBe(before + 1);
      t.last().serverClose(1006);
    }
    // Fourth attempt (800 ms) succeeds.
    vi.advanceTimersByTime(800);
    const fresh = { ...SNAPSHOT, asOf: "2026-01-01T00:05:00Z" };
    t.handshake(fresh);
    expect(t.last().sent[0]).toEqual({ type: "auth" });
    expect(t.last().sent[1]).toEqual({ type: "subscribe", machines: ["INSP-01"], channels: ["states"] });
    expect(t.snapshots.map((s) => [s.data.asOf, s.reconnect])).toEqual([
      ["2026-01-01T00:00:00Z", false],
      ["2026-01-01T00:05:00Z", true],
    ]);
    expect(t.client.connections).toBe(5);
  });

  it("reconnects after the server's 4408 slow-client close", () => {
    const t = setup();
    t.client.connect();
    t.handshake();
    t.last().serverClose(CLOSE_TOO_SLOW, "too slow");
    expect(t.client.status).toBe("reconnecting");
    vi.advanceTimersByTime(100);
    expect(t.sockets).toHaveLength(2);
    t.handshake();
    expect(t.snapshots.at(-1)?.reconnect).toBe(true);
  });

  it("resets the backoff after a successful snapshot", () => {
    const t = setup();
    t.client.connect();
    t.handshake();
    t.last().serverClose(1006);
    vi.advanceTimersByTime(100);
    t.last().serverClose(1006); // next delay would be 200
    vi.advanceTimersByTime(200);
    t.handshake(); // success: reset
    t.last().serverClose(1006);
    const before = t.sockets.length;
    vi.advanceTimersByTime(100);
    expect(t.sockets.length).toBe(before + 1);
  });

  it("does not reconnect after 4401 (bad key)", () => {
    const t = setup({ apiKey: "wrong" });
    t.client.connect();
    t.last().open();
    t.last().receive({ type: "error", data: { reason: "invalid or missing API key" } });
    t.last().serverClose(CLOSE_UNAUTHORIZED, "unauthorized");
    expect(t.client.status).toBe("unauthorized");
    vi.advanceTimersByTime(60_000);
    expect(t.sockets).toHaveLength(1);
    expect(t.client.pendingTimers).toBe(0);
  });

  it("detects a half-open socket with the heartbeat and reconnects", () => {
    const t = setup({ heartbeatMs: 1000, heartbeatTimeoutMs: 500 });
    t.client.connect();
    t.handshake();
    vi.advanceTimersByTime(1000);
    expect(t.last().sent.at(-1)).toEqual({ type: "ping" });
    t.last().receive({ type: "pong", data: {} }); // healthy
    vi.advanceTimersByTime(1000);
    expect(t.last().sent.at(-1)).toEqual({ type: "ping" });
    vi.advanceTimersByTime(500); // no pong: lost
    expect(t.sockets[0]?.closed?.code).toBe(4000);
    expect(t.client.status).toBe("reconnecting");
    vi.advanceTimersByTime(100);
    expect(t.sockets).toHaveLength(2);
  });

  it("close() stops reconnecting and leaves no timers", () => {
    const t = setup({ heartbeatMs: 1000 });
    t.client.connect();
    t.handshake();
    t.last().serverClose(1006);
    expect(t.client.pendingTimers).toBe(1);
    t.client.close();
    expect(t.client.pendingTimers).toBe(0);
    vi.advanceTimersByTime(60_000);
    expect(t.sockets).toHaveLength(1);
    expect(t.client.status).toBe("closed");

    const u = setup({ heartbeatMs: 1000 });
    u.client.connect();
    u.handshake();
    expect(u.client.pendingTimers).toBe(1); // heartbeat
    u.client.close();
    expect(u.last().closed?.code).toBe(1000);
    expect(u.client.pendingTimers).toBe(0);
    // Events from the old socket are ignored after close.
    u.last().receive({ type: "snapshot", data: SNAPSHOT });
    expect(u.snapshots).toHaveLength(1);
  });
});

describe("backoff and URLs", () => {
  it("grows exponentially up to the cap, with reproducible jitter", () => {
    const b = new Backoff({ baseMs: 100, factor: 2, maxMs: 1000, jitter: 0, seed: 1 });
    expect([1, 2, 3, 4, 5, 6].map(() => b.next())).toEqual([100, 200, 400, 800, 1000, 1000]);
    const j1 = new Backoff({ baseMs: 100, factor: 2, maxMs: 1000, jitter: 0.5, seed: 9 });
    const j2 = new Backoff({ baseMs: 100, factor: 2, maxMs: 1000, jitter: 0.5, seed: 9 });
    const a = [1, 2, 3].map(() => j1.next());
    expect(a).toEqual([1, 2, 3].map(() => j2.next()));
    expect(a[0]).toBeGreaterThanOrEqual(50);
    expect(a[0]).toBeLessThanOrEqual(100);
    expect(() => new Backoff({ baseMs: 0, factor: 2, maxMs: 1, jitter: 0, seed: 1 })).toThrow();
    expect(mulberry32(3)()).toBe(mulberry32(3)());
  });

  it("derives the stream URL from the API base URL", () => {
    expect(streamUrl("http://127.0.0.1:8000")).toBe("ws://127.0.0.1:8000/stream");
    expect(streamUrl("https://twin.example/api/")).toBe("wss://twin.example/api/stream");
  });
});

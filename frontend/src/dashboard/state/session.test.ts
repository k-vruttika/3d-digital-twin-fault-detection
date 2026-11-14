// @vitest-environment jsdom
/** Session: key check, role, stream with the key in the first frame; never in a URL or localStorage. */
import { afterEach, describe, expect, it } from "vitest";

import type { SocketLike } from "../../api/ws.ts";
import { LINE } from "../../test/fakes.ts";
import { BASE_URL, fakeFetch, pathOf } from "../../test/dashboard.tsx";
import { DashboardModel } from "./model.ts";
import { Session } from "./session.ts";

class FakeSocket implements SocketLike {
  sent: string[] = [];
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  constructor(readonly url: string) {}
  send(data: string): void {
    this.sent.push(data);
  }
  close(): void {
    // closed by the client
  }
}

const SECRET = "op-secret-key-123";

/** Everything a Storage holds, as text. */
function dump(storage: Storage): string {
  const out: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const k = storage.key(i);
    if (k !== null) out.push(`${k}=${storage.getItem(k) ?? ""}`);
  }
  return out.join(" | ");
}

afterEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

describe("Session", () => {
  it("checks the key, shows the role and connects the stream with the key in the first frame only", async () => {
    const { fetch, calls } = fakeFetch([
      (c) => (pathOf(c) === "/auth/me" ? { body: { principal: "alice", role: "operator", anonymous: false } } : undefined),
      (c) => (pathOf(c) === "/line" ? { body: LINE } : undefined),
    ]);
    const sockets: FakeSocket[] = [];
    const model = new DashboardModel();
    const session = new Session(model, {
      baseUrl: BASE_URL,
      fetch,
      storage: sessionStorage,
      socketFactory: (url) => {
        const s = new FakeSocket(url);
        sockets.push(s);
        return s;
      },
    });
    await session.applyKey(SECRET, false);
    expect(model.auth.get()).toMatchObject({ status: "ok", role: "operator", principal: "alice" });
    expect(model.line.get()?.lineId).toBe(LINE.lineId);
    // REST: the key travels as a header, never in the URL
    expect(calls.length).toBeGreaterThan(0);
    for (const c of calls) {
      expect(c.url).not.toContain(SECRET);
      expect(c.headers["X-API-Key"]).toBe(SECRET);
    }
    // WebSocket: no key in the URL; the first frame carries it
    const ws = sockets[0];
    expect(ws).toBeDefined();
    expect(ws?.url).not.toContain(SECRET);
    ws?.onopen?.({});
    expect(JSON.parse(ws?.sent[0] ?? "{}")).toEqual({ type: "auth", apiKey: SECRET });
    // not remembered: neither storage holds it
    expect(dump(localStorage)).not.toContain(SECRET);
    expect(dump(sessionStorage)).not.toContain(SECRET);
    session.stop();
  });

  it("remember keeps the key in sessionStorage only, never localStorage", async () => {
    const { fetch } = fakeFetch([
      (c) => (pathOf(c) === "/auth/me" ? { body: { principal: "victor", role: "viewer", anonymous: false } } : undefined),
      (c) => (pathOf(c) === "/line" ? { body: LINE } : undefined),
    ]);
    const model = new DashboardModel();
    const session = new Session(model, { baseUrl: BASE_URL, fetch, storage: sessionStorage, socketFactory: (u) => new FakeSocket(u) });
    await session.applyKey(SECRET, true);
    expect(dump(sessionStorage)).toContain(SECRET);
    expect(dump(localStorage)).not.toContain(SECRET);
    expect(model.auth.get().role).toBe("viewer");
    session.stop();
  });

  it("a rejected key stops there: no stream, status rejected", async () => {
    const { fetch } = fakeFetch([() => ({ status: 401, body: { detail: "invalid or missing API key" } })]);
    const sockets: FakeSocket[] = [];
    const model = new DashboardModel();
    const session = new Session(model, {
      baseUrl: BASE_URL,
      fetch,
      storage: null,
      socketFactory: (u) => {
        const s = new FakeSocket(u);
        sockets.push(s);
        return s;
      },
    });
    await session.applyKey("wrong", false);
    expect(model.auth.get().status).toBe("rejected");
    expect(model.connection.get().status).toBe("unauthorized");
    expect(sockets).toHaveLength(0);
    await session.applyKey(null, false);
    expect(model.auth.get().status).toBe("needsKey");
    session.stop();
    model.dispose();
  });
});

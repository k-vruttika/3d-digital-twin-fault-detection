/**
 * WebSocket client for `/stream` (docs/api.md, "WebSocket stream").
 *
 * Protocol: the first frame is `auth` (key optional); after the server's `auth` reply the client
 * sends `subscribe`; the server answers `subscribed` and a `snapshot`, then streams messages.
 *
 * Reconnects with exponential backoff on any close except 4401 (bad key): also after 4408
 * (closed for being too slow) and after a missed heartbeat. Every reconnect re-authenticates,
 * re-subscribes with the current selection and delivers a fresh snapshot (`reconnect: true`).
 * The backoff resets once a snapshot has arrived.
 */
import { Backoff, DEFAULT_BACKOFF, type BackoffOptions } from "./backoff.ts";
import type { Channel, WsClientMessage, WsServerMessage, WsSnapshotData } from "./types.ts";

export const CLOSE_UNAUTHORIZED = 4401;
export const CLOSE_TOO_SLOW = 4408;
/** Client-side close code for a missed heartbeat (private range). */
export const CLOSE_HEARTBEAT = 4000;

export interface SocketLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: { code: number; reason: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
}

export type SocketFactory = (url: string) => SocketLike;

export type StreamStatus =
  | "idle"
  | "connecting"
  | "authenticating"
  | "subscribing"
  | "live"
  | "reconnecting"
  | "unauthorized"
  | "closed";

export interface SnapshotInfo {
  /** true for every snapshot after the first one (the stream was re-established). */
  reconnect: boolean;
}

/** Streamed messages (everything except the protocol's control frames). */
export type StreamMessage = Exclude<
  WsServerMessage,
  { type: "auth" | "subscribed" | "snapshot" | "pong" }
>;

interface Events {
  snapshot: (data: WsSnapshotData, info: SnapshotInfo) => void;
  message: (msg: StreamMessage) => void;
  status: (status: StreamStatus, detail?: string) => void;
}

export interface StreamClientOptions {
  /** e.g. ws://127.0.0.1:8000/stream */
  url: string;
  apiKey?: string | undefined;
  machines?: string[] | null;
  channels?: Channel[];
  socketFactory?: SocketFactory;
  backoff?: Partial<BackoffOptions>;
  /** Ping interval while live, in ms; 0 turns the heartbeat off. */
  heartbeatMs?: number;
  /** A missing pong after this long closes the socket and reconnects. */
  heartbeatTimeoutMs?: number;
}

const defaultSocketFactory: SocketFactory = (url) => new WebSocket(url) as unknown as SocketLike;

export class StreamClient {
  private socket: SocketLike | null = null;
  private statusValue: StreamStatus = "idle";
  private readonly backoff: Backoff;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private pongTimer: ReturnType<typeof setTimeout> | null = null;
  private snapshots = 0;
  private stopped = false;
  private machines: string[] | null;
  private channels: Channel[] | undefined;
  private readonly listeners: { [K in keyof Events]: Set<Events[K]> } = {
    snapshot: new Set(),
    message: new Set(),
    status: new Set(),
  };
  /** Connections opened so far (1 + reconnects). */
  connections = 0;

  constructor(private readonly opts: StreamClientOptions) {
    this.backoff = new Backoff({ ...DEFAULT_BACKOFF, ...opts.backoff });
    this.machines = opts.machines ?? null;
    this.channels = opts.channels;
  }

  get status(): StreamStatus {
    return this.statusValue;
  }

  on<K extends keyof Events>(event: K, cb: Events[K]): () => void {
    const set = this.listeners[event] as Set<Events[K]>;
    set.add(cb);
    return () => set.delete(cb);
  }

  connect(): void {
    this.stopped = false;
    this.open();
  }

  /** Change the selection; sent at once when live, and used for every reconnect. */
  subscribe(machines: string[] | null, channels?: Channel[]): void {
    this.machines = machines;
    this.channels = channels;
    if (this.statusValue === "live" || this.statusValue === "subscribing") {
      this.sendSubscribe();
    }
  }

  /** Stop for good: no reconnect, timers cleared, listeners kept (call off() yourself). */
  close(): void {
    this.stopped = true;
    this.clearTimers();
    const socket = this.socket;
    this.socket = null;
    if (socket !== null) {
      this.detach(socket);
      socket.close(1000, "client closed");
    }
    this.setStatus("closed");
  }

  /** Active timers (for tests: nothing may be left after close()). */
  get pendingTimers(): number {
    return [this.reconnectTimer, this.heartbeatTimer, this.pongTimer].filter((t) => t !== null)
      .length;
  }

  // -- connection ----------------------------------------------------------------------------
  private open(): void {
    this.clearTimers();
    const factory = this.opts.socketFactory ?? defaultSocketFactory;
    const socket = factory(this.opts.url);
    this.socket = socket;
    this.connections += 1;
    this.setStatus("connecting");
    socket.onopen = () => {
      if (socket !== this.socket) return;
      this.setStatus("authenticating");
      const auth: WsClientMessage =
        this.opts.apiKey !== undefined && this.opts.apiKey !== ""
          ? { type: "auth", apiKey: this.opts.apiKey }
          : { type: "auth" };
      this.send(auth);
    };
    socket.onmessage = (ev) => {
      if (socket !== this.socket) return;
      this.handle(ev.data);
    };
    socket.onclose = (ev) => {
      if (socket !== this.socket) return;
      this.lost(ev.code, ev.reason);
    };
    socket.onerror = () => {
      /* a close event follows; handled there */
    };
  }

  private handle(raw: unknown): void {
    if (typeof raw !== "string") return;
    let msg: WsServerMessage;
    try {
      msg = JSON.parse(raw) as WsServerMessage;
    } catch {
      return;
    }
    switch (msg.type) {
      case "auth":
        if (this.statusValue === "authenticating") {
          this.setStatus("subscribing");
          this.sendSubscribe();
        }
        return;
      case "subscribed":
        return;
      case "snapshot": {
        const reconnect = this.snapshots > 0;
        this.snapshots += 1;
        this.backoff.reset();
        this.setStatus("live");
        this.startHeartbeat();
        for (const cb of this.listeners.snapshot) cb(msg.data, { reconnect });
        return;
      }
      case "pong":
        if (this.pongTimer !== null) {
          clearTimeout(this.pongTimer);
          this.pongTimer = null;
        }
        return;
      default:
        if (msg.type === "error" && this.statusValue !== "live") {
          this.setStatus(this.statusValue, msg.data.reason);
        }
        for (const cb of this.listeners.message) cb(msg);
    }
  }

  private lost(code: number, reason: string): void {
    const socket = this.socket;
    this.socket = null;
    if (socket !== null) this.detach(socket);
    this.clearTimers();
    if (this.stopped) {
      this.setStatus("closed");
      return;
    }
    if (code === CLOSE_UNAUTHORIZED) {
      this.setStatus("unauthorized", reason || "invalid or missing API key");
      return;
    }
    const delay = this.backoff.next();
    this.setStatus("reconnecting", `closed (${String(code)}); retry in ${String(delay)} ms`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.stopped) this.open();
    }, delay);
  }

  private sendSubscribe(): void {
    const sub: WsClientMessage =
      this.channels === undefined
        ? { type: "subscribe", machines: this.machines }
        : { type: "subscribe", machines: this.machines, channels: this.channels };
    this.send(sub);
  }

  private send(msg: WsClientMessage): void {
    this.socket?.send(JSON.stringify(msg));
  }

  // -- heartbeat -----------------------------------------------------------------------------
  private startHeartbeat(): void {
    const every = this.opts.heartbeatMs ?? 0;
    if (every <= 0 || this.heartbeatTimer !== null) return;
    const timeout = this.opts.heartbeatTimeoutMs ?? every;
    this.heartbeatTimer = setInterval(() => {
      if (this.pongTimer !== null) return; // still waiting for the previous pong
      this.send({ type: "ping" });
      this.pongTimer = setTimeout(() => {
        this.pongTimer = null;
        const socket = this.socket;
        // A half-open socket may never deliver a close event: treat it as lost now.
        this.lost(CLOSE_HEARTBEAT, "heartbeat timeout");
        socket?.close(CLOSE_HEARTBEAT, "heartbeat timeout");
      }, timeout);
    }, every);
  }

  private clearTimers(): void {
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    if (this.heartbeatTimer !== null) clearInterval(this.heartbeatTimer);
    if (this.pongTimer !== null) clearTimeout(this.pongTimer);
    this.reconnectTimer = null;
    this.heartbeatTimer = null;
    this.pongTimer = null;
  }

  private detach(socket: SocketLike): void {
    socket.onopen = null;
    socket.onmessage = null;
    socket.onclose = null;
    socket.onerror = null;
  }

  private setStatus(status: StreamStatus, detail?: string): void {
    this.statusValue = status;
    for (const cb of this.listeners.status) cb(status, detail);
  }
}

/** ws:// or wss:// URL of /stream for an http(s) base URL. */
export function streamUrl(baseUrl: string): string {
  const url = new URL(baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/stream`;
  url.search = "";
  return url.toString();
}

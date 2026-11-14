/**
 * Session: the REST and WebSocket clients for the current API key, wired into the model.
 *
 * Applying a key checks it with `GET /auth/me` first (role shown in the header), then loads the
 * layout and (re)connects the stream with that key. While the backend is unreachable it retries
 * with a doubling delay. A rejected key stops there: nothing reconnects until a new key is
 * applied. Keys are kept in memory or sessionStorage only (auth.ts).
 */
import { bindStream } from "../../api/bind.ts";
import { ApiError, RestClient } from "../../api/rest.ts";
import type { LineResponse } from "../../api/types.ts";
import { StreamClient, streamUrl, type SocketFactory } from "../../api/ws.ts";
import { persistKey, type KeyStorage } from "./auth.ts";
import type { DashboardModel } from "./model.ts";
import { createStore, type Store } from "./store.ts";

export interface SessionOptions {
  baseUrl: string;
  fetch?: typeof fetch;
  socketFactory?: SocketFactory;
  storage?: KeyStorage | null;
  /** Retry delays while the backend is unreachable (ms). */
  retryMinMs?: number;
  retryMaxMs?: number;
  heartbeatMs?: number;
  heartbeatTimeoutMs?: number;
}

export class Session {
  /** The REST client for the current key (replaced when the key changes). */
  readonly rest: Store<RestClient>;
  private stream: StreamClient | null = null;
  private unbind: (() => void) | null = null;
  private offStatus: (() => void) | null = null;
  private generation = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;

  constructor(
    private readonly model: DashboardModel,
    private readonly opts: SessionOptions,
  ) {
    this.rest = createStore(this.makeRest(null));
  }

  get streamClient(): StreamClient | null {
    return this.stream;
  }

  /** Use `key` (null = none) from now on; `remember` keeps it in sessionStorage for this tab. */
  async applyKey(key: string | null, remember: boolean): Promise<void> {
    const gen = ++this.generation;
    this.clearRetry();
    this.closeStream();
    const k = key === "" ? null : key;
    persistKey(k, remember, this.opts.storage);
    const rest = this.makeRest(k);
    this.rest.set(rest);
    this.model.auth.set({ key: k, remember, status: "checking", role: null, principal: null, anonymous: false, error: null });
    this.model.resetConnection();
    await this.connect(gen, rest, k, remember, this.opts.retryMinMs ?? 500);
  }

  /** Re-subscribe (the server answers with a fresh snapshot); used when leaving replay. */
  resubscribe(): void {
    this.stream?.subscribe(null);
  }

  stop(): void {
    this.stopped = true;
    this.generation++;
    this.clearRetry();
    this.closeStream();
  }

  // ------------------------------------------------------------------ internals
  private async connect(gen: number, rest: RestClient, key: string | null, remember: boolean, delayMs: number): Promise<void> {
    if (this.stopped || gen !== this.generation) return;
    let line: LineResponse;
    try {
      const me = await rest.getAuthMe();
      if (gen !== this.generation) return;
      this.model.auth.set({
        key,
        remember,
        status: "ok",
        role: me.role,
        principal: me.principal,
        anonymous: me.anonymous,
        error: null,
      });
      line = await rest.getLine();
    } catch (error) {
      if (gen !== this.generation) return;
      if (error instanceof ApiError && error.status === 401) {
        this.model.auth.set({
          key,
          remember,
          status: key === null ? "needsKey" : "rejected",
          role: null,
          principal: null,
          anonymous: false,
          error: typeof error.detail === "string" ? error.detail : null,
        });
        this.model.setStreamStatus("unauthorized", key === null ? "this backend needs an API key" : "the API key was rejected");
        return;
      }
      this.model.auth.update((a) => ({ ...a, status: "error", error: String(error) }));
      this.model.setStreamStatus("reconnecting", `backend not reachable at ${this.opts.baseUrl}`);
      const next = Math.min(delayMs * 2, this.opts.retryMaxMs ?? 10_000);
      this.retryTimer = setTimeout(() => {
        void this.connect(gen, rest, key, remember, next);
      }, delayMs);
      return;
    }
    if (gen !== this.generation) return;
    this.model.loadLayout(line);
    const client = new StreamClient({
      url: streamUrl(this.opts.baseUrl),
      apiKey: key ?? undefined,
      heartbeatMs: this.opts.heartbeatMs ?? 10_000,
      heartbeatTimeoutMs: this.opts.heartbeatTimeoutMs ?? 5_000,
      ...(this.opts.socketFactory === undefined ? {} : { socketFactory: this.opts.socketFactory }),
    });
    this.stream = client;
    this.offStatus = client.on("status", (status, detail) => {
      this.model.setStreamStatus(status, detail);
    });
    this.unbind = bindStream(client, this.model, {
      fetchLine: () => rest.getLine(),
      initialLine: line,
      onError: (error) => {
        this.model.setStreamStatus("reconnecting", `update failed: ${String(error)}`);
      },
    });
    client.connect();
  }

  private makeRest(key: string | null): RestClient {
    return new RestClient({
      baseUrl: this.opts.baseUrl,
      apiKey: key ?? undefined,
      ...(this.opts.fetch === undefined ? {} : { fetch: this.opts.fetch }),
    });
  }

  private closeStream(): void {
    this.unbind?.();
    this.offStatus?.();
    this.stream?.close();
    this.unbind = null;
    this.offStatus = null;
    this.stream = null;
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }
}

/**
 * Typed REST client for the LineTwin backend (docs/api.md).
 *
 * Paths, query parameters, bodies and responses are typed through the generated `paths`
 * (schema.gen.ts). `ENDPOINTS` lists every operation the client uses; a test checks each one
 * against the committed OpenAPI schema.
 */
import type {
  AdapterKind,
  Alert,
  AlertsResponse,
  ApiPaths,
  AuthMeResponse,
  CommandAck,
  CommandLimitsResponse,
  CommandRequest,
  CommandsResponse,
  HealthResponse,
  HistoryResponse,
  LineResponse,
  MachinesResponse,
  MachineStateResponse,
  PredictionsResponse,
  PredictResponse,
  ReplayRangeResponse,
  ReplayResponse,
  ScenarioOptionsResponse,
  ScenarioPayload,
  ScenarioRecord,
  ScenariosResponse,
  SourcesResponse,
} from "./types.ts";

type Method = "get" | "post";
type PathOf<M extends Method> = {
  [P in keyof ApiPaths]: ApiPaths[P][M] extends undefined ? never : P;
}[keyof ApiPaths];
type Operation<P extends keyof ApiPaths, M extends Method> = NonNullable<ApiPaths[P][M]>;
type OkBody<O> = O extends { responses: { 200: { content: { "application/json": infer R } } } }
  ? R
  : never;
type Query<O> = O extends { parameters: { query?: infer Q } } ? NonNullable<Q> : never;

/** Every (method, path) the client calls; checked against openapi.json in the tests. */
export const ENDPOINTS = [
  ["get", "/line"],
  ["get", "/machines"],
  ["get", "/machines/{machine_id}/state"],
  ["get", "/history"],
  ["get", "/replay"],
  ["get", "/alerts"],
  ["post", "/alerts/{alert_id}/ack"],
  ["get", "/alerts/history"],
  ["post", "/commands"],
  ["get", "/health"],
  ["get", "/sources"],
  ["post", "/sources/switch"],
  ["get", "/auth/me"],
  ["get", "/commands"],
  ["get", "/commands/limits"],
  ["get", "/replay/range"],
  ["get", "/scenario"],
  ["post", "/scenario"],
  ["get", "/scenario/options"],
  ["get", "/predict"],
  ["get", "/predict/history"],
  ["post", "/whatif"],
] as const satisfies readonly (readonly [Method, keyof ApiPaths])[];

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
    readonly path: string,
  ) {
    super(`${path}: HTTP ${String(status)}${typeof detail === "string" ? ` - ${detail}` : ""}`);
    this.name = "ApiError";
  }
}

export interface RestClientOptions {
  /** e.g. http://127.0.0.1:8000 */
  baseUrl: string;
  /** Sent as X-API-Key; omit for anonymous read on a loopback server without keys. */
  apiKey?: string | undefined;
  fetch?: typeof fetch;
}

type QueryValue = string | number | boolean | readonly string[] | null | undefined;

export class RestClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchFn: typeof fetch;

  constructor(opts: RestClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.fetchFn = opts.fetch ?? globalThis.fetch.bind(globalThis);
  }

  getLine(): Promise<LineResponse> {
    return this.get("/line", {});
  }

  getMachines(): Promise<MachinesResponse> {
    return this.get("/machines", {});
  }

  getMachineState(machineId: string): Promise<MachineStateResponse> {
    return this.get("/machines/{machine_id}/state", { machine_id: machineId });
  }

  getHistory(query: Query<Operation<"/history", "get">>, signal?: AbortSignal): Promise<HistoryResponse> {
    return this.get("/history", {}, query, signal);
  }

  getReplay(t: string, signal?: AbortSignal): Promise<ReplayResponse> {
    return this.get("/replay", {}, { t }, signal);
  }

  getReplayRange(): Promise<ReplayRangeResponse> {
    return this.get("/replay/range", {});
  }

  getAuthMe(): Promise<AuthMeResponse> {
    return this.get("/auth/me", {});
  }

  getCommands(query: Query<Operation<"/commands", "get">> = {}): Promise<CommandsResponse> {
    return this.get("/commands", {}, query);
  }

  getCommandLimits(): Promise<CommandLimitsResponse> {
    return this.get("/commands/limits", {});
  }

  switchSource(kind: AdapterKind): Promise<SourcesResponse> {
    return this.post("/sources/switch", {}, { kind });
  }

  getScenarios(query: Query<Operation<"/scenario", "get">> = {}): Promise<ScenariosResponse> {
    return this.get("/scenario", {}, query);
  }

  postScenario(payload: ScenarioPayload): Promise<ScenarioRecord> {
    return this.post("/scenario", {}, payload);
  }

  getScenarioOptions(): Promise<ScenarioOptionsResponse> {
    return this.get("/scenario/options", {});
  }

  /** Live predictions: inference state, deployed bundles, per-machine gate (Phase 11). */
  getPredict(query: Query<Operation<"/predict", "get">> = {}): Promise<PredictResponse> {
    return this.get("/predict", {}, query);
  }

  getPredictHistory(query: Query<Operation<"/predict/history", "get">>): Promise<PredictionsResponse> {
    return this.get("/predict/history", {}, query);
  }

  /** Phase 11 stub: rejects with ApiError 501 until the what-if service exists. */
  async postWhatif(body: unknown = {}): Promise<unknown> {
    return this.request("POST", "/whatif", {}, undefined, body);
  }

  getAlerts(query: Query<Operation<"/alerts", "get">> = {}): Promise<AlertsResponse> {
    return this.get("/alerts", {}, query);
  }

  getAlertHistory(query: Query<Operation<"/alerts/history", "get">> = {}): Promise<AlertsResponse> {
    return this.get("/alerts/history", {}, query);
  }

  ackAlert(alertId: string): Promise<Alert> {
    return this.post("/alerts/{alert_id}/ack", { alert_id: alertId }, undefined);
  }

  postCommand(request: CommandRequest): Promise<CommandAck> {
    return this.post("/commands", {}, request);
  }

  getHealth(): Promise<HealthResponse> {
    return this.get("/health", {});
  }

  getSources(): Promise<SourcesResponse> {
    return this.get("/sources", {});
  }

  // -- plumbing ------------------------------------------------------------------------------
  private get<P extends PathOf<"get">>(
    path: P,
    params: Record<string, string>,
    query?: Record<string, QueryValue>,
    signal?: AbortSignal,
  ): Promise<OkBody<Operation<P, "get">>> {
    return this.request("GET", path, params, query, undefined, signal) as Promise<
      OkBody<Operation<P, "get">>
    >;
  }

  private post<P extends PathOf<"post">>(
    path: P,
    params: Record<string, string>,
    body: unknown,
  ): Promise<OkBody<Operation<P, "post">>> {
    return this.request("POST", path, params, undefined, body) as Promise<
      OkBody<Operation<P, "post">>
    >;
  }

  /** The URL for a path template, its path parameters and query (exported for tests). */
  url(path: string, params: Record<string, string>, query?: Record<string, QueryValue>): string {
    const filled = path.replace(/\{(\w+)\}/g, (_m, name: string) => {
      const value = params[name];
      if (value === undefined) {
        throw new Error(`missing path parameter ${name} for ${path}`);
      }
      return encodeURIComponent(value);
    });
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value === null || value === undefined) continue;
      if (Array.isArray(value)) {
        for (const item of value as readonly unknown[]) search.append(key, String(item));
      } else {
        search.append(key, String(value));
      }
    }
    const qs = search.toString();
    return `${this.baseUrl}${filled}${qs ? `?${qs}` : ""}`;
  }

  protected async request(
    method: "GET" | "POST",
    path: string,
    params: Record<string, string>,
    query: Record<string, QueryValue> | undefined,
    body: unknown,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.apiKey !== undefined && this.apiKey !== "") headers["X-API-Key"] = this.apiKey;
    const init: RequestInit = { method, headers };
    if (signal !== undefined) init.signal = signal;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const response = await this.fetchFn(this.url(path, params, query), init);
    const text = await response.text();
    let parsed: unknown = undefined;
    if (text !== "") {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = text;
      }
    }
    if (!response.ok) {
      const detail =
        typeof parsed === "object" && parsed !== null && "detail" in parsed
          ? parsed.detail
          : parsed;
      throw new ApiError(response.status, detail, path);
    }
    return parsed;
  }
}

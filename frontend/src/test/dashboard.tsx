/** Test harness for dashboard components: a real model, fake REST, fake charts and engine. */
import { render, type RenderResult } from "@testing-library/react";
import type { ReactNode } from "react";

import type {
  Alert,
  LineResponse,
  MachineState,
  Reading,
  WsSnapshotData,
} from "../api/types.ts";
import type { ChartFactory, ChartLike, LineConfig } from "../dashboard/charts/chartFactory.ts";
import type { DashboardServices } from "../dashboard/context.ts";
import { Dashboard } from "../dashboard/Dashboard.tsx";
import type { EngineFactory, EngineHandle } from "../dashboard/engineFactory.ts";
import { INITIAL_AUTH, type AuthState } from "../dashboard/state/auth.ts";
import { DashboardModel } from "../dashboard/state/model.ts";
import { ReplayController } from "../dashboard/state/replay.ts";
import { Session } from "../dashboard/state/session.ts";
import { LINE } from "./fakes.ts";

export interface FetchCall {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: unknown;
}

export type Route = (call: FetchCall) => { status?: number; body?: unknown } | undefined;

/** A fetch that answers from `routes` (first match wins) and records every call. */
export function fakeFetch(routes: Route[]): { fetch: typeof fetch; calls: FetchCall[] } {
  const calls: FetchCall[] = [];
  const fn = (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const call: FetchCall = {
      method: init?.method ?? "GET",
      url,
      headers,
      body: typeof init?.body === "string" ? (JSON.parse(init.body) as unknown) : undefined,
    };
    calls.push(call);
    for (const r of routes) {
      const res = r(call);
      if (res !== undefined) {
        const status = res.status ?? 200;
        return Promise.resolve(
          new Response(res.body === undefined ? "" : JSON.stringify(res.body), {
            status,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
    }
    return Promise.resolve(new Response(JSON.stringify({ detail: `no route for ${url}` }), { status: 404 }));
  };
  return { fetch: fn, calls };
}

/** Path of a recorded call without the base URL and query. */
export function pathOf(call: FetchCall): string {
  return new URL(call.url).pathname;
}

export interface FakeChart extends ChartLike {
  config: LineConfig;
  updates: number;
  destroyed: boolean;
}

export function fakeChartFactory(): { factory: ChartFactory; charts: FakeChart[] } {
  const charts: FakeChart[] = [];
  const factory: ChartFactory = (_canvas, config, signal) => {
    if (signal.aborted) return Promise.resolve(null);
    const chart: FakeChart = {
      config,
      data: config.data,
      options: config.options ?? {},
      updates: 0,
      destroyed: false,
      update() {
        chart.updates++;
      },
      destroy() {
        chart.destroyed = true;
      },
      valueAtPixel: (x) => x * 1000,
    };
    charts.push(chart);
    return Promise.resolve(chart);
  };
  return { factory, charts };
}

export class FakeEngine implements EngineHandle {
  mode: "live" | "replay" = "live";
  labelMode: "auto" | "all" = "auto";
  calls: string[] = [];
  snapshots: unknown[] = [];
  selected: string | null = null;
  connected = true;
  private selectCbs = new Set<(e: { machineId: string | null }) => void>();
  loadLayout(): void {
    this.calls.push("loadLayout");
  }
  applySnapshot(s: unknown): void {
    this.calls.push("applySnapshot");
    this.snapshots.push(s);
  }
  applyReading(): void {
    this.calls.push("applyReading");
  }
  applyStateChange(): void {
    this.calls.push("applyStateChange");
  }
  applyDataQuality(): void {
    this.calls.push("applyDataQuality");
  }
  applyAlert(): void {
    this.calls.push("applyAlert");
  }
  applyLine(): void {
    this.calls.push("applyLine");
  }
  setPrediction(): void {
    this.calls.push("setPrediction");
  }
  setConnected(c: boolean): void {
    this.connected = c;
  }
  setMode(m: "live" | "replay"): void {
    this.mode = m;
    this.calls.push(`setMode:${m}`);
  }
  select(id: string | null): void {
    this.selected = id;
  }
  setLabelMode(m: "auto" | "all"): void {
    this.labelMode = m;
  }
  setTheme(): void {
    this.calls.push("setTheme");
  }
  resize(): void {
    // nothing to resize
  }
  dispose(): void {
    this.calls.push("dispose");
  }
  on(_event: "select", cb: (e: { machineId: string | null }) => void): () => void {
    this.selectCbs.add(cb);
    return () => this.selectCbs.delete(cb);
  }
  /** Simulate a click on a machine in the 3D view. */
  click(id: string | null): void {
    for (const cb of this.selectCbs) cb({ machineId: id });
  }
}

export interface Harness {
  services: DashboardServices;
  model: DashboardModel;
  calls: FetchCall[];
  charts: FakeChart[];
  engine: FakeEngine;
  result: RenderResult;
  replay: ReplayController;
  resubscribed: { count: number };
}

export const OPERATOR: AuthState = { ...INITIAL_AUTH, key: "op", status: "ok", role: "operator", principal: "alice" };
export const VIEWER: AuthState = { ...INITIAL_AUTH, key: "view", status: "ok", role: "viewer", principal: "victor" };
export const ANONYMOUS: AuthState = { ...INITIAL_AUTH, status: "ok", role: "viewer", principal: "anonymous", anonymous: true };

export function reading(machineId: string, state: MachineState, seq: number, over: Partial<Reading> = {}): Reading {
  return {
    schemaVersion: 2,
    machineId,
    seq,
    ts: new Date(Date.parse("2026-01-01T00:00:00Z") + seq * 1000).toISOString(),
    state,
    speed: 1000,
    temp: 30,
    vib: 1,
    load: 50,
    current: 10,
    runTime: seq,
    quality: "OK",
    ...over,
  };
}

export function alert(over: Partial<Alert> & { alertId: string; machineId: string }): Alert {
  return {
    ts: "2026-01-01T00:00:05Z",
    severity: "CRITICAL",
    kind: "machine_fault",
    message: `${over.machineId} is in FAULT`,
    source: "engine",
    active: true,
    acknowledged: false,
    ackedBy: null,
    ackedAt: null,
    rev: 0,
    ...over,
  };
}

export function snapshot(states: Record<string, MachineState>, alerts: Alert[] = []): WsSnapshotData {
  return {
    asOf: "2026-01-01T00:00:05Z",
    machines: Object.fromEntries(
      Object.entries(states).map(([id, st]) => [
        id,
        { machineId: id, state: st, stateTs: "2026-01-01T00:00:05Z", reportedState: st, lastReading: reading(id, st, 5), lastStateChange: null },
      ]),
    ),
    line: { status: "RUNNING", rootCauses: {}, impacted: {} },
    dataQuality: {},
    alerts,
  };
}

export interface HarnessOptions {
  auth?: AuthState;
  routes?: Route[];
  line?: LineResponse;
  connected?: boolean;
  states?: Record<string, MachineState>;
  alerts?: Alert[];
  now?: () => number;
  /** Extra React to render instead of the full dashboard. */
  ui?: (services: DashboardServices) => ReactNode;
}

export const BASE_URL = "http://backend.test";

export function renderDashboard(opts: HarnessOptions = {}): Harness {
  const { fetch, calls } = fakeFetch([...(opts.routes ?? []), () => ({ body: {} })]);
  const model = new DashboardModel(undefined, opts.now);
  const session = new Session(model, { baseUrl: BASE_URL, fetch, storage: null });
  const { factory, charts } = fakeChartFactory();
  const engine = new FakeEngine();
  const engineFactory: EngineFactory = () => Promise.resolve(engine);
  const resubscribed = { count: 0 };
  const replay = new ReplayController({
    model,
    rest: () => session.rest.get(),
    resubscribe: () => {
      resubscribed.count++;
    },
    cfg: model.cfg,
  });
  const services: DashboardServices = { model, session, chartFactory: factory, engineFactory, replay };
  model.auth.set(opts.auth ?? OPERATOR);
  model.loadLayout(opts.line ?? LINE);
  model.applySnapshot(snapshot(opts.states ?? { "CNV-01": "RUNNING", "CNC-01": "RUNNING", "INSP-01": "RUNNING", "CNV-02": "RUNNING" }, opts.alerts ?? []));
  model.setStreamStatus(opts.connected === false ? "reconnecting" : "live");
  const result = render(opts.ui ? <>{opts.ui(services)}</> : <Dashboard services={services} />);
  return { services, model, calls, charts, engine, result, replay, resubscribed };
}

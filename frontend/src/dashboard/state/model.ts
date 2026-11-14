/**
 * DashboardModel: the one place stream and replay data land (ADR 0018).
 *
 * It implements the `EngineSink` that `bindStream` feeds, keeps per-machine views, alerts, the
 * line view and connection state in small stores for the React panels, and forwards everything
 * to the 3D engine once it has loaded (the engine is a separate chunk, so it may attach after
 * the first snapshot; it then gets the current state replayed into it). Readings also go to
 * plain listeners (live charts), so they never re-render React by themselves.
 */
import type { EngineSink } from "../../api/bind.ts";
import type {
  Alert,
  Command,
  CommandAck,
  DataQualityReport,
  LineResponse,
  LineViewModel,
  MachineState,
  MachineView,
  Prediction,
  Reading,
  ReplayResponse,
  RootCauseView,
  SourcesResponse,
  StateChange,
  WsSnapshotData,
} from "../../api/types.ts";
import type { StreamStatus } from "../../api/ws.ts";
import type { SnapshotLike } from "../../engine/TwinEngine.ts";
import type { LabelMode } from "../../engine/labelPolicy.ts";
import { DASHBOARD_CONFIG, type DashboardConfig } from "../config.ts";
import { applyAlertEvent } from "./alerts.ts";
import { INITIAL_AUTH, type AuthState } from "./auth.ts";
import { connectionView, type ConnectionView } from "./connection.ts";
import { createStore, type Store } from "./store.ts";

export interface MachineInfo {
  id: string;
  state: MachineState | null;
  stateTs: string | null;
  reason: string | null;
  reading: Reading | null;
  dataQuality: DataQualityReport | null;
}

/** What the model needs from the 3D engine (TwinEngine implements it). */
export interface EngineLike {
  readonly mode: "live" | "replay";
  loadLayout(line: LineResponse): void;
  applySnapshot(snapshot: SnapshotLike): void;
  applyReading(reading: Reading): void;
  applyStateChange(change: StateChange): void;
  applyDataQuality(report: DataQualityReport): void;
  applyAlert(alert: Alert): void;
  applyLine(view: LineViewModel): void;
  setPrediction(prediction: Prediction): void;
  setConnected(connected: boolean): void;
  setMode(mode: "live" | "replay"): void;
  select(machineId: string | null): void;
  setLabelMode(mode: LabelMode): void;
}

export type ReadingListener = (reading: Reading) => void;
export type CommandEventListener = (event: { command?: Command; ack?: CommandAck }) => void;

function emptyMachine(id: string): MachineInfo {
  return { id, state: null, stateTs: null, reason: null, reading: null, dataQuality: null };
}

export class DashboardModel implements EngineSink {
  readonly line: Store<LineResponse | null> = createStore<LineResponse | null>(null);
  readonly machines: Store<Record<string, MachineInfo>> = createStore<Record<string, MachineInfo>>({});
  readonly lineView: Store<LineViewModel | null> = createStore<LineViewModel | null>(null);
  readonly alerts: Store<ReadonlyMap<string, Alert>> = createStore<ReadonlyMap<string, Alert>>(new Map());
  readonly connection: Store<ConnectionView>;
  readonly auth: Store<AuthState> = createStore<AuthState>(INITIAL_AUTH);
  readonly selection: Store<string | null> = createStore<string | null>(null);
  readonly modeStore: Store<"live" | "replay"> = createStore<"live" | "replay">("live");
  readonly labelMode: Store<LabelMode> = createStore<LabelMode>("auto");
  /** The event time the screen shows: the newest reading (live) or the replay time. */
  readonly asOf: Store<string | null> = createStore<string | null>(null);
  readonly engineReady: Store<boolean> = createStore(false);
  /** GET /sources, polled by the dashboard (active adapter, capabilities, health, ingest). */
  readonly sources: Store<SourcesResponse | null> = createStore<SourcesResponse | null>(null);

  private engine: EngineLike | null = null;
  private readonly readingListeners = new Set<ReadingListener>();
  private readonly commandListeners = new Set<CommandEventListener>();
  private streamStatus: StreamStatus = "idle";
  private streamDetail: string | null = null;
  private everLive = false;
  private notLiveSince: number | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(
    readonly cfg: DashboardConfig = DASHBOARD_CONFIG,
    private readonly now: () => number = () => Date.now(),
  ) {
    this.connection = createStore(connectionView("idle", false, null, now(), cfg.linkLostAfterMs));
  }

  // ------------------------------------------------------------------ EngineSink
  get mode(): "live" | "replay" {
    return this.modeStore.get();
  }

  loadLayout(line: LineResponse): void {
    this.line.set(line);
    const prev = this.machines.get();
    const next: Record<string, MachineInfo> = {};
    for (const m of line.machines) next[m.id] = prev[m.id] ?? emptyMachine(m.id);
    this.machines.set(next);
    if (this.selection.get() !== null && next[this.selection.get() ?? ""] === undefined) this.select(null);
    this.engine?.loadLayout(line);
  }

  applySnapshot(snapshot: WsSnapshotData | SnapshotLike): void {
    this.applyState(snapshot);
    if ("asOf" in snapshot && typeof snapshot.asOf === "string") this.asOf.set(snapshot.asOf);
    this.engine?.applySnapshot(snapshot);
  }

  applyReading(reading: Reading): void {
    const cur = this.machines.get()[reading.machineId];
    if (cur === undefined) return;
    if (cur.reading !== null && Date.parse(reading.ts) < Date.parse(cur.reading.ts)) return;
    const next: MachineInfo = { ...cur, reading };
    if (reading.quality !== "LATE" && reading.state !== cur.state) {
      // The same fold as the backend's live state: a reading carries the machine's state.
      next.state = reading.state;
      next.stateTs = reading.ts;
      next.reason = null;
    }
    this.putMachine(next);
    const asOf = this.asOf.get();
    if (asOf === null || Date.parse(reading.ts) > Date.parse(asOf)) this.asOf.set(reading.ts);
    this.engine?.applyReading(reading);
    for (const l of [...this.readingListeners]) l(reading);
  }

  applyStateChange(change: StateChange): void {
    const cur = this.machines.get()[change.machineId];
    if (cur === undefined) return;
    this.putMachine({ ...cur, state: change.state, stateTs: change.ts, reason: change.reason ?? null });
    this.engine?.applyStateChange(change);
  }

  applyDataQuality(report: DataQualityReport): void {
    const cur = this.machines.get()[report.machineId];
    if (cur === undefined) return;
    this.putMachine({ ...cur, dataQuality: report });
    this.engine?.applyDataQuality(report);
  }

  applyAlert(alert: Alert): void {
    this.alerts.set(applyAlertEvent(this.alerts.get(), alert));
    this.engine?.applyAlert(alert);
  }

  applyLine(view: LineViewModel): void {
    this.lineView.set(view);
    this.engine?.applyLine(view);
  }

  setPrediction(prediction: Prediction): void {
    this.engine?.setPrediction(prediction);
  }

  setConnected(connected: boolean): void {
    this.engine?.setConnected(connected);
  }

  applyCommand(command: Command): void {
    for (const l of [...this.commandListeners]) l({ command });
  }

  applyAck(ack: CommandAck): void {
    for (const l of [...this.commandListeners]) l({ ack });
  }

  // ------------------------------------------------------------------ replay
  setMode(mode: "live" | "replay"): void {
    this.modeStore.set(mode);
    this.engine?.setMode(mode);
  }

  /** Show a /replay result (engine and panels) as the state at its asOf. */
  applyReplay(r: ReplayResponse, t: string): void {
    const snapshot: SnapshotLike = {
      machines: r.machines ?? {},
      line: r.line,
      alerts: r.alerts ?? [],
      predictions: r.predictions ?? {},
      // data quality uses a live wall-clock window and is not replayable: none in replay
      dataQuality: {},
    };
    this.applyState(snapshot);
    this.asOf.set(t);
    this.engine?.applySnapshot(snapshot);
  }

  // ------------------------------------------------------------------ engine
  attachEngine(engine: EngineLike): () => void {
    this.engine = engine;
    const line = this.line.get();
    if (line !== null) {
      engine.loadLayout(line);
      engine.applySnapshot(this.toSnapshot());
    }
    engine.setMode(this.mode);
    engine.setConnected(this.connection.get().status === "connected");
    engine.setLabelMode(this.labelMode.get());
    engine.select(this.selection.get());
    this.engineReady.set(true);
    return () => {
      if (this.engine === engine) {
        this.engine = null;
        this.engineReady.set(false);
      }
    };
  }

  /** The panels' current state as a snapshot (to bring a newly attached engine up to date). */
  toSnapshot(): SnapshotLike {
    const machines: Record<string, MachineView> = {};
    for (const m of Object.values(this.machines.get())) {
      machines[m.id] = {
        machineId: m.id,
        state: m.state,
        stateTs: m.stateTs,
        reportedState: m.reading?.state ?? null,
        lastReading: m.reading,
        lastStateChange:
          m.state !== null && m.reason !== null && m.stateTs !== null
            ? { machineId: m.id, state: m.state, previousState: null, ts: m.stateTs, reason: m.reason }
            : null,
      };
    }
    const dataQuality: Record<string, DataQualityReport> = {};
    for (const m of Object.values(this.machines.get())) if (m.dataQuality !== null) dataQuality[m.id] = m.dataQuality;
    return { machines, line: this.lineView.get(), dataQuality, alerts: [...this.alerts.get().values()] };
  }

  select(machineId: string | null): void {
    const id = machineId !== null && this.machines.get()[machineId] !== undefined ? machineId : null;
    this.selection.set(id);
    this.engine?.select(id);
  }

  setLabelMode(mode: LabelMode): void {
    this.labelMode.set(mode);
    this.engine?.setLabelMode(mode);
  }

  // ------------------------------------------------------------------ listeners
  onReading(listener: ReadingListener): () => void {
    this.readingListeners.add(listener);
    return () => this.readingListeners.delete(listener);
  }

  onCommandEvent(listener: CommandEventListener): () => void {
    this.commandListeners.add(listener);
    return () => this.commandListeners.delete(listener);
  }

  // ------------------------------------------------------------------ connection
  setStreamStatus(status: StreamStatus, detail?: string): void {
    this.streamStatus = status;
    this.streamDetail = detail ?? null;
    if (status === "live") {
      this.everLive = true;
      this.notLiveSince = null;
    } else {
      this.notLiveSince ??= this.now();
    }
    this.refreshConnection();
    if (status === "live" || status === "unauthorized" || status === "closed") this.stopTimer();
    else this.startTimer();
  }

  /** A new key / client: the next connection counts as a first connect again. */
  resetConnection(): void {
    this.everLive = false;
    this.notLiveSince = null;
    this.setStreamStatus("connecting");
  }

  refreshConnection(): void {
    const view = connectionView(
      this.streamStatus,
      this.everLive,
      this.notLiveSince,
      this.now(),
      this.cfg.linkLostAfterMs,
      this.streamDetail,
    );
    const cur = this.connection.get();
    if (cur.status !== view.status || cur.detail !== view.detail) this.connection.set(view);
  }

  dispose(): void {
    this.stopTimer();
    this.readingListeners.clear();
    this.commandListeners.clear();
    this.engine = null;
  }

  // ------------------------------------------------------------------ internals
  private startTimer(): void {
    if (this.timer !== null) return;
    this.timer = setInterval(() => {
      this.refreshConnection();
    }, this.cfg.connectionTickMs);
  }

  private stopTimer(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }

  private putMachine(m: MachineInfo): void {
    this.machines.set({ ...this.machines.get(), [m.id]: m });
  }

  private applyState(snapshot: SnapshotLike): void {
    const next = { ...this.machines.get() };
    for (const [id, cur] of Object.entries(next)) {
      const view = snapshot.machines?.[id];
      const hasDq = snapshot.dataQuality !== undefined;
      const dq = snapshot.dataQuality?.[id] ?? null;
      if (view === undefined && !hasDq) continue;
      next[id] = {
        ...cur,
        ...(view === undefined
          ? {}
          : {
              state: view.state ?? null,
              stateTs: view.stateTs ?? null,
              reason: view.lastStateChange?.reason ?? null,
              reading: view.lastReading ?? null,
            }),
        ...(hasDq ? { dataQuality: dq } : {}),
      };
    }
    this.machines.set(next);
    if (snapshot.line !== undefined) this.lineView.set(snapshot.line);
    if (snapshot.alerts !== undefined) {
      let alerts: Map<string, Alert> = new Map();
      for (const a of snapshot.alerts) alerts = applyAlertEvent(alerts, a);
      this.alerts.set(alerts);
    }
  }
}

/** Root cause of a machine from the line view, if known. */
export function rootCauseOf(view: LineViewModel | null, id: string): RootCauseView | null {
  return view?.rootCauses[id] ?? null;
}

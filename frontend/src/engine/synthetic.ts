/**
 * Synthetic lines and readings for the dev page's performance and screenshot modes (no
 * backend). Deterministic: the same seed gives the same data. This is test data for the
 * renderer only; it never reaches the backend or any experiment.
 */
import { mulberry32 } from "../api/backoff.ts";
import type {
  LineMachine,
  LineResponse,
  LineViewModel,
  MachineKind,
  MachineState,
  Reading,
} from "../api/types.ts";
import { DEFAULT_CONFIG } from "./config.ts";
import { footprintOf } from "./layout.ts";
import { lineView } from "./rootCause.ts";

const SENSORS = ["speed", "temp", "vib", "load", "current"];
const CATALOG: LineResponse["sensors"] = {
  speed: { unit: "RPM", min: 0, max: 15000 },
  temp: { unit: "degC", min: -20, max: 150 },
  vib: { unit: "mm/s", min: 0, max: 50 },
  load: { unit: "%", min: 0, max: 130 },
  current: { unit: "A", min: 0, max: 100 },
};
const KIND_PARTS: Record<MachineKind, { parts: string[]; sensorParts: Record<string, string> }> = {
  CONVEYOR: {
    parts: ["belt", "motor", "bearing"],
    sensorParts: { speed: "belt", load: "belt", temp: "motor", current: "motor", vib: "bearing" },
  },
  CNC: {
    parts: ["spindle", "spindle_motor", "spindle_bearing"],
    sensorParts: {
      speed: "spindle",
      load: "spindle",
      temp: "spindle_motor",
      current: "spindle_motor",
      vib: "spindle_bearing",
    },
  },
  INSPECTION: {
    parts: ["indexer", "motor", "bearing"],
    sensorParts: { speed: "indexer", load: "indexer", temp: "motor", current: "motor", vib: "bearing" },
  },
};
const KIND_CYCLE: MachineKind[] = ["CONVEYOR", "CNC", "INSPECTION"];

export function syntheticMachine(id: string, kind: MachineKind, name = ""): LineMachine {
  const { parts, sensorParts } = KIND_PARTS[kind];
  return { id, kind, name: name || `${kind.toLowerCase()} ${id}`, sensors: [...SENSORS], parts: [...parts], sensorParts: { ...sensorParts }, layout: null };
}

/** A chain line from machines in flow order, with a buffer between each pair. */
export function chainLine(lineId: string, machines: LineMachine[], capacity = 8): LineResponse {
  const ids = machines.map((m) => m.id);
  const upstream: Record<string, string | null> = {};
  const downstream: Record<string, string | null> = {};
  ids.forEach((id, i) => {
    upstream[id] = ids[i - 1] ?? null;
    downstream[id] = ids[i + 1] ?? null;
  });
  return {
    lineId,
    sensors: CATALOG,
    machines,
    buffers: ids.slice(1).map((to, i) => ({ id: `B${String(i + 1)}`, from: ids[i] ?? "", to, capacity })),
    topology: { order: ids, upstream, downstream },
  };
}

/** n machines cycling CONVEYOR, CNC, INSPECTION (the performance check uses n = 20). */
export function generateLine(n: number): LineResponse {
  const machines = Array.from({ length: n }, (_, i) => {
    const kind = KIND_CYCLE[i % KIND_CYCLE.length] ?? "CONVEYOR";
    return syntheticMachine(`M-${String(i + 1).padStart(2, "0")}`, kind);
  });
  return chainLine(`synthetic_${String(n)}`, machines);
}

/**
 * The states scene (screenshots of every visual): independent short segments, one per row, so
 * every combination is one the simulator can produce (twin_sim/line.py):
 *
 * 1. RUNNING -> WARNING -> RUNNING: a WARNING machine keeps running and passing parts, so its
 *    neighbours run too (it can never make them STARVED / BLOCKED);
 * 2. BLOCKED -> FAULT -> STARVED: a faulted machine stops, its feeder fills up, its receiver
 *    runs dry;
 * 3. BLOCKED -> MAINTENANCE -> STARVED: the same for a machine in maintenance;
 * 4. STALE and OFFLINE alone (link states come from the backend, not the machine).
 *
 * Root causes are derived with `lineView` (the backend's rule), never written by hand.
 */
interface SceneMachine {
  id: string;
  kind: MachineKind;
  state: MachineState;
}

/** Segments in rows; `linked` segments have a buffer between neighbours. */
export const STATE_SEGMENTS: { linked: boolean; machines: SceneMachine[] }[] = [
  { linked: true, machines: [
    { id: "S-RUN", kind: "CONVEYOR", state: "RUNNING" },
    { id: "S-WARN", kind: "CNC", state: "WARNING" },
    { id: "S-RUN2", kind: "INSPECTION", state: "RUNNING" },
  ] },
  { linked: true, machines: [
    { id: "S-BLOCK", kind: "CONVEYOR", state: "BLOCKED" },
    { id: "S-FAULT", kind: "CNC", state: "FAULT" },
    { id: "S-STARVE", kind: "INSPECTION", state: "STARVED" },
  ] },
  { linked: true, machines: [
    { id: "S-BLOCK2", kind: "CONVEYOR", state: "BLOCKED" },
    { id: "S-MAINT", kind: "INSPECTION", state: "MAINTENANCE" },
    { id: "S-STARVE2", kind: "CONVEYOR", state: "STARVED" },
  ] },
  { linked: false, machines: [
    { id: "S-STALE", kind: "CNC", state: "STALE" },
    { id: "S-OFF", kind: "CONVEYOR", state: "OFFLINE" },
  ] },
];

export const STATE_SCENE: SceneMachine[] = STATE_SEGMENTS.flatMap((s) => s.machines);

/** The states scene as a line: segments are not linked; each sits in its own row. */
export function stateSceneLine(): LineResponse {
  const { gap, rowSpacing } = DEFAULT_CONFIG.layout;
  const machines: LineMachine[] = [];
  const buffers: LineResponse["buffers"] = [];
  const upstream: Record<string, string | null> = {};
  const downstream: Record<string, string | null> = {};
  STATE_SEGMENTS.forEach(({ linked, machines: segment }, row) => {
    let x = 0;
    let prev: string | null = null;
    for (const s of segment) {
      const m = syntheticMachine(s.id, s.kind);
      const len = footprintOf(s.kind, DEFAULT_CONFIG.layout).length;
      x += len / 2;
      m.layout = { position: [x, 0, row * rowSpacing], rotationDeg: 0 };
      x += len / 2 + gap;
      machines.push(m);
      upstream[s.id] = linked ? prev : null;
      downstream[s.id] = null;
      if (linked && prev !== null) {
        downstream[prev] = s.id;
        buffers.push({ id: `B-${s.id}`, from: prev, to: s.id, capacity: 8 });
      }
      prev = s.id;
    }
  });
  return {
    lineId: "synthetic_states",
    sensors: CATALOG,
    machines,
    buffers,
    topology: { order: machines.map((m) => m.id), upstream, downstream },
  };
}

/** Root causes of the states scene, derived with the backend's rule (rootCause.ts). */
export function stateSceneLineView(): LineViewModel {
  const states = Object.fromEntries(STATE_SCENE.map((s) => [s.id, s.state]));
  return lineView(states, stateSceneLine().topology);
}

export const SYNTHETIC_EPOCH_MS = Date.parse("2026-01-01T00:00:00Z");

/** A plausible reading near the kind's nominal speed (seeded noise; one sensor sometimes missing). */
export class SyntheticReadings {
  private readonly random: () => number;
  private readonly seqs = new Map<string, number>();

  constructor(seed = 7) {
    this.random = mulberry32(seed);
  }

  next(machine: LineMachine, state: MachineState, tS: number): Reading {
    const r = this.random;
    const seq = (this.seqs.get(machine.id) ?? 0) + 1;
    this.seqs.set(machine.id, seq);
    const ref = (DEFAULT_CONFIG.referenceRpm as Record<string, number | undefined>)[machine.kind] ?? DEFAULT_CONFIG.defaultReferenceRpm;
    const moving = state === "RUNNING" || state === "WARNING";
    const missingVib = r() < 0.05;
    return {
      schemaVersion: 2,
      machineId: machine.id,
      seq,
      ts: new Date(SYNTHETIC_EPOCH_MS + tS * 1000).toISOString(),
      state,
      speed: moving ? ref * (0.95 + 0.1 * r()) : 0,
      temp: 32 + 6 * r() + (state === "WARNING" ? 25 : 0),
      vib: missingVib ? null : 1.2 + 0.6 * r() + (state === "WARNING" ? 4 : 0),
      load: moving ? 55 + 10 * r() : 0,
      current: moving ? 9 + 2 * r() : 0.4,
      runTime: tS,
      quality: missingVib ? "SUSPECT" : "OK",
    };
  }
}

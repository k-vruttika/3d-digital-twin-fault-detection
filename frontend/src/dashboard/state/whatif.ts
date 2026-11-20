/**
 * What-if (Phase 11 Part C, ADR 0031): pure helpers, unit tested.
 *
 * The backend forks a copy of the simulation, applies ONE change at "now" and answers two
 * projections: `baseline` (nothing changed) and `scenario` (the change). Both are simulator
 * states and sensor values, and, when live inference is on, the deployed model's calibrated
 * probabilities and the evaluated alarm over them. Ghost states are the projected simulator
 * states at +10 s / +30 s / the end of the run, drawn as "ghosts" next to the real state.
 * Nothing here is sent to a machine; "apply as command" goes through the normal command path.
 */
import type { CommandType, MachineProjection, MachineState, ProjectionPoint, WhatIfPayload, WhatIfResponse } from "../../api/types.ts";
import type { ChartPoint } from "../charts/series.ts";

export interface ParamSpec {
  key: "speed" | "load";
  label: string;
  unit: string;
}

export interface ChangeSpec {
  type: CommandType;
  label: string;
  param: ParamSpec | null;
}

/** The changes that can be projected (the same set as commands). */
export const CHANGES: readonly ChangeSpec[] = [
  { type: "STOP", label: "Stop", param: null },
  { type: "START", label: "Start", param: null },
  { type: "SET_SPEED", label: "Set speed", param: { key: "speed", label: "Speed", unit: "RPM" } },
  { type: "SET_LOAD", label: "Set load", param: { key: "load", label: "Load", unit: "%" } },
  { type: "SCHEDULE_MAINTENANCE", label: "Maintenance", param: null },
  { type: "RESET", label: "Reset", param: null },
];

export function changeSpec(type: CommandType): ChangeSpec {
  return CHANGES.find((c) => c.type === type) ?? { type, label: type, param: null };
}

export type PayloadResult = { ok: true; payload: WhatIfPayload } | { ok: false; error: string };

/** The request body from the form; the backend still applies the safety rules and limits. */
export function buildPayload(machineId: string | null, type: CommandType, paramText: string, horizonS: number): PayloadResult {
  if (machineId === null) return { ok: false, error: "Select a machine." };
  const spec = changeSpec(type);
  const params: Record<string, number> = {};
  if (spec.param !== null) {
    const raw = paramText.trim();
    const v = Number(raw);
    if (raw === "" || !Number.isFinite(v)) return { ok: false, error: `${spec.param.label} needs a number (${spec.param.unit}).` };
    params[spec.param.key] = v;
  }
  return { ok: true, payload: { machineId, type, params, horizonS } };
}

/** What a projected value is: a sensor of the reading or a model output. */
export type SeriesField = "vib" | "temp" | "speed" | "load" | "current" | "pFault" | "pWarning";

export const SERIES_FIELDS: readonly { field: SeriesField; label: string; unit: string; model: boolean }[] = [
  { field: "vib", label: "Vibration", unit: "mm/s", model: false },
  { field: "temp", label: "Temperature", unit: "°C", model: false },
  { field: "speed", label: "Speed", unit: "RPM", model: false },
  { field: "load", label: "Load", unit: "%", model: false },
  { field: "current", label: "Current", unit: "A", model: false },
  { field: "pFault", label: "P(FAULT)", unit: "calibrated, +30 s", model: true },
  { field: "pWarning", label: "P(WARNING)", unit: "calibrated, +30 s", model: true },
];

function valueOf(p: ProjectionPoint, field: SeriesField): number | null {
  const v = p[field];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function pointsOf(pts: readonly ProjectionPoint[], field: SeriesField): ChartPoint[] {
  return pts.map((p) => ({ x: p.tS, y: valueOf(p, field) }));
}

export interface ProjectionSeries {
  /** Solid: what happened before the fork point (t <= 0). */
  history: ChartPoint[];
  /** Dashed: nothing changed. Starts at the last history point so the lines join. */
  baseline: ChartPoint[];
  /** Dashed: the change applied now. */
  scenario: ChartPoint[];
}

export function projectionSeries(r: WhatIfResponse, machineId: string, field: SeriesField): ProjectionSeries {
  const hist = r.history[machineId] ?? [];
  const history = pointsOf(hist, field);
  const joint = history.length > 0 ? [history[history.length - 1] as ChartPoint] : [];
  const future = (m: MachineProjection | undefined): ChartPoint[] => [...joint, ...pointsOf(m?.points ?? [], field)];
  return {
    history,
    baseline: future(r.baseline.machines[machineId]),
    scenario: future(r.scenario.machines[machineId]),
  };
}

/** Does any point carry a value for this field (model fields are absent without a model)? */
export function hasData(s: ProjectionSeries): boolean {
  return [s.history, s.baseline, s.scenario].some((pts) => pts.some((p) => p.y !== null));
}

export interface GhostCell {
  horizon: string; // "10", "30", "end"
  baseline: MachineState | null;
  scenario: MachineState | null;
  differs: boolean;
}

export interface GhostRow {
  machineId: string;
  now: MachineState | null;
  cells: GhostCell[];
  /** Any projected state differs between baseline and scenario. */
  changed: boolean;
}

function ghostAt(m: MachineProjection | undefined, horizon: string): MachineState | null {
  const ghost: Record<string, MachineState | undefined> | undefined = m?.ghost;
  return ghost?.[horizon] ?? null;
}

/** Real state at the fork point, then the projected ("ghost") states of both variants. */
export function ghostRows(r: WhatIfResponse, order: readonly string[]): GhostRow[] {
  const horizons = [...r.ghostHorizonsS.map(String), "end"];
  return order
    .filter((id) => r.baseline.machines[id] !== undefined)
    .map((id) => {
      const hist = r.history[id] ?? [];
      const now = hist.length > 0 ? (hist[hist.length - 1] as ProjectionPoint).state : null;
      const cells = horizons.map((h) => {
        const b = ghostAt(r.baseline.machines[id], h);
        const s = ghostAt(r.scenario.machines[id], h);
        return { horizon: h, baseline: b, scenario: s, differs: b !== s };
      });
      return { machineId: id, now, cells, changed: cells.some((c) => c.differs) };
    });
}

export function horizonLabel(h: string, endS: number): string {
  return h === "end" ? `+${String(endS)} s (end)` : `+${h} s`;
}

/** Plain-language lines about one machine's two projections (no numbers the response lacks). */
export function outcomeLines(r: WhatIfResponse, machineId: string): string[] {
  const base = r.baseline.machines[machineId];
  const scen = r.scenario.machines[machineId];
  if (base === undefined || scen === undefined) return [];
  const lines: string[] = [];
  const when = (s: number | null | undefined): string => (s === null || s === undefined ? "not within the run" : `at +${String(s)} s`);
  lines.push(`Simulator WARNING: without the change ${when(base.firstWarningS)}, with it ${when(scen.firstWarningS)}.`);
  lines.push(`Simulator FAULT: without the change ${when(base.firstFaultS)}, with it ${when(scen.firstFaultS)}.`);
  if (r.model !== null && r.model !== undefined) {
    const a = (s: number | null | undefined): string => (s === null || s === undefined ? "no alarm" : `alarm at +${String(s)} s`);
    lines.push(`Model (${String(r.primaryHorizonS ?? "")} s horizon): without the change ${a(base.firstAlarmS)}, with it ${a(scen.firstAlarmS)}.`);
  }
  return lines;
}

/** Text for the command the projection can be applied as. */
export function commandText(r: Pick<WhatIfResponse, "type" | "params" | "machineId">): string {
  const p = Object.entries(r.params).map(([k, v]) => `${k} ${String(v)}`);
  return `${r.type}${p.length > 0 ? ` (${p.join(", ")})` : ""} on ${r.machineId}`;
}

/** The command body for "apply as command": numeric params only (the command contract). */
export function applyBody(r: WhatIfResponse): { machineId: string; type: CommandType; params: Record<string, number> } {
  const params: Record<string, number> = {};
  for (const [k, v] of Object.entries(r.params)) if (typeof v === "number") params[k] = v;
  return { machineId: r.machineId, type: r.type, params };
}

/**
 * State -> visual mapping (pure functions, unit tested; ADR 0016).
 *
 * | State        | Motion   | Look                                                     |
 * |--------------|----------|----------------------------------------------------------|
 * | RUNNING      | animated | normal                                                   |
 * | IDLE         | stopped  | normal                                                   |
 * | WARNING      | animated | orange outline on the at-risk part(s)                    |
 * | FAULT        | stopped  | at-risk part(s) flash red                                |
 * | STARVED      | stopped  | yellow beacon                                            |
 * | BLOCKED      | stopped  | yellow beacon                                            |
 * | MAINTENANCE  | stopped  | blue beacon, blue tint                                   |
 * | STALE        | stopped  | semi-transparent                                         |
 * | OFFLINE      | stopped  | greyed out                                               |
 * | link lost    | stopped  | semi-transparent (the WebSocket is down)                 |
 *
 * An active `unusual_behaviour` alert (autoencoder, cause unknown) outlines the whole machine in
 * purple when no state outline is shown: it never names a part (ADR 0026 / 0030).
 *
 * Independent of the state: data quality DEGRADED shows a subtle halo, and active alerts a
 * floating marker in the colour of the highest severity. Every state also gets a text label
 * (colour is never the only cue), e.g. "BLOCKED by CNC-01".
 *
 * The at-risk part is the highest-weighted attribution of the newest prediction that names one
 * of the machine's parts; without one, every part of the machine.
 */
import type { AlertSeverity, MachineState } from "../api/types.ts";
import type { VisualConfig } from "./config.ts";

export type Tone =
  | "neutral"
  | "running"
  | "warning"
  | "fault"
  | "waiting"
  | "maintenance"
  | "stale"
  | "offline";

export type Beacon = "none" | "waiting" | "maintenance";

export interface VisualInput {
  state: MachineState | null;
  /** Line root cause of a STARVED / BLOCKED machine (WS channel `line`). */
  rootCause?: { cause?: string | null; unknown?: boolean } | null;
  /** The newest StateChange reason (fallback when no root cause is known yet). */
  reason?: string | null;
  dataQuality?: "OK" | "DEGRADED" | null;
  alerts?: readonly { severity: AlertSeverity; active?: boolean; kind?: string }[];
  attributions?: readonly { part: string; weight: number }[] | null;
  parts: readonly string[];
  /** false while the stream is down. */
  connected?: boolean;
  /** Display hold: keep the unusual-behaviour outline briefly after its alert cleared. */
  unusualHeld?: boolean;
}

export interface VisualSpec {
  tone: Tone;
  animate: boolean;
  /** 1 = as built, below 1 = see-through. */
  opacity: number;
  /** 0 = full colour, 1 = grey. */
  grey: number;
  /** Colour blended over the machine, and by how much. */
  tint: number | null;
  tintAmount: number;
  beacon: Beacon;
  outlineParts: string[];
  outlineColor: number | null;
  flashParts: string[];
  flashColor: number | null;
  dqBadge: boolean;
  alertMarker: AlertSeverity | null;
  /** An active unusual-behaviour alarm (autoencoder): whole machine, cause unknown. */
  unusual: boolean;
  label: { text: string; tone: Tone } | null;
}

/** Alert kind of the autoencoder alarm (engine.yaml). */
export const UNUSUAL_KIND = "unusual_behaviour";

const SEVERITY_RANK: Record<AlertSeverity, number> = { INFO: 0, WARNING: 1, CRITICAL: 2 };

/** The highest severity among active alerts, or null. */
export function highestActiveSeverity(
  alerts: readonly { severity: AlertSeverity; active?: boolean }[],
): AlertSeverity | null {
  let best: AlertSeverity | null = null;
  for (const a of alerts) {
    if (a.active === false) continue;
    if (best === null || SEVERITY_RANK[a.severity] > SEVERITY_RANK[best]) best = a.severity;
  }
  return best;
}

/** At-risk parts: the top attribution naming one of the machine's parts, else every part. */
export function atRiskParts(
  attributions: readonly { part: string; weight: number }[] | null | undefined,
  parts: readonly string[],
): string[] {
  const known = new Set(parts);
  let best: { part: string; weight: number } | null = null;
  for (const a of attributions ?? []) {
    if (!known.has(a.part)) continue;
    if (best === null || Math.abs(a.weight) > Math.abs(best.weight)) best = a;
  }
  return best === null ? [...parts] : [best.part];
}

/** The text shown above the machine. */
export function stateLabel(input: VisualInput): string {
  if (input.connected === false && input.state !== "OFFLINE") return "LINK LOST";
  const state = input.state;
  if (state === null) return "NO DATA";
  if (state === "STARVED" || state === "BLOCKED") {
    const cause = input.rootCause?.cause;
    if (cause !== null && cause !== undefined && cause !== "") {
      return `${state} by ${cause}${input.rootCause?.unknown === true ? " (unknown)" : ""}`;
    }
    const reason = input.reason ?? "";
    if (reason.startsWith(`${state} by `)) return reason;
    return state;
  }
  return state;
}

export function machineVisual(input: VisualInput, cfg: VisualConfig): VisualSpec {
  const c = cfg.colors;
  const spec: VisualSpec = {
    tone: "neutral",
    animate: false,
    opacity: 1,
    grey: 0,
    tint: null,
    tintAmount: 0,
    beacon: "none",
    outlineParts: [],
    outlineColor: null,
    flashParts: [],
    flashColor: null,
    dqBadge: input.dataQuality === "DEGRADED",
    alertMarker: highestActiveSeverity(input.alerts ?? []),
    label: null,
    unusual:
      input.unusualHeld === true || (input.alerts ?? []).some((a) => a.kind === UNUSUAL_KIND && a.active !== false),
  };
  const linkLost = input.connected === false;

  if (input.state === "OFFLINE") {
    spec.tone = "offline";
    spec.grey = cfg.offlineGrey;
  } else if (linkLost || input.state === "STALE") {
    spec.tone = "stale";
    spec.opacity = cfg.staleOpacity;
  } else {
    switch (input.state) {
      case "RUNNING":
        spec.tone = "running";
        spec.animate = true;
        break;
      case "IDLE":
        spec.tone = "neutral";
        break;
      case "WARNING":
        spec.tone = "warning";
        spec.animate = true;
        spec.outlineParts = atRiskParts(input.attributions, input.parts);
        spec.outlineColor = c.warning;
        break;
      case "FAULT":
        spec.tone = "fault";
        spec.flashParts = atRiskParts(input.attributions, input.parts);
        spec.flashColor = c.fault;
        break;
      case "STARVED":
      case "BLOCKED":
        spec.tone = "waiting";
        spec.beacon = "waiting";
        break;
      case "MAINTENANCE":
        spec.tone = "maintenance";
        spec.beacon = "maintenance";
        spec.tint = c.maintenance;
        spec.tintAmount = cfg.maintenanceTint;
        break;
      case null:
        spec.tone = "neutral";
        break;
    }
  }
  // Unusual behaviour (autoencoder): the whole machine, never a part, and never over a state
  // outline or a stale / offline look.
  const visible = spec.tone !== "stale" && spec.tone !== "offline";
  if (spec.unusual && visible && spec.outlineParts.length === 0 && spec.flashParts.length === 0) {
    spec.outlineParts = [...input.parts];
    spec.outlineColor = c.unusual;
  }
  if (cfg.labels.enabled) spec.label = { text: stateLabel(input), tone: spec.tone };
  return spec;
}

/** Animation rate from the reported speed: 1 at the reference RPM, clamped to [0, maxRate]. */
export function animationRate(
  speed: number | null | undefined,
  kind: string,
  cfg: VisualConfig,
): number | null {
  if (speed === null || speed === undefined || !Number.isFinite(speed)) return null;
  const ref = (cfg.referenceRpm as Record<string, number | undefined>)[kind] ?? cfg.defaultReferenceRpm;
  return Math.min(cfg.maxRate, Math.max(0, speed / ref));
}

/** Exponential smoothing step towards a target rate. */
export function smoothRate(current: number, target: number, dt: number, response: number): number {
  const k = 1 - Math.exp(-response * Math.max(0, dt));
  return current + (target - current) * k;
}

/** Colour of a tone (label backgrounds, beacons). */
export function toneColor(tone: Tone, cfg: VisualConfig): number {
  const c = cfg.colors;
  switch (tone) {
    case "running":
      return c.running;
    case "warning":
      return c.warning;
    case "fault":
      return c.fault;
    case "waiting":
      return c.waiting;
    case "maintenance":
      return c.maintenance;
    case "stale":
    case "offline":
      return c.offline;
    case "neutral":
      return c.neutral;
  }
}

// ---------------------------------------------------------------- buffers
export type BufferStatus = "FULL" | "EMPTY" | "FLOWING" | "UNKNOWN";

const LINK_DOWN: readonly (MachineState | null | undefined)[] = ["STALE", "OFFLINE", null, undefined];

/**
 * What the data says about a buffer, without inventing a count (ADR 0016): the backend has no
 * buffer telemetry, but STARVED / BLOCKED states are defined by buffer levels in the simulator.
 */
export function bufferStatus(
  from: string,
  fromState: MachineState | null | undefined,
  to: string,
  toState: MachineState | null | undefined,
): { status: BufferStatus; why: string } {
  if (LINK_DOWN.includes(fromState) || LINK_DOWN.includes(toState)) {
    const who = LINK_DOWN.includes(fromState) ? from : to;
    return {
      status: "UNKNOWN",
      why: `${who} has no live data, so the buffer level cannot be inferred.`,
    };
  }
  if (fromState === "BLOCKED") {
    return {
      status: "FULL",
      why: `${from} is BLOCKED: it cannot hand parts on, so this buffer is full.`,
    };
  }
  if (toState === "STARVED") {
    return {
      status: "EMPTY",
      why: `${to} is STARVED: it gets no parts, so this buffer is empty.`,
    };
  }
  return {
    status: "FLOWING",
    why: `Neither ${from} is BLOCKED nor ${to} STARVED: parts are passing; the exact count is not reported.`,
  };
}

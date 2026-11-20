/**
 * Live predictions in the dashboard (Phase 11, ADR 0030): pure view helpers, unit tested.
 *
 * Rules the dashboard must keep:
 * - Part highlights come only from supervised attributions (`Prediction.attributions`, filled
 *   by the backend only while the supervised model predicts WARNING / FAULT). The autoencoder
 *   never names a part: an "unusual behaviour" alarm is shown on the whole machine.
 * - Calibrated probabilities and conformal sets are confidence, not alarms. The alarm is the
 *   evaluated rule (`Prediction.alarm`, and the predicted_* alerts).
 * - No prediction is shown for a stopped or waiting machine, nor one older than the stale time
 *   relative to the time on screen (live: newest data; replay: the replay time).
 */
import type { Alert, MachineState, Prediction, PredictResponse } from "../../api/types.ts";

export type HealthClass = "NORMAL" | "WARNING" | "FAULT";
export const HEALTH_CLASSES: readonly HealthClass[] = ["NORMAL", "WARNING", "FAULT"];

export type GateStatus = "predict" | "warming_up" | "waiting" | "stopped" | "unavailable" | "no_recent";

const STOPPED: ReadonlySet<string> = new Set(["IDLE", "FAULT", "MAINTENANCE", "STALE", "OFFLINE"]);
const WAITING: ReadonlySet<string> = new Set(["STARVED", "BLOCKED"]);

/** The prediction for one horizon of one machine (keys are `f"{horizonS:g}"`). */
export function predictionAt(
  byHorizon: Record<string, Prediction> | undefined,
  horizonS: number,
): Prediction | null {
  return byHorizon?.[String(horizonS)] ?? null;
}

/** Is the prediction recent enough to describe the time on screen? */
export function isFresh(p: Prediction | null, asOf: string | null, staleMs: number): boolean {
  if (p === null) return false;
  if (asOf === null) return true;
  const age = Date.parse(asOf) - Date.parse(p.ts);
  return age <= staleMs;
}

/**
 * Gate status when the backend's live gate is not available (replay, or before /predict
 * answered): from the machine state and the age of its newest prediction.
 */
export function localGate(
  state: MachineState | null,
  p: Prediction | null,
  asOf: string | null,
  staleMs: number,
): GateStatus {
  if (state !== null && STOPPED.has(state)) return "stopped";
  if (state !== null && WAITING.has(state)) return "waiting";
  return isFresh(p, asOf, staleMs) ? "predict" : "no_recent";
}

/** Text shown instead of a prediction. */
export function gateText(status: GateStatus, runLen = 0, window = 0): string | null {
  switch (status) {
    case "predict":
      return null;
    case "stopped":
      return "No prediction (machine stopped)";
    case "waiting":
      return "No prediction (machine waiting)";
    case "warming_up":
      return `Warming up (${String(Math.min(runLen, window))}/${String(window)} s of running data)`;
    case "unavailable":
      return "Predictions unavailable (inference not ready)";
    case "no_recent":
      return "No recent prediction";
  }
}

/** The class with the highest calibrated probability (ties: the first, as the backend). */
export function topClass(p: Prediction): { cls: HealthClass; prob: number } {
  const prob = (c: HealthClass): number => p.probs[c] ?? 0;
  let best: HealthClass = "NORMAL";
  for (const c of HEALTH_CLASSES) if (prob(c) > prob(best)) best = c;
  return { cls: best, prob: prob(best) };
}

/** A conformal set as words: "WARNING or FAULT". */
export function setText(set: readonly string[]): string {
  if (set.length === 0) return "empty set";
  if (set.length === 1) return set[0] ?? "";
  if (set.length === HEALTH_CLASSES.length) return "any state";
  return `${set.slice(0, -1).join(", ")} or ${set[set.length - 1] ?? ""}`;
}

/**
 * The most likely state first, then what the conformal set cannot rule out:
 * "Most likely NORMAL (1.00) · cannot rule out FAULT". The set is a coverage statement (per
 * class), so it can name a state that is not the most likely one.
 */
export function confidenceText(p: Prediction): string {
  const { cls, prob } = topClass(p);
  const others = p.conformalSet.filter((c) => c !== cls);
  const head = `Most likely ${cls} (${prob.toFixed(2)})`;
  if (p.conformalSet.length === 0) return `${head} · empty set`;
  if (others.length === 0) return `${head} · no other state in the set`;
  return `${head} · cannot rule out ${others.join(", ")}`;
}

/** Time to fault with its interval; the model's cap means "no fault expected within it". */
export function ttfText(p: Prediction, capS: number | null): string {
  if (p.ttfS === null || p.ttfS === undefined) return "–";
  const cap = capS ?? Number.POSITIVE_INFINITY;
  if (p.ttfS >= cap - 0.5) return `beyond ${String(Math.round(cap))} s (no fault expected sooner)`;
  const iv = p.ttfInterval;
  const range =
    iv === null || iv === undefined ? "" : ` (interval ${String(Math.round(iv[0]))}–${String(Math.round(iv[1]))} s)`;
  return `about ${String(Math.round(p.ttfS))} s${range}`;
}

/**
 * Parts to highlight from a supervised prediction: the parts with positive evidence for the
 * predicted alarm class, strongest first, intensity relative to the strongest. Empty when the
 * prediction is missing, stale, or has no attributions (NORMAL predictions have none).
 */
export function highlightsFor(
  p: Prediction | null,
  asOf: string | null,
  staleMs: number,
  topN: number,
): { part: string; intensity: number }[] {
  if (p === null || !isFresh(p, asOf, staleMs)) return [];
  const byPart = new Map<string, number>();
  for (const a of p.attributions ?? []) {
    if (a.weight > 0) byPart.set(a.part, (byPart.get(a.part) ?? 0) + a.weight);
  }
  const ranked = [...byPart.entries()].sort((a, b) => b[1] - a[1]).slice(0, topN);
  const max = ranked[0]?.[1] ?? 0;
  if (max <= 0) return [];
  return ranked.map(([part, w]) => ({ part, intensity: w / max }));
}

/** The highlight on screen for one machine, with the window it explains. */
export interface HighlightState {
  parts: { part: string; intensity: number }[];
  /** End of the window the explanation describes (`attributionsTs`, else the prediction time). */
  windowTs: string;
  /** true while the last explanation is kept because a newer one is late. */
  held: boolean;
}

/**
 * Display rule (ADR 0030, Part C): a fresh supervised explanation replaces the highlight. While
 * the predicted alarm stays on and no fresh explanation is attached (explanations arrive
 * beside the prediction path and can be late), the last highlight is kept, labelled with the
 * time of the window it explains. It clears when the alarm clears, when the prediction is
 * stale (the machine stopped), or when time moved backwards. The alarm rule is not touched.
 */
export function resolveHighlight(
  p: Prediction | null,
  prev: HighlightState | null,
  asOf: string | null,
  staleMs: number,
  topN: number,
): HighlightState | null {
  const fresh = highlightsFor(p, asOf, staleMs, topN);
  if (p !== null && fresh.length > 0) {
    return { parts: fresh, windowTs: p.attributionsTs ?? p.ts, held: false };
  }
  if (p === null || prev === null || p.alarm !== true || !isFresh(p, asOf, staleMs)) return null;
  if (Date.parse(prev.windowTs) > Date.parse(p.ts)) return null; // time moved backwards (replay)
  return { ...prev, held: true };
}

/** A stable key of a highlight list (to skip redundant engine calls). */
export function highlightKey(h: readonly { part: string; intensity: number }[]): string {
  return h.map((x) => `${x.part}:${x.intensity.toFixed(2)}`).join("|");
}

export interface UnusualConfig {
  /** Look back this long (event time) for on/off switches. */
  flapWindowS: number;
  /** "Intermittent" from this many switches within the window. */
  flapMinSwitches: number;
  /** Keep the indicator this long after the alert clears. */
  holdS: number;
}

export interface UnusualView {
  /** Show the indicator: the alert is active, or it cleared less than holdS ago. */
  visible: boolean;
  /** An active alert (not only held). */
  active: boolean;
  intermittent: boolean;
  label: string;
}

/** On / off transitions of one machine's unusual-behaviour alerts, from the stored lifecycle events. */
export function unusualSwitches(events: readonly Alert[], machineId: string): { ts: number; on: boolean }[] {
  const byId = new Map<string, Alert[]>();
  for (const e of events) {
    if (e.machineId !== machineId || e.kind !== "unusual_behaviour") continue;
    byId.set(e.alertId, [...(byId.get(e.alertId) ?? []), e]);
  }
  const out: { ts: number; on: boolean }[] = [];
  for (const list of byId.values()) {
    list.sort((a, b) => a.rev - b.rev);
    let prev = false;
    for (const e of list) {
      if (e.active !== prev) out.push({ ts: Date.parse(e.ts), on: e.active });
      prev = e.active;
    }
  }
  return out.sort((a, b) => a.ts - b.ts);
}

/**
 * Display only (the alert lifecycle and the stored events stay as they are): an unusual-behaviour
 * alarm that switched on and off several times within the period shows as "intermittent", and
 * the indicator stays visible for a short hold after it clears. While the supervised alarm is on
 * the hold does not apply: the alarm supersedes "unusual behaviour" (ADR 0030), so the machine is
 * never shown with a part highlight and a whole-machine unusual outline at once.
 */
export function unusualView(
  events: readonly Alert[],
  machineId: string,
  asOf: string | null,
  cfg: UnusualConfig,
  supersededByAlarm = false,
): UnusualView {
  const sw = unusualSwitches(events, machineId);
  const now = asOf === null ? (sw[sw.length - 1]?.ts ?? 0) : Date.parse(asOf);
  const seen = sw.filter((x) => x.ts <= now);
  const recent = seen.filter((x) => now - x.ts <= cfg.flapWindowS * 1000);
  const intermittent = recent.length >= cfg.flapMinSwitches;
  const last = seen[seen.length - 1];
  const active = last?.on === true;
  const held = !supersededByAlarm && last !== undefined && !last.on && now - last.ts <= cfg.holdS * 1000;
  return {
    visible: active || held,
    active,
    intermittent,
    label: intermittent ? "unusual behaviour (intermittent)" : "unusual behaviour",
  };
}

export interface DriftNotice {
  machineId: string;
  /** The backend's message: channels, PSI against the alert level, since when. */
  message: string;
  raisedAt: string;
}

/**
 * Machines with an active model-drift alert, newest first. Drift is the operating conditions
 * (speed, load) differing from the training data: predictions are then marked DEGRADED and
 * their confidence may not hold. It comes from the alert, so replay shows it as it was.
 */
export function activeDrift(alerts: Iterable<Alert>): DriftNotice[] {
  const out: DriftNotice[] = [];
  for (const a of alerts) {
    if (a.kind === "model_drift" && a.active) out.push({ machineId: a.machineId, message: a.message, raisedAt: a.ts });
  }
  return out.sort((x, y) => Date.parse(y.raisedAt) - Date.parse(x.raisedAt));
}

/** How an alert kind is shown: label, icon, CSS class and a note. */
export interface AlertKindView {
  label: string;
  icon: string;
  className: string;
  note: string | null;
}

export function alertKindView(kind: string): AlertKindView {
  switch (kind) {
    case "unusual_behaviour":
      return {
        label: "unusual behaviour",
        icon: "?",
        className: "kind-unusual",
        note: "cause unknown: the autoencoder flags inputs unlike normal operation; no part is named",
      };
    case "predicted_warning":
      return { label: "predicted WARNING", icon: "◔", className: "kind-predicted", note: "model prediction, +30 s" };
    case "predicted_fault":
      return { label: "predicted FAULT", icon: "◕", className: "kind-predicted", note: "model prediction, +30 s" };
    case "model_drift":
      return {
        label: "model drift",
        icon: "≈",
        className: "kind-drift",
        note: "operating conditions differ from the training data; predictions are DEGRADED",
      };
    default:
      return { label: kind, icon: "", className: "", note: null };
  }
}

/** The predicted class as a tone for badges. */
export function classTone(cls: HealthClass): "ok" | "warning" | "fault" {
  return cls === "NORMAL" ? "ok" : cls === "WARNING" ? "warning" : "fault";
}

export interface GateView {
  status: GateStatus;
  text: string | null;
}

/** The gate of one machine: the backend's live gate, or derived locally (replay). */
export function gateView(
  inference: PredictResponse | null,
  mode: "live" | "replay",
  id: string,
  state: MachineState | null,
  primary: Prediction | null,
  asOf: string | null,
  staleMs: number,
): GateView {
  const local = localGate(state, primary, asOf, staleMs);
  const g = mode === "live" ? inference?.machines?.[id]?.gate : undefined;
  if (g === undefined) return { status: local, text: gateText(local) };
  // the stream can be a moment ahead of the last poll: a fresh prediction wins
  const status: GateStatus = local === "predict" ? "predict" : g.status;
  return { status, text: gateText(status, g.runLen, g.window) };
}

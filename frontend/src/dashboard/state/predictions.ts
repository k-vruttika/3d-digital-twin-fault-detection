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
import type { MachineState, Prediction, PredictResponse } from "../../api/types.ts";

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

/** A stable key of a highlight list (to skip redundant engine calls). */
export function highlightKey(h: readonly { part: string; intensity: number }[]): string {
  return h.map((x) => `${x.part}:${x.intensity.toFixed(2)}`).join("|");
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
      return { label: "model drift", icon: "≈", className: "kind-drift", note: "inputs differ from the training data" };
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

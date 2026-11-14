/**
 * Which 3D labels are shown, and how big and opaque they are (pure functions, unit tested).
 *
 * Declutter rule: by default a machine label shows only when the machine is not in a normal
 * state (anything but `labels.normalStates`, i.e. RUNNING / IDLE; "NO DATA" and "LINK LOST"
 * count as not normal), and a buffer label only for `labels.bufferStatuses` (FULL / EMPTY /
 * UNKNOWN). Hovering or selecting shows a label anyway, and the label mode "all" shows every one.
 *
 * Size: labels keep a constant height on screen (`labels.pixelHeight`); on a viewport smaller
 * than `labels.referenceViewportPx` they shrink with it, never below `labels.minPixelHeight`.
 * Distance: beyond `labels.fadeStart` metres from the camera a label fades out and is hidden at
 * `labels.hideDistance` (too far to belong to what you are looking at); hovered and selected
 * labels never fade.
 */
import type { MachineState } from "../api/types.ts";
import type { BufferStatus } from "./visuals.ts";

export type LabelMode = "auto" | "all";

export interface LabelSizing {
  pixelHeight: number;
  minPixelHeight: number;
  referenceViewportPx: number;
  fadeStart: number;
  hideDistance: number;
}

export interface LabelFlags {
  mode: LabelMode;
  hovered: boolean;
  selected: boolean;
}

/** Below this opacity a label is not drawn at all. */
export const MIN_VISIBLE_OPACITY = 0.02;

export function machineLabelVisible(
  state: MachineState | null,
  connected: boolean,
  normalStates: readonly MachineState[],
  flags: LabelFlags,
): boolean {
  if (flags.mode === "all" || flags.hovered || flags.selected) return true;
  if (!connected || state === null) return true; // LINK LOST / NO DATA
  return !normalStates.includes(state);
}

export function bufferLabelVisible(
  status: BufferStatus,
  shownStatuses: readonly BufferStatus[],
  flags: LabelFlags,
): boolean {
  if (flags.mode === "all" || flags.hovered || flags.selected) return true;
  return shownStatuses.includes(status);
}

/** On-screen label height in px for a viewport of `viewportPx` pixels height. */
export function labelPixelHeight(viewportPx: number, s: LabelSizing): number {
  const scaled = s.pixelHeight * Math.min(1, Math.max(0, viewportPx) / s.referenceViewportPx);
  return Math.max(s.minPixelHeight, scaled);
}

/**
 * World height that shows `labelPixelHeight` px at `distance` metres with a vertical field of
 * view of `fovDeg`, and the label's opacity at that distance.
 */
export function labelScreenScale(
  distance: number,
  fovDeg: number,
  viewportPx: number,
  s: LabelSizing,
  exempt: boolean,
): { worldHeight: number; opacity: number } {
  const d = Math.max(0, distance);
  const h = Math.max(1, viewportPx);
  const worldPerPx = (2 * d * Math.tan((fovDeg * Math.PI) / 360)) / h;
  const worldHeight = labelPixelHeight(h, s) * worldPerPx;
  if (exempt || d <= s.fadeStart) return { worldHeight, opacity: 1 };
  if (d >= s.hideDistance) return { worldHeight, opacity: 0 };
  return { worldHeight, opacity: 1 - (d - s.fadeStart) / (s.hideDistance - s.fadeStart) };
}

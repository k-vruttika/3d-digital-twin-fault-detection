/**
 * Chart annotations (DESIGN.md §8), drawn by one inline Chart.js plugin (no dependency):
 *
 * - NO DATA bands: every run of null points in the measured series is hatched, and labelled
 *   "NO DATA" when it is at least 48 px wide. The line itself still breaks there
 *   (spanGaps: false); the band only makes the gap impossible to misread as a flat value.
 * - alert markers: a vertical 1 px line at each alert's raise time for the charted machine, in
 *   its status colour, with a 6 px cap and a micro label ("FAULT 00:12:57"). Predicted kinds are
 *   dashed; unusual behaviour is purple. Built from alerts already in the model.
 *
 * No threshold lines: the backend provides no alarm limits (docs/frontend.md).
 */
import type { Chart, Plugin } from "chart.js";

import type { Alert } from "../../api/types.ts";
import { formatTime } from "../format.ts";
import type { ChartPoint } from "./series.ts";

export type MarkerTone = "fault" | "warning" | "maintenance" | "unusual" | "degraded";

export interface ChartMarker {
  /** epoch ms of the raise */
  x: number;
  tone: MarkerTone;
  /** predicted (model) alerts are dashed, like every other prediction */
  dashed: boolean;
  label: string;
  alertId: string;
}

export interface AnnotationStyle {
  hatch: string;
  noDataText: string;
  font: string;
  /** marker lines and caps (status fills) */
  tones: Record<MarkerTone, string>;
  /** marker labels (status inks: AA as text on the panel) */
  inks: Record<MarkerTone, string>;
}

/** Minimum band width (px) that gets a "NO DATA" label. */
export const NO_DATA_LABEL_MIN_PX = 48;
const CAP_PX = 6;
const HATCH_PX = 6;
const LABEL_GAP_PX = 64;

const SEVERITY_TONE: Record<Alert["severity"], MarkerTone> = { CRITICAL: "fault", WARNING: "warning", INFO: "maintenance" };

function toneOf(a: Alert): { tone: MarkerTone; dashed: boolean } {
  switch (a.kind) {
    case "unusual_behaviour":
      return { tone: "unusual", dashed: false };
    case "predicted_fault":
      return { tone: "fault", dashed: true };
    case "predicted_warning":
      return { tone: "warning", dashed: true };
    case "model_drift":
      return { tone: "degraded", dashed: false };
    default:
      return { tone: SEVERITY_TONE[a.severity], dashed: false };
  }
}

/** "machine_fault" -> "FAULT", "unusual_behaviour" -> "UNUSUAL BEHAVIOUR". */
export function markerKindLabel(kind: string): string {
  return kind.replace(/^machine_/, "").replace(/_/g, " ").toUpperCase();
}

/**
 * One marker per alert raised on `machineId`, at its raise time: from the alert's own lifecycle
 * events (rev 0, active) where known, else from a current alert that is still at rev 0.
 */
export function alertMarkers(
  alerts: Iterable<Alert>,
  events: ReadonlyMap<string, readonly Alert[]>,
  machineId: string,
): ChartMarker[] {
  const raises = new Map<string, Alert>();
  for (const list of events.values()) {
    for (const e of list) if (e.machineId === machineId && e.rev === 0 && e.active) raises.set(e.alertId, e);
  }
  for (const a of alerts) if (a.machineId === machineId && a.rev === 0 && a.active && !raises.has(a.alertId)) raises.set(a.alertId, a);
  const out: ChartMarker[] = [];
  for (const a of raises.values()) {
    const x = Date.parse(a.ts);
    if (!Number.isFinite(x)) continue;
    out.push({ x, ...toneOf(a), label: `${markerKindLabel(a.kind)} ${formatTime(a.ts)}`, alertId: a.alertId });
  }
  return out.sort((p, q) => p.x - q.x);
}

/** Runs of null points as [fromX, toX] in data units: from the last value before to the first after. */
export function gapRuns(points: readonly ChartPoint[]): [number, number][] {
  const runs: [number, number][] = [];
  let i = 0;
  while (i < points.length) {
    if (points[i]?.y !== null) {
      i++;
      continue;
    }
    const start = i;
    while (i < points.length && points[i]?.y === null) i++;
    const before = points[start - 1];
    const after = points[i];
    const from = before?.x ?? points[start]?.x;
    const to = after?.x ?? points[i - 1]?.x;
    if (from !== undefined && to !== undefined && to > from) runs.push([from, to]);
  }
  return runs;
}

interface AnnotationState {
  markers: readonly ChartMarker[];
  style: AnnotationStyle;
}

function measured(chart: Chart): readonly ChartPoint[] {
  const ds = chart.data.datasets.at(-1);
  return (ds?.data ?? []) as unknown as readonly ChartPoint[];
}

/**
 * The plugin reads its state through `get` on every draw, so markers and colours follow the
 * component's latest props without rebuilding the chart.
 */
export function annotationPlugin(get: () => AnnotationState): Plugin<"line"> {
  return {
    id: "linetwinAnnotations",
    beforeDatasetsDraw(chart) {
      const { style } = get();
      const x = chart.scales.x;
      const area = chart.chartArea;
      if (x === undefined) return;
      const ctx = chart.ctx;
      for (const [from, to] of gapRuns(measured(chart))) {
        const left = Math.max(area.left, x.getPixelForValue(from));
        const right = Math.min(area.right, x.getPixelForValue(to));
        const w = right - left;
        if (w <= 1) continue;
        ctx.save();
        ctx.beginPath();
        ctx.rect(left, area.top, w, area.bottom - area.top);
        ctx.clip();
        ctx.strokeStyle = style.hatch;
        ctx.lineWidth = 1;
        ctx.beginPath();
        const h = area.bottom - area.top;
        for (let sx = left - h; sx < right; sx += HATCH_PX) {
          ctx.moveTo(sx, area.bottom);
          ctx.lineTo(sx + h, area.top);
        }
        ctx.stroke();
        ctx.restore();
        if (w >= NO_DATA_LABEL_MIN_PX) {
          ctx.save();
          ctx.font = `800 11px ${style.font}`;
          ctx.fillStyle = style.noDataText;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText("NO DATA", left + w / 2, area.top + 4);
          ctx.restore();
        }
      }
    },
    afterDatasetsDraw(chart) {
      const { markers, style } = get();
      const x = chart.scales.x;
      const area = chart.chartArea;
      if (x === undefined || markers.length === 0) return;
      const ctx = chart.ctx;
      let lastLabelPx = Number.NEGATIVE_INFINITY;
      for (const m of markers) {
        if (m.x < x.min || m.x > x.max) continue;
        const px = Math.round(x.getPixelForValue(m.x)) + 0.5;
        const color = style.tones[m.tone];
        const ink = style.inks[m.tone];
        ctx.save();
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash(m.dashed ? [4, 3] : []);
        ctx.beginPath();
        ctx.moveTo(px, area.top);
        ctx.lineTo(px, area.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(px - CAP_PX / 2, area.top);
        ctx.lineTo(px + CAP_PX / 2, area.top);
        ctx.lineTo(px, area.top + CAP_PX);
        ctx.closePath();
        ctx.fill();
        if (px - lastLabelPx >= LABEL_GAP_PX) {
          ctx.font = `800 11px ${style.font}`;
          const tw = ctx.measureText(m.label).width;
          const right = px + 4 + tw <= area.right;
          ctx.fillStyle = ink;
          ctx.textAlign = right ? "left" : "right";
          ctx.textBaseline = "top";
          ctx.fillText(m.label, right ? px + 4 : px - 4, area.top + CAP_PX + 2);
          lastLabelPx = px;
        }
        ctx.restore();
      }
    },
  };
}

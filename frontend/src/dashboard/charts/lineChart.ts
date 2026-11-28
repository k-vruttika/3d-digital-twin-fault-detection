/**
 * Chart.js configuration for the sensor charts (pure; unit tested). Styling: DESIGN.md §8.
 *
 * - x is a linear axis in epoch ms with UTC clock ticks (no date adapter dependency);
 * - y has the sensor name and unit as its title;
 * - `spanGaps: false` on every dataset: a null point is a gap, never interpolated or zero;
 * - an optional min-max band (bucketed history) is drawn as a filled area behind the mean;
 * - the trend is neutral (charts show measurements, not status); horizontal hairlines only;
 *   a small dot marks the newest value;
 * - no threshold lines: the backend provides no alarm limits (see docs/frontend.md).
 */
import type { ScriptableContext, TooltipItem } from "chart.js";

import { formatTime } from "../format.ts";
import type { ChartMarker } from "./annotations.ts";
import type { LineConfig } from "./chartFactory.ts";
import type { ChartTheme } from "./chartTheme.ts";
import type { ChartPoint } from "./series.ts";

export type { ChartTheme } from "./chartTheme.ts";

export interface SensorChartData {
  label: string;
  yTitle: string;
  points: ChartPoint[];
  band?: { min: ChartPoint[]; max: ChartPoint[] } | null;
  xMin?: number | null;
  xMax?: number | null;
  /** alert markers for the charted machine (drawn by the annotation plugin) */
  markers?: readonly ChartMarker[];
}

/** Index of the newest non-null point (the "now" dot), or -1. */
export function lastValueIndex(points: readonly ChartPoint[]): number {
  for (let i = points.length - 1; i >= 0; i--) if (points[i]?.y !== null && points[i]?.y !== undefined) return i;
  return -1;
}

/** Axis, tick and tooltip styling shared by every dashboard chart. */
export function axisStyle(t: ChartTheme) {
  return {
    tick: { color: t.tick, font: { family: t.fontMono, size: 11 } },
    title: { color: t.title, font: { family: t.fontDisplay, size: 11, weight: 800 } },
    tooltip: {
      backgroundColor: t.tooltipBg,
      borderColor: t.tooltipBorder,
      borderWidth: 1,
      cornerRadius: 4,
      padding: 8,
      displayColors: false,
      titleColor: t.tooltipText,
      bodyColor: t.tooltipText,
      titleFont: { family: t.fontMono, size: 11, weight: 400 },
      bodyFont: { family: t.fontMono, size: 12, weight: 500 },
    },
  } as const;
}

export function lineChartConfig(d: SensorChartData, t: ChartTheme): LineConfig {
  const datasets: LineConfig["data"]["datasets"] = [];
  const s = axisStyle(t);
  if (d.band) {
    datasets.push(
      {
        label: `${d.label} min`,
        data: d.band.min,
        borderWidth: 0,
        pointRadius: 0,
        spanGaps: false,
        fill: false,
        borderColor: "transparent",
      },
      {
        label: `${d.label} max`,
        data: d.band.max,
        borderWidth: 0,
        pointRadius: 0,
        spanGaps: false,
        fill: "-1", // fill down to the min dataset: the min-max band
        backgroundColor: t.band,
        borderColor: "transparent",
      },
    );
  }
  const last = lastValueIndex(d.points);
  datasets.push({
    label: d.label,
    data: d.points,
    borderColor: t.line,
    backgroundColor: t.line,
    borderWidth: 1.25,
    pointRadius: (ctx: ScriptableContext<"line">) => (ctx.dataIndex === last ? 2.5 : 0),
    pointHitRadius: 6,
    spanGaps: false,
    tension: 0,
    fill: false,
  });
  return {
    type: "line",
    data: { datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      normalized: true,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...s.tooltip,
          filter: (item: TooltipItem<"line">) => item.datasetIndex === datasets.length - 1,
          callbacks: {
            title: (items: TooltipItem<"line">[]) => `${formatTime(items[0]?.parsed.x ?? null)} UTC`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          ...(d.xMin === null || d.xMin === undefined ? {} : { min: d.xMin }),
          ...(d.xMax === null || d.xMax === undefined ? {} : { max: d.xMax }),
          grid: { display: false },
          border: { display: false },
          ticks: {
            ...s.tick,
            maxTicksLimit: 6,
            maxRotation: 0,
            callback: (v: number | string) => formatTime(Number(v)),
          },
          title: { display: true, text: "TIME · UTC", ...s.title },
        },
        y: {
          type: "linear",
          grid: { color: t.grid, drawTicks: false },
          border: { display: false },
          ticks: { ...s.tick, maxTicksLimit: 4, padding: 8 },
          title: { display: true, text: d.yTitle, ...s.title },
        },
      },
    },
  };
}

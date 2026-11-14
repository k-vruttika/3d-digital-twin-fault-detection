/**
 * Chart.js configuration for the sensor charts (pure; unit tested).
 *
 * - x is a linear axis in epoch ms with UTC clock ticks (no date adapter dependency);
 * - y has the sensor name and unit as its title;
 * - `spanGaps: false` on every dataset: a null point is a gap, never interpolated or zero;
 * - an optional min-max band (bucketed history) is drawn as a filled area behind the mean;
 * - no threshold lines: the backend provides no alarm limits (see docs/frontend.md).
 */
import type { TooltipItem } from "chart.js";

import { formatTime } from "../format.ts";
import type { LineConfig } from "./chartFactory.ts";
import type { ChartPoint } from "./series.ts";

export interface ChartColors {
  line: string;
  band: string;
  grid: string;
  text: string;
}

export const CHART_COLORS: Record<"light" | "dark", ChartColors> = {
  light: { line: "#1f6fd1", band: "rgba(31, 111, 209, 0.16)", grid: "rgba(0, 0, 0, 0.08)", text: "#4a5563" },
  dark: { line: "#6aa8f0", band: "rgba(106, 168, 240, 0.2)", grid: "rgba(255, 255, 255, 0.09)", text: "#aeb7c2" },
};

export interface SensorChartData {
  label: string;
  yTitle: string;
  points: ChartPoint[];
  band?: { min: ChartPoint[]; max: ChartPoint[] } | null;
  xMin?: number | null;
  xMax?: number | null;
}

export function lineChartConfig(d: SensorChartData, colors: ChartColors): LineConfig {
  const datasets: LineConfig["data"]["datasets"] = [];
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
        backgroundColor: colors.band,
        borderColor: "transparent",
      },
    );
  }
  datasets.push({
    label: d.label,
    data: d.points,
    borderColor: colors.line,
    backgroundColor: colors.line,
    borderWidth: 1.6,
    pointRadius: 0,
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
          filter: (item: TooltipItem<"line">) => item.datasetIndex === datasets.length - 1,
          callbacks: {
            title: (items: TooltipItem<"line">[]) => formatTime(items[0]?.parsed.x ?? null),
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          ...(d.xMin === null || d.xMin === undefined ? {} : { min: d.xMin }),
          ...(d.xMax === null || d.xMax === undefined ? {} : { max: d.xMax }),
          grid: { color: colors.grid },
          ticks: {
            color: colors.text,
            maxTicksLimit: 6,
            maxRotation: 0,
            callback: (v: number | string) => formatTime(Number(v)),
          },
          title: { display: true, text: "time (UTC)", color: colors.text },
        },
        y: {
          type: "linear",
          grid: { color: colors.grid },
          ticks: { color: colors.text, maxTicksLimit: 5 },
          title: { display: true, text: d.yTitle, color: colors.text },
        },
      },
    },
  };
}

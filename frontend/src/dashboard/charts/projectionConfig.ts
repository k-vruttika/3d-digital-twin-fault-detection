/**
 * Chart.js configuration of a what-if projection (pure; unit tested).
 *
 * x is seconds after the fork point ("now" = 0): the solid line is what happened, the dashed
 * lines are projections, never data. Dashed grey = nothing changed, dashed accent = the change.
 * A null value is a gap (`spanGaps: false`), as everywhere else.
 */
import type { LineConfig } from "./chartFactory.ts";
import type { ChartColors } from "./lineChart.ts";
import type { ChartPoint } from "./series.ts";

export interface ProjectionColors extends ChartColors {
  baseline: string;
  scenario: string;
}

export const PROJECTION_COLORS: Record<"light" | "dark", ProjectionColors> = {
  light: { line: "#1f6fd1", band: "", grid: "rgba(0, 0, 0, 0.08)", text: "#4a5563", baseline: "#6b7480", scenario: "#c4620a" },
  dark: { line: "#6aa8f0", band: "", grid: "rgba(255, 255, 255, 0.09)", text: "#aeb7c2", baseline: "#9aa4b0", scenario: "#f0a35e" },
};

export const DASH: [number, number] = [6, 4];

export interface ProjectionChartData {
  label: string;
  yTitle: string;
  history: ChartPoint[];
  baseline: ChartPoint[];
  scenario: ChartPoint[];
  /** Fixed y range (probabilities), else automatic. */
  yMin?: number | null;
  yMax?: number | null;
}

export function projectionChartConfig(d: ProjectionChartData, colors: ProjectionColors): LineConfig {
  const common = { pointRadius: 0, pointHitRadius: 6, spanGaps: false, tension: 0, fill: false } as const;
  return {
    type: "line",
    data: {
      datasets: [
        { ...common, label: "recent (measured)", data: d.history, borderColor: colors.line, backgroundColor: colors.line, borderWidth: 1.8 },
        {
          ...common,
          label: "projected, no change",
          data: d.baseline,
          borderColor: colors.baseline,
          backgroundColor: colors.baseline,
          borderWidth: 1.8,
          borderDash: DASH,
        },
        {
          ...common,
          label: "projected, with the change",
          data: d.scenario,
          borderColor: colors.scenario,
          backgroundColor: colors.scenario,
          borderWidth: 1.8,
          borderDash: DASH,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      normalized: true,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { display: true, labels: { color: colors.text, boxWidth: 24 } },
        tooltip: { callbacks: { title: (items) => `${String(items[0]?.parsed.x ?? "")} s` } },
      },
      scales: {
        x: {
          type: "linear",
          grid: { color: colors.grid },
          ticks: { color: colors.text, maxTicksLimit: 8, callback: (v) => `${String(v)} s` },
          title: { display: true, text: "seconds from now (0 = fork point)", color: colors.text },
        },
        y: {
          type: "linear",
          ...(d.yMin === null || d.yMin === undefined ? {} : { min: d.yMin }),
          ...(d.yMax === null || d.yMax === undefined ? {} : { max: d.yMax }),
          grid: { color: colors.grid },
          ticks: { color: colors.text, maxTicksLimit: 5 },
          title: { display: true, text: d.yTitle, color: colors.text },
        },
      },
    },
  };
}

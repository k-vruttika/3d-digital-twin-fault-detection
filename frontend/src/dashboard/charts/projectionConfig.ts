/**
 * Chart.js configuration of a what-if projection (pure; unit tested).
 *
 * x is seconds after the fork point ("now" = 0): the solid line is what happened, the dashed
 * lines are projections, never data. Dashed grey = nothing changed, dashed accent (bone) = the change
 * (DESIGN.md §8).
 * A null value is a gap (`spanGaps: false`), as everywhere else.
 */
import type { LineConfig } from "./chartFactory.ts";
import type { ChartTheme } from "./chartTheme.ts";
import { axisStyle } from "./lineChart.ts";
import type { ChartPoint } from "./series.ts";

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

export function projectionChartConfig(d: ProjectionChartData, colors: ChartTheme): LineConfig {
  const st = axisStyle(colors);
  const common = { pointRadius: 0, pointHitRadius: 6, spanGaps: false, tension: 0, fill: false } as const;
  return {
    type: "line",
    data: {
      datasets: [
        { ...common, label: "recent (measured)", data: d.history, borderColor: colors.line, backgroundColor: colors.line, borderWidth: 1.5 },
        {
          ...common,
          label: "projected, no change",
          data: d.baseline,
          borderColor: colors.baseline,
          backgroundColor: colors.baseline,
          borderWidth: 1.5,
          borderDash: DASH,
        },
        {
          ...common,
          label: "projected, with the change",
          data: d.scenario,
          borderColor: colors.scenario,
          backgroundColor: colors.scenario,
          borderWidth: 1.5,
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
        legend: { display: true, labels: { color: colors.title, boxWidth: 24, boxHeight: 1, font: { family: colors.fontBody, size: 12 } } },
        tooltip: { ...st.tooltip, callbacks: { title: (items) => `${String(items[0]?.parsed.x ?? "")} s` } },
      },
      scales: {
        x: {
          type: "linear",
          grid: { display: false },
          border: { display: false },
          ticks: { ...st.tick, maxTicksLimit: 8, callback: (v) => `${String(v)} s` },
          title: { display: true, text: "SECONDS FROM NOW · 0 = FORK POINT", ...st.title },
        },
        y: {
          type: "linear",
          ...(d.yMin === null || d.yMin === undefined ? {} : { min: d.yMin }),
          ...(d.yMax === null || d.yMax === undefined ? {} : { max: d.yMax }),
          grid: { color: colors.grid, drawTicks: false },
          border: { display: false },
          ticks: { ...st.tick, maxTicksLimit: 4, padding: 8 },
          title: { display: true, text: d.yTitle, ...st.title },
        },
      },
    },
  };
}

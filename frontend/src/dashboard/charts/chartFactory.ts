/**
 * Chart.js behind a small injectable factory (like the engine's injectable renderer), so
 * components are testable in jsdom without a canvas. The real factory loads Chart.js with a
 * dynamic import (its own chunk) and registers only what the line charts use.
 */
import type { Chart, ChartConfiguration } from "chart.js";

import type { ChartPoint } from "./series.ts";

export type LineConfig = ChartConfiguration<"line", ChartPoint[]>;

/** The part of a Chart.js chart the components use. */
export interface ChartLike {
  data: LineConfig["data"];
  options: NonNullable<LineConfig["options"]>;
  update(mode?: "none"): void;
  destroy(): void;
  /** Time (epoch ms) at a canvas x pixel, for drag-to-zoom; undefined outside the plot. */
  valueAtPixel(x: number): number | undefined;
}

/** Resolves to null when `signal` was aborted before the chart could be created. */
export type ChartFactory = (canvas: HTMLCanvasElement, config: LineConfig, signal: AbortSignal) => Promise<ChartLike | null>;

let loaded: Promise<typeof Chart> | null = null;

async function loadChartJs(): Promise<typeof Chart> {
  loaded ??= import("chart.js").then((m) => {
    m.Chart.register(
      m.LineController,
      m.LineElement,
      m.PointElement,
      m.LinearScale,
      m.Filler,
      m.Tooltip,
      m.Legend,
    );
    return m.Chart;
  });
  return loaded;
}

export const chartJsFactory: ChartFactory = async (canvas, config, signal) => {
  const ChartCtor = await loadChartJs();
  if (signal.aborted) return null; // unmounted (or StrictMode's first pass) while loading
  const chart = new ChartCtor(canvas, config);
  return {
    get data() {
      return chart.data;
    },
    get options() {
      return chart.options;
    },
    update: (mode) => {
      chart.update(mode);
    },
    destroy: () => {
      chart.destroy();
    },
    valueAtPixel: (x) => {
      const scale = chart.scales.x;
      if (scale === undefined || x < scale.left || x > scale.right) return undefined;
      return scale.getValueForPixel(x);
    },
  };
};

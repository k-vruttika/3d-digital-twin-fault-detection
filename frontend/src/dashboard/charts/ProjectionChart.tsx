/** One Chart.js what-if projection chart; created once, updated in place (like SensorChart). */
import { useEffect, useLayoutEffect, useRef } from "react";

import { useServices } from "../context.ts";
import { useTheme } from "../theme.ts";
import type { ChartLike } from "./chartFactory.ts";
import { chartTheme } from "./chartTheme.ts";
import { projectionChartConfig, type ProjectionChartData } from "./projectionConfig.ts";

interface Props {
  data: ProjectionChartData;
  height?: number;
  testId?: string;
}

export function ProjectionChart({ data, height = 220, testId }: Props) {
  const { chartFactory } = useServices();
  const theme = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<ChartLike | null>(null);
  const dataRef = useRef(data);
  useLayoutEffect(() => {
    dataRef.current = data;
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ctl = new AbortController();
    let chart: ChartLike | null = null;
    void chartFactory(canvas, projectionChartConfig(dataRef.current, chartTheme()), ctl.signal).then((c) => {
      if (c === null) return;
      if (ctl.signal.aborted) {
        c.destroy();
        return;
      }
      chart = c;
      chartRef.current = c;
      const next = projectionChartConfig(dataRef.current, chartTheme());
      c.data.datasets = next.data.datasets;
      c.options.scales = next.options?.scales ?? {};
      c.update("none");
    });
    return () => {
      ctl.abort();
      chart?.destroy();
      chartRef.current = null;
    };
  }, [chartFactory, theme]);

  useEffect(() => {
    const chart = chartRef.current;
    if (chart === null) return;
    const next = projectionChartConfig(data, chartTheme());
    chart.data.datasets = next.data.datasets;
    chart.options.scales = next.options?.scales ?? {};
    chart.update("none");
  }, [data, theme]);

  return (
    <div className="chart" style={{ height }} data-testid={testId}>
      <canvas ref={canvasRef} role="img" aria-label={`${data.label} projection, ${data.yTitle}`} />
    </div>
  );
}

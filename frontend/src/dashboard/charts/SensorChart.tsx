/** One Chart.js line chart; created once, updated in place when its data changes. */
import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import { useServices } from "../context.ts";
import { useTheme } from "../theme.ts";
import { annotationPlugin } from "./annotations.ts";
import type { ChartLike } from "./chartFactory.ts";
import { chartTheme, type ChartTheme } from "./chartTheme.ts";
import { lineChartConfig, type SensorChartData } from "./lineChart.ts";

interface Props {
  data: SensorChartData;
  /** Drag across the plot to zoom to that time range (epoch ms). */
  onZoom?: (from: number, to: number) => void;
  /** Pixel height, or "fill" to take the remaining height of a flex column. */
  height?: number | "fill";
  testId?: string;
}

/** Minimum drag width (px) that counts as a zoom rather than a click. */
const MIN_DRAG_PX = 8;

export function SensorChart({ data, onZoom, height = 140, testId }: Props) {
  const { chartFactory } = useServices();
  const theme = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<ChartLike | null>(null);
  const dataRef = useRef(data);
  const themeRef = useRef<ChartTheme | null>(null);
  useLayoutEffect(() => {
    dataRef.current = data;
  });
  const [drag, setDrag] = useState<{ x0: number; x1: number } | null>(null);

  // (Re)create the chart when the theme changes; data updates go through the effect below.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ctl = new AbortController();
    let chart: ChartLike | null = null;
    const t = chartTheme();
    themeRef.current = t;
    // NO DATA bands and alert markers (DESIGN.md §8); read the latest props on every draw
    const annotations = annotationPlugin(() => ({
      markers: dataRef.current.markers ?? [],
      style: (themeRef.current ?? t).annotations,
    }));
    const config = { ...lineChartConfig(dataRef.current, t), plugins: [annotations] };
    void chartFactory(canvas, config, ctl.signal).then((c) => {
      if (c === null) return;
      if (ctl.signal.aborted) {
        c.destroy();
        return;
      }
      chart = c;
      chartRef.current = c;
      // data may have changed while Chart.js was loading
      const next = lineChartConfig(dataRef.current, chartTheme());
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
    const next = lineChartConfig(data, chartTheme());
    chart.data.datasets = next.data.datasets;
    chart.options.scales = next.options?.scales ?? {};
    chart.update("none");
  }, [data, theme]);

  const local = (e: ReactPointerEvent): number => {
    const rect = e.currentTarget.getBoundingClientRect();
    return e.clientX - rect.left;
  };

  const zoomable = onZoom !== undefined;
  return (
    <div
      className={height === "fill" ? "chart chart-fill" : "chart"}
      style={height === "fill" ? undefined : { height }}
      data-testid={testId}
      onPointerDown={zoomable ? (e) => { const x = local(e); setDrag({ x0: x, x1: x }); } : undefined}
      onPointerMove={zoomable && drag !== null ? (e) => { setDrag({ ...drag, x1: local(e) }); } : undefined}
      onPointerUp={
        zoomable && drag !== null
          ? () => {
              const chart = chartRef.current;
              const [a, b] = [Math.min(drag.x0, drag.x1), Math.max(drag.x0, drag.x1)];
              setDrag(null);
              if (chart === null || b - a < MIN_DRAG_PX) return;
              const from = chart.valueAtPixel(a);
              const to = chart.valueAtPixel(b);
              if (from !== undefined && to !== undefined && to > from) onZoom(from, to);
            }
          : undefined
      }
      onPointerLeave={zoomable ? () => { setDrag(null); } : undefined}
    >
      <canvas ref={canvasRef} role="img" aria-label={`${data.label} chart, ${data.yTitle}`} />
      {drag !== null && Math.abs(drag.x1 - drag.x0) >= MIN_DRAG_PX && (
        <div className="chart-drag" style={{ left: Math.min(drag.x0, drag.x1), width: Math.abs(drag.x1 - drag.x0) }} />
      )}
    </div>
  );
}

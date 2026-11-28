/**
 * Live sensor charts for the selected machine: one small chart per sensor over the last
 * `liveWindowS` seconds of event time. Prefilled from `/history` (raw), then fed from the
 * stream. Readings arrive outside React (model.onReading) and are flushed into state at most
 * once per animation frame.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import type { LineResponse, Reading, SensorField } from "../../api/types.ts";
import { useModel, useRest } from "../context.ts";
import { axisTitle } from "../format.ts";
import { useStore } from "../state/store.ts";
import { readingsToSeries, trimWindow } from "./series.ts";
import { SensorChart } from "./SensorChart.tsx";
import { useAlertMarkers } from "./useAlertMarkers.ts";

const EPS_MS = 1;

function merge(older: readonly Reading[], newer: readonly Reading[], windowS: number): Reading[] {
  const first = newer[0];
  const keep = first === undefined ? older : older.filter((r) => Date.parse(r.ts) < Date.parse(first.ts));
  return trimWindow([...keep, ...newer], windowS);
}

function MachineLiveCharts({ id, line }: { id: string; line: LineResponse }) {
  const model = useModel();
  const rest = useRest();
  const mode = useStore(model.modeStore);
  const cfg = model.cfg;
  const [readings, setReadings] = useState<Reading[]>([]);
  const markers = useAlertMarkers(id);
  const pending = useRef<Reading[]>([]);
  const frame = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const flush = (): void => {
      frame.current = null;
      const batch = pending.current;
      pending.current = [];
      if (batch.length > 0) setReadings((prev) => trimWindow([...prev, ...batch], cfg.liveWindowS));
    };
    const off = model.onReading((r) => {
      if (r.machineId !== id || model.mode !== "live") return;
      pending.current.push(r);
      if (frame.current === null) {
        frame.current =
          typeof requestAnimationFrame === "function" ? requestAnimationFrame(flush) : (setTimeout(flush, 16) as unknown as number);
      }
    });
    // Prefill with the last window before the newest reading we know of (event time).
    const latest = model.machines.get()[id]?.reading?.ts ?? model.asOf.get();
    if (latest !== null) {
      const to = Date.parse(latest) + EPS_MS;
      const from = to - cfg.liveWindowS * 1000;
      rest
        .getHistory({ from: new Date(from).toISOString(), to: new Date(to).toISOString(), resolution: "raw", machines: [id] })
        .then((h) => {
          if (!cancelled) setReadings((live) => merge(h.readings ?? [], live, cfg.liveWindowS));
        })
        .catch(() => {
          // no history (e.g. store disabled): the charts fill from the stream
        });
    }
    return () => {
      cancelled = true;
      off();
      if (frame.current !== null && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame.current);
      frame.current = null;
    };
  }, [id, model, rest, cfg.liveWindowS]);

  const sensors = useMemo(() => (line.machines.find((m) => m.id === id)?.sensors ?? []) as SensorField[], [line, id]);
  const series = useMemo(() => {
    const last = readings.at(-1);
    const xMax = last === undefined ? null : Date.parse(last.ts);
    return sensors.map((s) => ({
      sensor: s,
      points: readingsToSeries(readings, s, cfg),
      xMin: xMax === null ? null : xMax - cfg.liveWindowS * 1000,
      xMax,
    }));
  }, [readings, sensors, cfg]);

  return (
    <div data-testid="live-charts">
      <p className="small muted">
        {id}: last {Math.round(cfg.liveWindowS / 60)} min of event time
        {mode === "replay" ? " (paused in replay mode; see History)" : ""}. Gaps are missing values. No alarm limits are
        drawn: the backend provides none.
      </p>
      <div className="chart-grid">
        {series.map((s) => (
          <SensorChart
            key={s.sensor}
            testId={`live-chart-${s.sensor}`}
            data={{ label: s.sensor, yTitle: axisTitle(s.sensor, line.sensors[s.sensor]), points: s.points, xMin: s.xMin, xMax: s.xMax, markers }}
          />
        ))}
      </div>
    </div>
  );
}

export function LiveCharts() {
  const model = useModel();
  const id = useStore(model.selection);
  const line = useStore(model.line);
  if (id === null || line === null) return <p className="muted">Select a machine to see its live sensor charts.</p>;
  // Keyed by machine: switching machines starts with a fresh buffer.
  return <MachineLiveCharts key={id} id={id} line={line} />;
}

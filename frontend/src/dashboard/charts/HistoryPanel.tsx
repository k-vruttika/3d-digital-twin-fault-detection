/**
 * History of one sensor of the selected machine from `/history`, with the resolution chosen by
 * the visible span (raw / 1 min / 10 min buckets with a min-max band). Range presets, drag to
 * zoom, reset. Fetches are debounced and the previous one is aborted.
 */
import { useEffect, useMemo, useState } from "react";

import { ApiError } from "../../api/rest.ts";
import type { HistoryResponse, SensorField } from "../../api/types.ts";
import { useModel, useRest } from "../context.ts";
import { axisTitle, formatDateTime } from "../format.ts";
import { useStore } from "../state/store.ts";
import { BUCKET_S, bucketsToSeries, chooseResolution, readingsToSeries, type Resolution } from "./series.ts";
import { SensorChart } from "./SensorChart.tsx";

interface Range {
  from: number;
  to: number;
}

export function HistoryPanel() {
  const model = useModel();
  const rest = useRest();
  const id = useStore(model.selection);
  const line = useStore(model.line);
  const asOf = useStore(model.asOf);
  const mode = useStore(model.modeStore);
  const cfg = model.cfg;
  const sensors = useMemo(() => (line?.machines.find((m) => m.id === id)?.sensors ?? []) as SensorField[], [line, id]);
  const [sensor, setSensor] = useState<SensorField>("temp");
  const [presetS, setPresetS] = useState<number | "all">(cfg.rangePresetsS[0]?.seconds ?? 900);
  const [zoom, setZoom] = useState<Range | null>(null);
  const [allRange, setAllRange] = useState<Range | null>(null);
  const [data, setData] = useState<{ res: Resolution; body: HistoryResponse } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // The anchor is event time: the newest data (live) or the replay time, never the wall clock.
  const end = asOf === null ? null : Date.parse(asOf);
  // In live mode the anchor moves every second; round it so the range is refetched only every
  // few seconds rather than on every reading.
  const anchor = end === null ? null : mode === "live" ? Math.ceil(end / 10_000) * 10_000 : end;

  // The stored range: "all", and no preset reaches back before the first stored event.
  useEffect(() => {
    let cancelled = false;
    rest
      .getReplayRange()
      .then((r) => {
        if (!cancelled && r.start && r.end) setAllRange({ from: Date.parse(r.start), to: Date.parse(r.end) + 1 });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [rest, anchor]);

  const range: Range | null = useMemo(() => {
    if (zoom !== null) return zoom;
    if (presetS === "all") return allRange;
    if (anchor === null) return null;
    const from = anchor - presetS * 1000;
    return { from: allRange !== null ? Math.max(from, Math.min(allRange.from, anchor - 1000)) : from, to: anchor };
  }, [zoom, presetS, allRange, anchor]);

  useEffect(() => {
    if (id === null || range === null) return;
    const ctl = new AbortController();
    const res = chooseResolution((range.to - range.from) / 1000, cfg);
    const timer = setTimeout(() => {
      setLoading(true);
      rest
        .getHistory(
          {
            from: new Date(range.from).toISOString(),
            to: new Date(range.to).toISOString(),
            resolution: res,
            machines: [id],
          },
          ctl.signal,
        )
        .then((body) => {
          setData({ res, body });
          setError(null);
        })
        .catch((e: unknown) => {
          if (ctl.signal.aborted) return;
          if (e instanceof ApiError && e.status === 413) setError("Too many points for this range at this resolution; zoom in or out.");
          else setError(`History unavailable: ${String(e)}`);
        })
        .finally(() => {
          if (!ctl.signal.aborted) setLoading(false);
        });
    }, cfg.historyDebounceMs);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
  }, [id, range, rest, cfg]);

  const chart = useMemo(() => {
    if (data === null || line === null) return null;
    const yTitle = axisTitle(sensor, line.sensors[sensor]);
    if (data.res === "raw") {
      return { label: sensor, yTitle, points: readingsToSeries(data.body.readings ?? [], sensor, cfg), band: null };
    }
    const s = bucketsToSeries(data.body.buckets ?? [], sensor, BUCKET_S[data.res]);
    return { label: sensor, yTitle, points: s.mean, band: { min: s.min, max: s.max } };
  }, [data, sensor, line, cfg]);

  if (id === null || line === null) return <p className="muted">Select a machine to see its history.</p>;
  const res = range === null ? null : chooseResolution((range.to - range.from) / 1000, cfg);
  return (
    <div data-testid="history-panel">
      <div className="toolbar">
        <label>
          Sensor{" "}
          <select
            value={sensor}
            onChange={(e) => {
              setSensor(e.target.value as SensorField);
            }}
          >
            {sensors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span role="group" aria-label="Range" className="seg">
          {cfg.rangePresetsS.map((p) => (
            <button
              key={p.label}
              type="button"
              aria-pressed={zoom === null && presetS === p.seconds}
              onClick={() => {
                setPresetS(p.seconds);
                setZoom(null);
              }}
            >
              {p.label}
            </button>
          ))}
          <button
            type="button"
            aria-pressed={zoom === null && presetS === "all"}
            onClick={() => {
              setPresetS("all");
              setZoom(null);
            }}
          >
            all
          </button>
        </span>
        {zoom !== null && (
          <button
            type="button"
            onClick={() => {
              setZoom(null);
            }}
          >
            Reset zoom
          </button>
        )}
        <span className="badge tone-idle" data-testid="history-resolution">
          {res === null ? "–" : res === "raw" ? "raw" : `${res} mean, min–max band`}
        </span>
        {loading && <span className="muted small">loading…</span>}
      </div>
      {range !== null && (
        <p className="small muted">
          {formatDateTime(range.from)} → {formatDateTime(range.to)} · drag across the chart to zoom · gaps are missing values ·
          no alarm limits are drawn (the backend provides none)
        </p>
      )}
      {error !== null && (
        <p className="error small" role="alert">
          {error}
        </p>
      )}
      {chart !== null && range !== null && (
        <SensorChart
          testId="history-chart"
          height={220}
          data={{ ...chart, xMin: range.from, xMax: range.to }}
          onZoom={(from, to) => {
            setZoom({ from, to });
          }}
        />
      )}
    </div>
  );
}

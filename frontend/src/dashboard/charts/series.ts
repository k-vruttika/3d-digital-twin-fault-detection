/**
 * Readings and history buckets -> chart points (pure functions, unit tested).
 *
 * Missing data is a gap, never a zero:
 * - a missing sensor value (absent or null in a Reading, `count: 0` in a bucket) is `y: null`;
 * - a time gap longer than `gapFactor` x the series' median sample interval (raw) or a missing
 *   bucket (bucketed) gets a `y: null` point in between, so the line breaks there.
 * Charts draw with `spanGaps: false`, so a null always shows as a gap.
 */
import type { HistoryBucket, Reading, SensorField } from "../../api/types.ts";

export interface ChartPoint {
  /** epoch ms */
  x: number;
  y: number | null;
}

export type Resolution = "raw" | "1m" | "10m";

export const BUCKET_S: Record<Exclude<Resolution, "raw">, number> = { "1m": 60, "10m": 600 };

/** Resolution for a visible span: raw for short spans, then 1-minute and 10-minute buckets. */
export function chooseResolution(spanS: number, cfg: { rawMaxSpanS: number; oneMinuteMaxSpanS: number }): Resolution {
  if (spanS <= cfg.rawMaxSpanS) return "raw";
  if (spanS <= cfg.oneMinuteMaxSpanS) return "1m";
  return "10m";
}

function value(r: Reading, sensor: SensorField): number | null {
  const v = r[sensor];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Median spacing of sorted timestamps (ms), or null with fewer than two. */
export function medianIntervalMs(xs: readonly number[]): number | null {
  if (xs.length < 2) return null;
  const d: number[] = [];
  for (let i = 1; i < xs.length; i++) {
    const a = xs[i - 1];
    const b = xs[i];
    if (a !== undefined && b !== undefined && b > a) d.push(b - a);
  }
  if (d.length === 0) return null;
  d.sort((p, q) => p - q);
  return d[Math.floor(d.length / 2)] ?? null;
}

/** Insert a null point between two points further apart than `maxGapMs`. */
export function breakGaps(points: readonly ChartPoint[], maxGapMs: number): ChartPoint[] {
  const out: ChartPoint[] = [];
  let prev: ChartPoint | null = null;
  for (const p of points) {
    if (prev !== null && p.x - prev.x > maxGapMs) out.push({ x: (prev.x + p.x) / 2, y: null });
    out.push(p);
    prev = p;
  }
  return out;
}

export function readingsToSeries(
  readings: readonly Reading[],
  sensor: SensorField,
  opts: { gapFactor: number; fallbackIntervalS: number },
): ChartPoint[] {
  const pts = readings
    .map((r) => ({ x: Date.parse(r.ts), y: value(r, sensor) }))
    .filter((p) => Number.isFinite(p.x))
    .sort((a, b) => a.x - b.x);
  const interval = medianIntervalMs(pts.map((p) => p.x)) ?? opts.fallbackIntervalS * 1000;
  return breakGaps(pts, interval * opts.gapFactor);
}

export interface BucketSeries {
  mean: ChartPoint[];
  min: ChartPoint[];
  max: ChartPoint[];
}

/** Mean, min and max per bucket; empty and missing buckets are gaps. */
export function bucketsToSeries(buckets: readonly HistoryBucket[], sensor: SensorField, widthS: number): BucketSeries {
  const rows = buckets
    .map((b) => {
      const s = b.sensors[sensor];
      const has = s !== undefined && s.count > 0;
      // the bucket's centre on the time axis
      return {
        x: Date.parse(b.bucket) + (widthS * 1000) / 2,
        mean: has ? (s.mean ?? null) : null,
        min: has ? (s.min ?? null) : null,
        max: has ? (s.max ?? null) : null,
      };
    })
    .filter((r) => Number.isFinite(r.x))
    .sort((a, b) => a.x - b.x);
  const maxGap = widthS * 1000 * 1.5; // more than one bucket apart: a bucket is missing
  return {
    mean: breakGaps(rows.map((r) => ({ x: r.x, y: r.mean })), maxGap),
    min: breakGaps(rows.map((r) => ({ x: r.x, y: r.min })), maxGap),
    max: breakGaps(rows.map((r) => ({ x: r.x, y: r.max })), maxGap),
  };
}

/** Keep the readings of the last `windowS` seconds before the newest one (live ring buffer). */
export function trimWindow(readings: Reading[], windowS: number): Reading[] {
  const last = readings.at(-1);
  if (last === undefined) return readings;
  const cutoff = Date.parse(last.ts) - windowS * 1000;
  const i = readings.findIndex((r) => Date.parse(r.ts) >= cutoff);
  return i <= 0 ? readings : readings.slice(i);
}

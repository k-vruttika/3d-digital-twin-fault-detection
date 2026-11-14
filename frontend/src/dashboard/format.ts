/** Display helpers for the dashboard (pure functions). */
import type { MachineState, SensorSpec } from "../api/types.ts";
import { DEFAULT_CONFIG } from "../engine/config.ts";

/** Unit as shown to people ("degC" in the config is "°C"). */
export function displayUnit(unit: string): string {
  return unit === "degC" ? "°C" : unit;
}

/** A sensor value with its configured decimals, or an en dash when missing. */
export function formatValue(value: number | null | undefined, sensor: string): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  const decimals = (DEFAULT_CONFIG.sensorDecimals as Record<string, number | undefined>)[sensor] ?? 1;
  return value.toFixed(decimals);
}

export function axisTitle(sensor: string, spec: SensorSpec | undefined): string {
  return spec === undefined ? sensor : `${sensor} (${displayUnit(spec.unit)})`;
}

/** Badge colour family of a state (text is always shown too). */
export type Tone = "ok" | "idle" | "warning" | "fault" | "waiting" | "maintenance" | "stale" | "unknown";

export function stateTone(state: MachineState | null): Tone {
  switch (state) {
    case "RUNNING":
      return "ok";
    case "IDLE":
      return "idle";
    case "WARNING":
      return "warning";
    case "FAULT":
      return "fault";
    case "STARVED":
    case "BLOCKED":
      return "waiting";
    case "MAINTENANCE":
      return "maintenance";
    case "STALE":
    case "OFFLINE":
      return "stale";
    case null:
      return "unknown";
  }
}

/** "12:34:56" in UTC (event time is UTC throughout the system). */
export function formatTime(iso: string | number | null | undefined): string {
  if (iso === null || iso === undefined) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "–";
  return d.toISOString().slice(11, 19);
}

/** "2026-01-01 12:34:56 UTC". */
export function formatDateTime(iso: string | number | null | undefined): string {
  if (iso === null || iso === undefined) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "–";
  return `${d.toISOString().slice(0, 19).replace("T", " ")} UTC`;
}

/** "3 s", "4 min", "2 h" (an age or a duration). */
export function formatAge(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "–";
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 90) return `${String(s)} s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${String(m)} min`;
  return `${String(Math.round(m / 60))} h`;
}

export function formatPct(v: number | null | undefined): string {
  return v === null || v === undefined ? "–" : `${v.toFixed(1)} %`;
}

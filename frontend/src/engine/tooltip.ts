/** Tooltip content for machines and buffers (pure; rendered as text, never as HTML). */
import type {
  Alert,
  DataQualityReport,
  LineMachine,
  Reading,
  SensorSpec,
} from "../api/types.ts";
import type { VisualConfig } from "./config.ts";
import type { BufferStatus } from "./visuals.ts";

export const MISSING = "–";

export interface TooltipModel {
  title: string;
  subtitle: string;
  rows: { label: string; value: string }[];
  notes: string[];
}

/** A sensor value with its unit; "–" when the value is missing (null, absent or not finite). */
export function formatSensor(
  sensor: string,
  value: number | null | undefined,
  spec: SensorSpec | undefined,
  cfg: VisualConfig,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  const decimals = cfg.sensorDecimals[sensor] ?? 2;
  const unit = spec === undefined ? "" : ` ${spec.unit === "degC" ? "°C" : spec.unit}`;
  return `${value.toFixed(decimals)}${unit}`;
}

export function formatAge(fromIso: string | null | undefined, nowMs: number): string {
  if (fromIso === null || fromIso === undefined) return MISSING;
  const t = Date.parse(fromIso);
  if (!Number.isFinite(t)) return MISSING;
  const s = Math.max(0, (nowMs - t) / 1000);
  if (s < 60) return `${s.toFixed(0)} s ago`;
  if (s < 3600) return `${(s / 60).toFixed(0)} min ago`;
  return `${(s / 3600).toFixed(1)} h ago`;
}

export interface MachineTooltipInput {
  machine: LineMachine;
  stateLabel: string;
  reading: Reading | null;
  sensors: Record<string, SensorSpec>;
  dataQuality: DataQualityReport | null;
  alerts: readonly Alert[];
  /** Wall clock (ms) for the reading age; event time and wall time match for live sources. */
  nowMs: number;
}

export function machineTooltip(input: MachineTooltipInput, cfg: VisualConfig): TooltipModel {
  const { machine, reading } = input;
  const rows = machine.sensors.map((sensor) => ({
    label: sensor,
    value: formatSensor(
      sensor,
      reading === null ? null : (reading as unknown as Record<string, number | null | undefined>)[sensor],
      input.sensors[sensor],
      cfg,
    ),
  }));
  rows.push({ label: "quality", value: reading?.quality ?? MISSING });
  rows.push({ label: "reading", value: formatAge(reading?.ts, input.nowMs) });
  const notes: string[] = [];
  const dq = input.dataQuality;
  if (dq !== null && dq.status === "DEGRADED") {
    const reasons = dq.reasons ?? [];
    notes.push(`Data quality DEGRADED${reasons.length > 0 ? `: ${reasons.join(", ")}` : ""}`);
  }
  for (const a of input.alerts) {
    if (!a.active) continue;
    notes.push(`${a.severity}: ${a.message}`);
  }
  return {
    title: `${machine.id} · ${machine.name || machine.kind}`,
    subtitle: input.stateLabel,
    rows,
    notes,
  };
}

export function bufferTooltip(
  buffer: { id: string; from: string; to: string; capacity: number },
  status: { status: BufferStatus; why: string },
): TooltipModel {
  return {
    title: `Buffer ${buffer.id} · ${buffer.from} → ${buffer.to}`,
    subtitle: status.status,
    rows: [{ label: "capacity", value: `${String(buffer.capacity)} parts` }],
    notes: [status.why, "Inferred from the neighbours' states; the line reports no buffer level."],
  };
}

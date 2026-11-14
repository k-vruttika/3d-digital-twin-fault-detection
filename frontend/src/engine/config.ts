/**
 * Every visual tuning value of the 3D twin, in one place (CLAUDE.md: no magic constants).
 *
 * These are presentation settings (sizes, colours, animation rates), not domain thresholds:
 * machine states, data quality and alerts always come from the backend. Units: metres, seconds,
 * RPM. Colours are 0xRRGGBB.
 */
import type { MachineKind, MachineState } from "../api/types.ts";
import type { LabelSizing } from "./labelPolicy.ts";
import type { BufferStatus } from "./visuals.ts";

export interface Footprint {
  /** along the material flow */
  length: number;
  /** across the flow */
  width: number;
}

export interface LayoutConfig {
  /** Machines per row before the line turns (serpentine). */
  rowLength: number;
  /** Distance between row centre lines. */
  rowSpacing: number;
  /** Gap between neighbouring machines along a row (buffers sit in it). */
  gap: number;
  footprints: Record<MachineKind, Footprint>;
  defaultFootprint: Footprint;
}

export interface VisualConfig {
  layout: LayoutConfig;
  /** The animation runs at speed / reference RPM (1 = nominal), clamped to maxRate. */
  referenceRpm: Record<MachineKind, number>;
  defaultReferenceRpm: number;
  maxRate: number;
  /** How fast the shown rate follows the reported speed (1/s, exponential smoothing). */
  rateResponse: number;
  /** Belt surface speed at rate 1 (m/s). */
  beltSpeed: number;
  /** Visual turns per second at rate 1 (not the real RPM, which would only blur). */
  spindleTurnsPerS: number;
  indexerTurnsPerS: number;
  /** Camera-head sweeps per second at rate 1. */
  cameraSweepHz: number;
  conveyor: {
    beltWidth: number;
    height: number;
    rollerPitch: number;
    itemSpacing: number;
    itemSize: number;
  };
  buffer: { maxStack: number; boxSize: number; perLayer: number };
  /** Transfer track where a row turns into the next (metres). */
  transfer: { width: number; height: number; railWidth: number; chevronSpacing: number };
  colors: {
    background: number;
    replayBackground: number;
    floor: number;
    grid: number;
    frame: number;
    body: number;
    accent: number;
    belt: number;
    beltStripe: number;
    glass: number;
    item: number;
    neutral: number;
    running: number;
    warning: number;
    fault: number;
    waiting: number;
    maintenance: number;
    offline: number;
    dataQuality: number;
    highlight: number;
    selection: number;
    alert: { INFO: number; WARNING: number; CRITICAL: number };
    label: { text: number; background: number };
  };
  /** Opacity of a STALE (or link-lost) machine. */
  staleOpacity: number;
  /** How far an OFFLINE machine is desaturated (0..1). */
  offlineGrey: number;
  /** How strongly a MAINTENANCE machine is tinted blue (0..1). */
  maintenanceTint: number;
  /** FAULT part flash frequency (Hz). */
  flashHz: number;
  /** Outline thickness (m) of the at-risk part. */
  outlineThickness: number;
  labels: LabelSizing & {
    enabled: boolean;
    /** Texture font size (px); the on-screen size comes from pixelHeight. */
    fontPx: number;
    /** Machine states whose label is hidden unless hovered, selected or mode "all". */
    normalStates: MachineState[];
    /** Buffer statuses whose label is shown by default. */
    bufferStatuses: BufferStatus[];
  };
  /** Decimals shown per sensor in tooltips. */
  sensorDecimals: Record<string, number>;
  render: {
    maxPixelRatio: number;
    /** Shadows only for lines up to this many machines. */
    shadowMaxMachines: number;
    fov: number;
  };
}

export const DEFAULT_CONFIG: VisualConfig = {
  layout: {
    rowLength: 6,
    rowSpacing: 7,
    gap: 1.8,
    footprints: {
      CONVEYOR: { length: 4, width: 1.3 },
      CNC: { length: 2.6, width: 2.4 },
      INSPECTION: { length: 2.4, width: 2.2 },
    },
    defaultFootprint: { length: 2, width: 2 },
  },
  // Nominal set speeds of the simulated machines (config/simulator.yaml speed_set_rpm).
  referenceRpm: { CONVEYOR: 1450, CNC: 8000, INSPECTION: 600 },
  defaultReferenceRpm: 1000,
  maxRate: 2,
  rateResponse: 4,
  beltSpeed: 0.6,
  spindleTurnsPerS: 6,
  indexerTurnsPerS: 0.4,
  cameraSweepHz: 0.25,
  conveyor: { beltWidth: 0.8, height: 0.85, rollerPitch: 0.25, itemSpacing: 0.9, itemSize: 0.28 },
  buffer: { maxStack: 12, boxSize: 0.24, perLayer: 6 },
  transfer: { width: 0.7, height: 0.04, railWidth: 0.05, chevronSpacing: 0.9 },
  colors: {
    background: 0x1d2229,
    replayBackground: 0x2a2620,
    floor: 0x2b3038,
    grid: 0x3a414c,
    frame: 0x6b7480,
    body: 0xb8c0c8,
    accent: 0x4d5866,
    belt: 0x3d444c,
    beltStripe: 0x5f6873,
    glass: 0x9ec9e6,
    item: 0xc9a46a,
    neutral: 0x9aa4ae,
    running: 0x43a047,
    warning: 0xff8c1a,
    fault: 0xe53935,
    waiting: 0xf5c518,
    maintenance: 0x2f7de1,
    offline: 0x6e7378,
    dataQuality: 0xb39ddb,
    highlight: 0x40c4ff,
    selection: 0xffffff,
    alert: { INFO: 0x4fc3f7, WARNING: 0xffb300, CRITICAL: 0xff3d00 },
    label: { text: 0xffffff, background: 0x14181d },
  },
  staleOpacity: 0.35,
  offlineGrey: 0.9,
  maintenanceTint: 0.45,
  flashHz: 2,
  outlineThickness: 0.035,
  labels: {
    enabled: true,
    fontPx: 44,
    pixelHeight: 22,
    minPixelHeight: 14,
    referenceViewportPx: 720,
    fadeStart: 45,
    hideDistance: 80,
    normalStates: ["RUNNING", "IDLE"],
    bufferStatuses: ["FULL", "EMPTY", "UNKNOWN"],
  },
  sensorDecimals: { speed: 0, temp: 1, vib: 2, load: 1, current: 1 },
  render: { maxPixelRatio: 2, shadowMaxMachines: 8, fov: 45 },
};

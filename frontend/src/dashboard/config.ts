/**
 * Dashboard tuning values (CLAUDE.md: no magic constants). Presentation and polling settings
 * only; every domain threshold (states, alerts, data quality, command limits) comes from the
 * backend.
 */
export interface DashboardConfig {
  /** Not live for longer than this after having been live: "link lost" instead of "reconnecting". */
  linkLostAfterMs: number;
  /** How often the connection banner re-evaluates while not live. */
  connectionTickMs: number;
  /** Live chart window (event-time seconds before the newest reading). */
  liveWindowS: number;
  /** A gap longer than this many median sample intervals breaks a chart line. */
  gapFactor: number;
  /** Sample interval assumed when a series has too few points to measure it. */
  fallbackIntervalS: number;
  /** /history resolution by visible span: raw up to rawMaxSpanS, 1 min up to oneMinuteMaxSpanS. */
  rawMaxSpanS: number;
  oneMinuteMaxSpanS: number;
  /** History range presets (seconds); "all" uses the stored range. */
  rangePresetsS: { label: string; seconds: number }[];
  /** Wait this long after the last zoom / range change before fetching history. */
  historyDebounceMs: number;
  /** GET /sources poll interval. */
  sourcesPollMs: number;
  /** Recent commands / scenario requests: this far back (event time). */
  auditLookbackS: number;
  /** Replay: one step every replayTickMs advances event time by speed x replayTickMs. */
  replayTickMs: number;
  replaySpeeds: number[];
  /** GET /predict poll interval (inference state, per-machine gate) in live mode. */
  predictPollMs: number;
  /** Prediction horizon the alarm and the highlight use, until /predict says otherwise. */
  primaryHorizonS: number;
  /** Older predictions are not shown, until /predict reports the backend's value. */
  predictionStaleMsFallback: number;
  /** At most this many parts are highlighted from one prediction's attributions. */
  highlightTopParts: number;
  /** Unusual-behaviour display (UI only): "intermittent" after this many on/off switches ... */
  unusualFlapMinSwitches: number;
  /** ... within this period (event-time seconds), and the indicator stays this long after a clear. */
  unusualFlapWindowS: number;
  unusualHoldS: number;
}

export const DASHBOARD_CONFIG: DashboardConfig = {
  linkLostAfterMs: 5_000,
  connectionTickMs: 500,
  liveWindowS: 300,
  gapFactor: 3,
  fallbackIntervalS: 1,
  rawMaxSpanS: 30 * 60,
  oneMinuteMaxSpanS: 24 * 3600,
  rangePresetsS: [
    { label: "15 min", seconds: 15 * 60 },
    { label: "1 h", seconds: 3600 },
    { label: "6 h", seconds: 6 * 3600 },
    { label: "24 h", seconds: 24 * 3600 },
  ],
  historyDebounceMs: 300,
  sourcesPollMs: 5_000,
  auditLookbackS: 3600,
  replayTickMs: 500,
  replaySpeeds: [1, 5, 20],
  predictPollMs: 2_000,
  primaryHorizonS: 30,
  predictionStaleMsFallback: 5_000,
  highlightTopParts: 2,
  unusualFlapMinSwitches: 4,
  unusualFlapWindowS: 300,
  unusualHoldS: 30,
};

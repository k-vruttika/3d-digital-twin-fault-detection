/**
 * The 3D scene's look for the dashboard (DESIGN.md §10), applied through the engine's public
 * `config` option, never by editing the engine. Status colours mirror the dark fills in
 * tokens.css (the scene is lit, so it uses the saturated values in both themes):
 * purple only for "unusual behaviour, cause unknown"; data quality uses the degraded colour;
 * STARVED / BLOCKED / OFFLINE use the inactive grey; the selection uses the bone accent.
 */
import type { VisualConfig } from "../engine/config.ts";

export const SCENE_COLORS = {
  frame: 0x59616a,
  body: 0x9ea6ae,
  accent: 0x3a4149,
  belt: 0x2b3036,
  beltStripe: 0x4a525a,
  glass: 0x8fb4c9,
  item: 0xb89a6a,
  neutral: 0x8e97a1,
  running: 0x6fc28e,
  warning: 0xf4a42c,
  fault: 0xff5b4f,
  waiting: 0xa0a8b1,
  maintenance: 0x6e9bff,
  offline: 0xa0a8b1,
  dataQuality: 0xe6dc5a,
  highlight: 0x3fe0e8,
  unusual: 0xc58aff,
  selection: 0xe8dfc8,
  alert: { INFO: 0x6e9bff, WARNING: 0xf4a42c, CRITICAL: 0xff5b4f },
  label: { text: 0xeceef0, background: 0x0d0f12 },
} as const satisfies Partial<VisualConfig["colors"]>;

/** The engine defaults with the dashboard's palette, a calmer fault blink and a flatter lens. */
export function sceneConfig(base: VisualConfig): VisualConfig {
  return {
    ...base,
    colors: { ...base.colors, ...SCENE_COLORS, alert: { ...SCENE_COLORS.alert }, label: { ...SCENE_COLORS.label } },
    flashHz: 1.25,
    render: { ...base.render, fov: 38 },
  };
}

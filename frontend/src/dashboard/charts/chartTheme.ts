/**
 * Chart colours and fonts read from the design tokens at runtime (DESIGN.md §8), so charts
 * never carry their own hex values. Canvas cannot resolve CSS variables, so the resolved
 * values are passed to Chart.js. Where no stylesheet is loaded (unit tests with jsdom) the
 * fallbacks are plain CSS colour names; such charts are never seen.
 */
export interface ChartTheme {
  /** measured trend line */
  line: string;
  /** min-max band of bucketed history */
  band: string;
  /** horizontal gridlines */
  grid: string;
  /** tick labels */
  tick: string;
  /** axis titles and legend */
  title: string;
  tooltipBg: string;
  tooltipBorder: string;
  tooltipText: string;
  /** what-if: projected, no change */
  baseline: string;
  /** what-if: projected, with the change (the accent) */
  scenario: string;
  fontMono: string;
  fontBody: string;
  fontDisplay: string;
}

function token(style: CSSStyleDeclaration | null, name: string, fallback: string): string {
  const v = style?.getPropertyValue(name).trim() ?? "";
  return v === "" ? fallback : v;
}

/** "#RRGGBB" at an opacity, as rgba(); anything else is returned unchanged. */
export function withAlpha(color: string, alpha: number): string {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(color);
  if (m === null) return color;
  const [r, g, b] = [m[1], m[2], m[3]].map((h) => parseInt(h ?? "0", 16));
  return `rgba(${String(r)}, ${String(g)}, ${String(b)}, ${String(alpha)})`;
}

/**
 * The chart theme for the current document. Callers re-read it whenever the theme changes
 * (theme.ts sets data-theme before notifying subscribers, so the new tokens are in place).
 */
export function chartTheme(): ChartTheme {
  let style: CSSStyleDeclaration | null;
  try {
    style = typeof getComputedStyle === "function" && typeof document !== "undefined" ? getComputedStyle(document.documentElement) : null;
  } catch {
    style = null;
  }
  const text1 = token(style, "--text-1", "gray");
  return {
    line: withAlpha(text1, 0.85),
    band: withAlpha(text1, 0.1),
    grid: token(style, "--hairline", "lightgray"),
    tick: token(style, "--text-3", "gray"),
    title: token(style, "--text-2", "gray"),
    tooltipBg: token(style, "--surface-2", "black"),
    tooltipBorder: token(style, "--hairline-strong", "gray"),
    tooltipText: text1,
    baseline: token(style, "--text-2", "gray"),
    scenario: token(style, "--accent", "gray"),
    fontMono: token(style, "--font-mono", '"IBM Plex Mono"'),
    fontBody: token(style, "--font-body", '"IBM Plex Sans"'),
    fontDisplay: token(style, "--font-display", '"Barlow Condensed"'),
  };
}

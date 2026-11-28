#!/usr/bin/env node
/**
 * Contrast check for the design tokens (DESIGN.md §3.5). Zero dependencies.
 *
 * Reads src/dashboard/tokens.css, resolves the dark (:root) and light (:root[data-theme="light"])
 * themes, recomputes the composites the browser draws (a status tint = color-mix in sRGB into
 * --surface-1; the HUD glass = --hud alpha-blended over the scene), and asserts:
 *
 *   text      --text-1/2/3, --accent, every --st-*-ink   on --surface-0..3        >= 4.5
 *   tint      every --st-*-ink on its own tint                                    >= 4.5
 *   fill      --on-status on every --st-* fill, --on-accent on --accent           >= 4.5
 *   non-text  every --st-* fill against --surface-1 and --surface-3                >= 3.0
 *   HUD       --text-1, --text-2, --accent, every --st-*-ink on the glass over
 *             --surface-0 and over the light machine body --scene-body            >= 4.5
 *   distinct  minimum OKLab distance between the status fills and the accent      >= 0.11 dark, 0.09 light
 *
 * Exit code 1 with a list of failures, 0 otherwise. Run by `npm test` (package.json).
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const TOKENS = resolve(here, "../src/dashboard/tokens.css");

const STATUS = ["running", "inactive", "warning", "fault", "maintenance", "unusual", "degraded", "explained"];
const SURFACES = ["--surface-0", "--surface-1", "--surface-2", "--surface-3"];
const AA = 4.5;
const NON_TEXT = 3.0;
const MIN_DISTANCE = { dark: 0.11, light: 0.09 };

// ------------------------------------------------------------------ parsing
function block(css, selector) {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`tokens.css: no "${selector} {" block`);
  const end = css.indexOf("}", start);
  const vars = {};
  for (const m of css.slice(start, end).matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) vars[m[1]] = m[2].trim();
  return vars;
}

/** "#RRGGBB" or "rgb(r g b / a)" -> { rgb: [0..1]*3, a } */
function color(value, name) {
  const hex = /^#([0-9a-f]{6})$/i.exec(value);
  if (hex) return { rgb: [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16) / 255), a: 1 };
  const rgb = /^rgb\(\s*(\d+)\s+(\d+)\s+(\d+)\s*(?:\/\s*([\d.]+))?\s*\)$/.exec(value);
  if (rgb) return { rgb: [rgb[1], rgb[2], rgb[3]].map((v) => Number(v) / 255), a: rgb[4] === undefined ? 1 : Number(rgb[4]) };
  throw new Error(`tokens.css: ${name} = "${value}" is not #RRGGBB or rgb(r g b / a)`);
}

// ------------------------------------------------------------------ colour maths
const lin = (v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const luminance = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
function ratio(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
/** color-mix(in srgb, top p, bottom) and alpha blending are the same sRGB lerp. */
const mix = (top, p, bottom) => top.map((t, i) => t * p + bottom[i] * (1 - p));
function oklab(rgb) {
  const [r, g, b] = rgb.map(lin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}
const distance = (a, b) => Math.hypot(...oklab(a).map((v, i) => v - oklab(b)[i]));
const hex = (rgb) => `#${rgb.map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("").toUpperCase()}`;

// ------------------------------------------------------------------ checks
function check(theme, vars) {
  const c = (name) => {
    if (!(name in vars)) throw new Error(`tokens.css (${theme}): ${name} is not defined`);
    return color(vars[name], name).rgb;
  };
  const results = [];
  const expect = (label, fg, bg, min) => {
    results.push({ theme, label, ratio: ratio(fg, bg), min });
  };

  const textTokens = ["--text-1", "--text-2", "--text-3", "--accent", ...STATUS.map((s) => `--st-${s}-ink`)];
  for (const t of textTokens) for (const s of SURFACES) expect(`${t} on ${s}`, c(t), c(s), AA);

  const tintPct = Number.parseFloat(vars["--tint"]) / 100;
  if (!(tintPct > 0 && tintPct < 1)) throw new Error(`tokens.css (${theme}): --tint must be a percentage`);
  for (const s of STATUS) {
    const tint = mix(c(`--st-${s}`), tintPct, c("--surface-1"));
    expect(`--st-${s}-ink on its tint (${hex(tint)})`, c(`--st-${s}-ink`), tint, AA);
    expect(`--on-status on --st-${s}`, c("--on-status"), c(`--st-${s}`), AA);
    for (const surf of ["--surface-1", "--surface-3"]) expect(`--st-${s} fill vs ${surf}`, c(`--st-${s}`), c(surf), NON_TEXT);
  }
  expect("--on-accent on --accent", c("--on-accent"), c("--accent"), AA);

  const hud = color(vars["--hud"], "--hud");
  for (const under of ["--surface-0", "--scene-body"]) {
    const glass = mix(hud.rgb, hud.a, c(under));
    for (const t of ["--text-1", "--text-2", "--accent", ...STATUS.map((s) => `--st-${s}-ink`)]) {
      expect(`${t} on HUD over ${under} (${hex(glass)})`, c(t), glass, AA);
    }
  }

  const fills = [...STATUS.map((s) => `--st-${s}`), "--accent"];
  let closest = { d: Infinity, a: "", b: "" };
  for (let i = 0; i < fills.length; i++) {
    for (let j = i + 1; j < fills.length; j++) {
      const d = distance(c(fills[i]), c(fills[j]));
      if (d < closest.d) closest = { d, a: fills[i], b: fills[j] };
    }
  }
  return { results, closest };
}

const css = readFileSync(TOKENS, "utf8");
const dark = block(css, ":root");
const themes = { dark, light: { ...dark, ...block(css, ':root[data-theme="light"]') } };

let failures = 0;
let total = 0;
for (const [theme, vars] of Object.entries(themes)) {
  const { results, closest } = check(theme, vars);
  total += results.length + 1;
  const bad = results.filter((r) => r.ratio < r.min);
  const worst = results.reduce((w, r) => (r.ratio / r.min < w.ratio / w.min ? r : w));
  for (const r of bad) console.error(`FAIL ${theme}: ${r.label} = ${r.ratio.toFixed(2)} (needs ${r.min})`);
  const distinctOk = closest.d >= MIN_DISTANCE[theme];
  if (!distinctOk) {
    console.error(`FAIL ${theme}: ${closest.a} / ${closest.b} are only ${closest.d.toFixed(3)} apart (OKLab; needs ${MIN_DISTANCE[theme]})`);
  }
  failures += bad.length + (distinctOk ? 0 : 1);
  console.log(
    `${theme}: ${String(results.length - bad.length)}/${String(results.length)} ratios pass; tightest ${worst.label} = ${worst.ratio.toFixed(2)}; ` +
      `closest status pair ${closest.a} / ${closest.b} = ${closest.d.toFixed(3)}`,
  );
}
if (failures > 0) {
  console.error(`contrast check: ${String(failures)} of ${String(total)} checks failed (DESIGN.md §3.5)`);
  process.exit(1);
}
console.log(`contrast check: all ${String(total)} checks pass (DESIGN.md §3.5)`);

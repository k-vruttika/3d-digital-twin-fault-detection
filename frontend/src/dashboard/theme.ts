/**
 * Light / dark theme. Defaults to the system preference; an explicit choice is kept in
 * localStorage (a display preference, never anything secret). CSS reads `data-theme` on <html>.
 */
import { createStore, useStore } from "./state/store.ts";

export type Theme = "light" | "dark";

const THEME_KEY = "linetwin.theme";

function systemTheme(): Theme {
  try {
    return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function savedTheme(): Theme | null {
  try {
    const v = typeof localStorage === "undefined" ? null : localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export const themeStore = createStore<Theme>(savedTheme() ?? systemTheme());

export function setTheme(theme: Theme): void {
  themeStore.set(theme);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // storage disabled: the choice lasts for this page only
  }
}

export function useTheme(): Theme {
  return useStore(themeStore);
}

/**
 * 3D scene colours per theme (engine.setTheme). In the light theme the replay background equals
 * the normal one: a warm tint over a light scene looks muddy, so replay there is shown by the
 * REPLAY banner and an amber frame with a soft vignette around the 3D view (styles.css).
 */
export const ENGINE_THEME: Record<Theme, { background: number; replayBackground: number; floor: number; grid: number }> = {
  dark: { background: 0x1d2229, replayBackground: 0x2a2620, floor: 0x2b3038, grid: 0x3a414c },
  light: { background: 0xdfe4ea, replayBackground: 0xdfe4ea, floor: 0xc9d0d8, grid: 0xb3bcc7 },
};

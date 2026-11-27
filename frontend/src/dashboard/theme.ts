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
  // DESIGN.md §10: a stage slightly lighter than the UI's --surface-0 so the docked panels separate.
  dark: { background: 0x111418, replayBackground: 0x1a1712, floor: 0x171b20, grid: 0x222830 },
  light: { background: 0xe6e4de, replayBackground: 0xe6e4de, floor: 0xd8d5cd, grid: 0xc4bfb3 },
};

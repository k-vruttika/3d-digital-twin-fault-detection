import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../engine/config.ts";
import { SCENE_COLORS, sceneConfig } from "./sceneConfig.ts";

describe("sceneConfig (DESIGN.md §10)", () => {
  const cfg = sceneConfig(DEFAULT_CONFIG);

  it("reserves the unusual-behaviour purple for that one meaning", () => {
    const { unusual, alert, label, ...rest } = cfg.colors;
    expect(unusual).toBe(SCENE_COLORS.unusual);
    for (const [name, value] of Object.entries(rest)) expect(value, name).not.toBe(unusual);
    expect(Object.values(alert)).not.toContain(unusual);
    expect(Object.values(label)).not.toContain(unusual);
  });

  it("no longer draws data quality in lavender", () => {
    expect(cfg.colors.dataQuality).not.toBe(DEFAULT_CONFIG.colors.dataQuality);
    expect(cfg.colors.dataQuality).toBe(SCENE_COLORS.dataQuality);
  });

  it("keeps everything else from the engine defaults and does not mutate them", () => {
    expect(cfg.layout).toBe(DEFAULT_CONFIG.layout);
    expect(cfg.labels).toBe(DEFAULT_CONFIG.labels);
    expect(cfg.render.maxPixelRatio).toBe(DEFAULT_CONFIG.render.maxPixelRatio);
    expect(cfg.render.fov).toBe(38);
    expect(cfg.flashHz).toBe(1.25);
    expect(DEFAULT_CONFIG.colors.unusual).not.toBe(SCENE_COLORS.unusual);
  });

  it("opts in to the scene realism; the engine default stays without it", () => {
    expect(cfg.scene?.lighting).toBeDefined();
    expect(cfg.scene?.softShadows).toBeDefined();
    expect(cfg.scene?.environment).toBeDefined();
    expect(cfg.scene?.fog).toBeDefined();
    expect(DEFAULT_CONFIG.scene).toBeUndefined();
  });
});

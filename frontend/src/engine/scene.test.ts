/** Optional scene realism (config.scene, DESIGN.md §10): every part opt-in, the default unchanged. */
import { DirectionalLight, HemisphereLight, Vector3, type Object3D } from "three";
import { describe, expect, it } from "vitest";

import { createTestEngine, LINE } from "../test/fakes.ts";
import { DEFAULT_CONFIG, type SceneConfig, type VisualConfig } from "./config.ts";

const RIG: NonNullable<SceneConfig["lighting"]> = {
  hemi: { sky: 0xdfe6f0, ground: 0x1a1d22, intensity: 0.6 },
  key: { color: 0xffe2c4, intensity: 2.6, direction: [-0.5, 1, 0.6] },
  fill: { color: 0xbcd2ff, intensity: 0.7, direction: [0.8, 0.5, 0.4] },
  rim: { color: 0xffffff, intensity: 1.1, direction: [0.2, 0.8, -1] },
};

function withScene(scene: SceneConfig): VisualConfig {
  return { ...DEFAULT_CONFIG, scene };
}

function lightsOf(root: Object3D) {
  const dir: DirectionalLight[] = [];
  const hemi: HemisphereLight[] = [];
  root.traverse((o) => {
    if (o instanceof DirectionalLight) dir.push(o);
    if (o instanceof HemisphereLight) hemi.push(o);
  });
  return { dir, hemi };
}

describe("default scene (no config.scene)", () => {
  it("keeps the original hemisphere + one white sun, no fog, no environment", () => {
    const { engine } = createTestEngine();
    engine.loadLayout(LINE);
    const { dir, hemi } = lightsOf(engine.scene);
    expect(hemi).toHaveLength(1);
    expect(hemi[0]?.intensity).toBe(1.6);
    expect(dir).toHaveLength(1);
    expect(dir[0]?.color.getHex()).toBe(0xffffff);
    expect(dir[0]?.intensity).toBe(2.2);
    expect(dir[0]?.castShadow).toBe(true);
    expect(engine.scene.fog).toBeNull();
    expect(engine.scene.environment).toBeNull();
    engine.dispose();
  });
});

describe("lighting rig", () => {
  it("builds a warm key (shadows), cool fill and rim, each on its direction from the line centre", () => {
    const { engine } = createTestEngine({ config: withScene({ lighting: RIG }) });
    engine.loadLayout(LINE);
    const { dir, hemi } = lightsOf(engine.scene);
    expect(hemi[0]?.color.getHex()).toBe(RIG.hemi.sky);
    expect(hemi[0]?.groundColor.getHex()).toBe(RIG.hemi.ground);
    expect(dir).toHaveLength(3);
    const [key, fill, rim] = dir;
    expect(key?.color.getHex()).toBe(RIG.key.color);
    expect(key?.castShadow).toBe(true);
    expect(fill?.castShadow).toBe(false);
    expect(rim?.castShadow).toBe(false);
    for (const [light, spec] of [
      [key, RIG.key],
      [fill, RIG.fill],
      [rim, RIG.rim],
    ] as const) {
      if (light === undefined) throw new Error("missing rig light");
      const toLight = light.position.clone().sub(light.target.position).normalize();
      const want = new Vector3(...spec.direction).normalize();
      expect(toLight.distanceTo(want)).toBeLessThan(0.15); // targets sit on the floor, the rig points at its centre
    }
    engine.dispose();
  });

  it("releases every light on dispose", () => {
    const { engine } = createTestEngine({ config: withScene({ lighting: RIG }) });
    engine.loadLayout(LINE);
    engine.dispose();
    expect(engine.resourceCounts().other).toBe(0);
  });
});

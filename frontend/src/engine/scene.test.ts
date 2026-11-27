/** Optional scene realism (config.scene, DESIGN.md §10): every part opt-in, the default unchanged. */
import { DataTexture, DirectionalLight, Fog, HemisphereLight, RGBAFormat, Vector3, type Object3D } from "three";
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

describe("soft shadows", () => {
  const SOFT: NonNullable<SceneConfig["softShadows"]> = { radius: 4, mapSize: 4096, intensity: 0.75 };

  function shadowCamOf(config?: VisualConfig) {
    const { engine } = createTestEngine(config === undefined ? {} : { config });
    engine.loadLayout(LINE);
    const key = lightsOf(engine.scene).dir[0];
    if (key === undefined) throw new Error("no key light");
    return { engine, key, cam: key.shadow.camera };
  }

  it("blurs and lightens the key light's shadow", () => {
    const { engine, key } = shadowCamOf(withScene({ softShadows: SOFT }));
    expect(key.shadow.radius).toBe(4);
    expect(key.shadow.mapSize.x).toBe(4096);
    expect(key.shadow.intensity).toBe(0.75);
    engine.dispose();
  });

  it("fits the shadow camera tighter than the original square, still around every machine", () => {
    const base = shadowCamOf();
    const soft = shadowCamOf(withScene({ softShadows: SOFT }));
    const area = (c: typeof base.cam) => (c.right - c.left) * (c.top - c.bottom);
    expect(area(soft.cam)).toBeLessThan(area(base.cam));
    soft.cam.updateMatrixWorld(true);
    for (const id of LINE.machines.map((m) => m.id)) {
      const root = soft.engine.machineModel(id)?.root;
      if (root === undefined) throw new Error(`no model for ${id}`);
      const p = root.getWorldPosition(new Vector3()).applyMatrix4(soft.cam.matrixWorldInverse);
      expect(p.x).toBeGreaterThan(soft.cam.left);
      expect(p.x).toBeLessThan(soft.cam.right);
      expect(p.y).toBeGreaterThan(soft.cam.bottom);
      expect(p.y).toBeLessThan(soft.cam.top);
    }
    base.engine.dispose();
    soft.engine.dispose();
  });
});

describe("environment reflections", () => {
  it("uses the reflection map with the configured intensity and releases it on dispose", () => {
    const texture = new DataTexture(new Uint8Array(4), 1, 1, RGBAFormat);
    let disposed = false;
    texture.addEventListener("dispose", () => {
      disposed = true;
    });
    const { engine } = createTestEngine({ config: withScene({ environment: { intensity: 0.4 } }), createEnvironment: () => texture });
    expect(engine.scene.environment).toBe(texture);
    expect(engine.scene.environmentIntensity).toBe(0.4);
    engine.dispose();
    expect(disposed).toBe(true);
    expect(engine.scene.environment).toBeNull();
  });

  it("skips reflections when the renderer is not WebGL (the default needs a real GPU context)", () => {
    const { engine } = createTestEngine({ config: withScene({ environment: { intensity: 0.4 } }) });
    expect(engine.scene.environment).toBeNull();
    engine.dispose();
  });
});

describe("fog", () => {
  it("fades in the background colour, scaled to the framing distance, following theme and replay", () => {
    const { engine } = createTestEngine({ config: withScene({ fog: { near: 1.2, far: 3 } }) });
    engine.loadLayout(LINE);
    const fog = engine.scene.fog;
    if (!(fog instanceof Fog)) throw new Error("no fog");
    // near / far are the configured multiples of one framing distance, which is camera-scale
    const framing = fog.far / 3;
    expect(fog.near).toBeCloseTo(framing * 1.2, 5);
    const camDist = engine.camera.position.length();
    expect(framing).toBeGreaterThan(camDist * 0.5);
    expect(framing).toBeLessThan(camDist * 2);
    expect(fog.color.getHex()).toBe(DEFAULT_CONFIG.colors.background);
    engine.setTheme({ background: 0x123456, replayBackground: 0x654321 });
    expect((engine.scene.fog as Fog).color.getHex()).toBe(0x123456);
    engine.setMode("replay");
    expect((engine.scene.fog as Fog).color.getHex()).toBe(0x654321);
    engine.dispose();
  });
});

describe("view insets", () => {
  /** Container pixel of a machine's base (fake container: 800 x 600). */
  function screenOf(engine: ReturnType<typeof createTestEngine>["engine"], id: string, w = 800, h = 600) {
    const root = engine.machineModel(id)?.root;
    if (root === undefined) throw new Error(`no model for ${id}`);
    engine.camera.updateMatrixWorld(true);
    const p = root.getWorldPosition(new Vector3()).project(engine.camera);
    return { x: ((p.x + 1) / 2) * w, y: ((1 - p.y) / 2) * h };
  }

  it("frames the whole line inside the rectangle the HUD leaves free", () => {
    const { engine } = createTestEngine();
    engine.loadLayout(LINE);
    const insets = { top: 120, right: 0, bottom: 150, left: 0 };
    engine.setViewInsets(insets);
    for (const m of LINE.machines) {
      const s = screenOf(engine, m.id);
      expect(s.y).toBeGreaterThan(insets.top);
      expect(s.y).toBeLessThan(600 - insets.bottom);
      expect(s.x).toBeGreaterThan(0);
      expect(s.x).toBeLessThan(800);
    }
    expect(engine.camera.view?.enabled).toBe(true);
    expect(engine.camera.view?.offsetY).toBe((150 - 120) / 2);
    engine.dispose();
  });

  it("keeps the offset through a resize and clears it with all-zero insets", () => {
    const { engine } = createTestEngine();
    engine.loadLayout(LINE);
    const before = engine.camera.position.clone();
    engine.setViewInsets({ top: 100, right: 40, bottom: 100, left: 0 });
    engine.resize(1000, 500);
    expect(engine.camera.view?.fullWidth).toBe(1000);
    expect(engine.camera.view?.offsetX).toBe(20);
    engine.resize(800, 600);
    engine.setViewInsets({ top: 0, right: 0, bottom: 0, left: 0 });
    expect(engine.camera.view?.enabled ?? false).toBe(false);
    expect(engine.camera.position.distanceTo(before)).toBeLessThan(1e-9);
    engine.dispose();
  });

  it("still picks the machine under the pointer", () => {
    const { engine } = createTestEngine();
    engine.loadLayout(LINE);
    engine.setViewInsets({ top: 140, right: 0, bottom: 160, left: 0 });
    const s = screenOf(engine, "CNC-01");
    const hit = engine.pickAt(s.x, s.y - 20); // a little above the base: the machine body
    expect(hit?.kind === "machine" ? hit.id : null).toBe("CNC-01");
    engine.dispose();
  });
});

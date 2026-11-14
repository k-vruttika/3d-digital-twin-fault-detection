/** Picking, hover/select events, highlightParts, setMode, FPS counter and dispose() (Part B). */
import type { Material, Mesh, Texture } from "three";
import { describe, expect, it } from "vitest";

import { createTestEngine, LINE } from "../test/fakes.ts";
import { FpsCounter } from "./fps.ts";
import { generateLine, STATE_SCENE, stateSceneLine, SyntheticReadings } from "./synthetic.ts";
import type { HoverEvent, SelectEvent } from "./TwinEngine.ts";

function readyEngine() {
  const t = createTestEngine({ autoStart: false });
  t.engine.loadLayout(LINE);
  t.engine.tick((t.clock.now = 0));
  return t;
}

describe("picking", () => {
  it("returns the machine under its screen position, and nothing on empty floor", () => {
    const { engine } = readyEngine();
    for (const id of ["CNV-01", "CNC-01", "INSP-01", "CNV-02"]) {
      const p = engine.screenPosition(id);
      if (p === null) throw new Error(id);
      expect(engine.pickAt(p.x, p.y), id).toEqual({ kind: "machine", id });
    }
    expect(engine.pickAt(2, 2)).toBeNull(); // sky corner
    engine.dispose();
  });

  it("picks buffers too", () => {
    const { engine } = readyEngine();
    const b = engine.bufferModel("B2");
    if (b === undefined) throw new Error("no B2");
    engine.camera.updateMatrixWorld();
    const v = b.root.position.clone();
    v.y = 0.56;
    v.project(engine.camera);
    const x = ((v.x + 1) / 2) * 800;
    const y = ((1 - v.y) / 2) * 600;
    expect(engine.pickAt(x, y)).toEqual({ kind: "buffer", id: "B2" });
    engine.dispose();
  });
});

describe("pointer events", () => {
  it("click selects (drag does not), hover emits once per target with a tooltip", () => {
    const { engine, renderer, clock } = readyEngine();
    const selects: SelectEvent[] = [];
    const hovers: HoverEvent[] = [];
    engine.on("select", (e) => selects.push(e));
    const off = engine.on("hover", (e) => hovers.push(e));
    const p = engine.screenPosition("CNC-01");
    if (p === null) throw new Error("no CNC");
    const el = renderer.element;

    el.dispatch("pointerdown", { clientX: p.x, clientY: p.y });
    el.dispatch("pointerup", { clientX: p.x, clientY: p.y });
    expect(selects).toEqual([{ machineId: "CNC-01" }]);
    expect(engine.selected).toBe("CNC-01");
    expect(engine.machineModel("CNC-01")?.isSelected).toBe(true);

    el.dispatch("pointerdown", { clientX: 2, clientY: 2 });
    el.dispatch("pointerup", { clientX: 60, clientY: 60 }); // drag: orbiting, not a click
    expect(engine.selected).toBe("CNC-01");
    el.dispatch("pointerdown", { clientX: 2, clientY: 2 });
    el.dispatch("pointerup", { clientX: 2, clientY: 2 }); // click on nothing: deselect
    expect(selects.at(-1)).toEqual({ machineId: null });
    expect(engine.machineModel("CNC-01")?.isSelected).toBe(false);

    el.dispatch("pointermove", { clientX: p.x, clientY: p.y });
    engine.tick((clock.now += 16));
    engine.tick((clock.now += 16)); // same target: no second event
    expect(hovers).toHaveLength(1);
    expect(hovers[0]?.target).toEqual({ kind: "machine", id: "CNC-01" });
    expect(hovers[0]?.tooltip?.title).toBe("CNC-01 · CNC machining station");
    expect(hovers[0]?.tooltip?.rows.every((r) => r.value === "–")).toBe(true); // no data yet
    el.dispatch("pointerleave");
    expect(hovers.at(-1)?.target).toBeNull();
    off();
    el.dispatch("pointermove", { clientX: p.x, clientY: p.y });
    engine.tick((clock.now += 16));
    expect(hovers).toHaveLength(2);
    engine.dispose();
  });
});

describe("highlightParts", () => {
  it("adds emissive to the named parts, clamps intensity, reports unknown parts, composes with FAULT", () => {
    const { engine, clock } = readyEngine();
    const cnc = engine.machineModel("CNC-01");
    if (cnc === undefined) throw new Error("no CNC");
    const unknown = engine.highlightParts("CNC-01", [
      { part: "spindle_motor", intensity: 0.5 },
      { part: "spindle", intensity: 7 },
      { part: "coolant", intensity: 1 },
    ]);
    expect(unknown).toEqual(["coolant"]);
    expect(cnc.highlightMap.get("spindle")).toBe(1);
    engine.tick((clock.now += 16));
    const e = (part: string): number => cnc.partMaterials(part)[0]?.material.emissive.b ?? -1;
    expect(e("spindle_motor")).toBeGreaterThan(0);
    expect(e("spindle")).toBeGreaterThan(e("spindle_motor"));
    expect(e("spindle_bearing")).toBe(0);

    // FAULT flashes red on every part (no attribution) and wins over the highlight.
    engine.applyStateChange({ machineId: "CNC-01", state: "FAULT", previousState: "RUNNING", ts: "2026-01-01T00:00:01Z", reason: null });
    engine.tick((clock.now += 16));
    const spindle = cnc.partMaterials("spindle")[0]?.material.emissive;
    expect(spindle && spindle.r > spindle.b).toBe(true);

    engine.highlightParts("CNC-01", []);
    engine.applyStateChange({ machineId: "CNC-01", state: "IDLE", previousState: "FAULT", ts: "2026-01-01T00:00:02Z", reason: null });
    engine.tick((clock.now += 16));
    expect(e("spindle")).toBe(0);
    expect(engine.highlightParts("NOPE", [{ part: "x", intensity: 1 }])).toEqual(["x"]);
    engine.dispose();
  });
});

describe("setMode", () => {
  it("switches the scene cue and records the mode", () => {
    const { engine, container } = readyEngine();
    expect(engine.mode).toBe("live");
    engine.setMode("replay");
    expect(engine.mode).toBe("replay");
    expect(container.dataset.twinMode).toBe("replay");
    engine.setMode("live");
    expect(container.dataset.twinMode).toBe("live");
    engine.dispose();
  });
});

describe("dispose", () => {
  it("frees every geometry, material and texture, all listeners, the frame loop and the DOM", () => {
    const t = createTestEngine(); // frame loop running
    const { engine, renderer, controls, container, frames, clock } = t;
    engine.loadLayout(LINE);
    engine.loadLayout(generateLine(6)); // a replaced layout must not leak either
    const readings = new SyntheticReadings();
    for (const m of generateLine(6).machines) engine.applyReading(readings.next(m, "WARNING", 1));
    engine.on("select", () => undefined);
    engine.on("hover", () => undefined);
    frames.run((clock.now = 16));
    frames.run((clock.now = 32));

    // Every GPU-backed object reachable from the scene, plus what the trackers hold.
    const disposables = new Set<{ dispose(): void; addEventListener(t: string, f: () => void): void }>();
    engine.scene.traverse((o) => {
      const mesh = o as Partial<Mesh> & { isSprite?: boolean };
      // Sprite geometry is a single three.js-internal buffer shared by every sprite (never ours).
      if (mesh.geometry !== undefined && mesh.isSprite !== true) disposables.add(mesh.geometry);
      const mats = mesh.material === undefined ? [] : Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const m of mats) {
        disposables.add(m);
        const map = (m as Material & { map?: Texture | null }).map;
        if (map) disposables.add(map);
      }
    });
    expect(disposables.size).toBeGreaterThan(50);
    const disposed = new Set<unknown>();
    for (const d of disposables) d.addEventListener("dispose", () => disposed.add(d));

    const before = engine.resourceCounts();
    expect(before.listeners).toBeGreaterThan(0);
    expect(before.frames).toBe(1);
    expect(renderer.element.listenerCount).toBeGreaterThan(0);
    expect(container.children).toContain(renderer.element);

    engine.dispose();
    const missing = [...disposables].filter((d) => !disposed.has(d));
    expect(missing.map((m) => (m as { type?: string }).type)).toEqual([]);
    expect(engine.resourceCounts()).toEqual({
      geometries: 0,
      materials: 0,
      textures: 0,
      other: 0,
      listeners: 0,
      handlers: 0,
      frames: 0,
    });
    expect(renderer.element.listenerCount).toBe(0);
    expect(frames.pending.size).toBe(0);
    expect(renderer.disposed).toBe(true);
    expect(renderer.contextLost).toBe(true);
    expect(controls.disposed).toBe(true);
    expect(container.children).not.toContain(renderer.element);
    expect(() => {
      engine.dispose();
    }).not.toThrow(); // re-entrant
    expect(() => {
      engine.loadLayout(LINE);
    }).toThrow(/disposed/);
  });

  it("frees the old layout's resources on loadLayout", () => {
    const { engine } = createTestEngine({ autoStart: false });
    engine.loadLayout(LINE);
    const one = engine.resourceCounts();
    engine.loadLayout(LINE);
    engine.loadLayout(LINE);
    expect(engine.resourceCounts()).toEqual(one);
    engine.dispose();
  });
});

describe("FpsCounter", () => {
  it("measures frames per second and the worst frame over a sliding window", () => {
    const c = new FpsCounter(1000);
    expect(c.fps).toBe(0);
    for (let i = 0; i <= 60; i++) c.frame(i * (1000 / 60));
    expect(c.fps).toBeCloseTo(60, 0);
    c.frame(1000 + 100); // one 100 ms hitch
    expect(c.worstFrameMs).toBeGreaterThan(80);
    for (let i = 1; i <= 120; i++) c.frame(1100 + i * 33.3);
    expect(c.fps).toBeCloseTo(30, 0);
  });
});

describe("synthetic data", () => {
  it("is deterministic and plausible", () => {
    const line = generateLine(20);
    expect(line.machines.map((m) => m.kind).slice(0, 4)).toEqual(["CONVEYOR", "CNC", "INSPECTION", "CONVEYOR"]);
    const a = new SyntheticReadings(3);
    const b = new SyntheticReadings(3);
    const m = line.machines[1];
    if (m === undefined) throw new Error("empty");
    expect(a.next(m, "RUNNING", 1)).toEqual(b.next(m, "RUNNING", 1));
    const r = a.next(m, "IDLE", 2);
    expect(r.speed).toBe(0);
    expect(r.seq).toBe(2);
    expect(stateSceneLine().machines).toHaveLength(STATE_SCENE.length);
  });
});

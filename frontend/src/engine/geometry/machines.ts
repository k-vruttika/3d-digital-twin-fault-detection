/**
 * Procedural machine models (no external model files). Local frame: material flows along +x,
 * y is up, the footprint is centred on the origin.
 *
 * Every machine exposes its config parts as named groups (`parts`), so a part can be outlined,
 * flashed or highlighted. Parts the kind's builder does not know get a small generic housing
 * (`ensureParts`), so every part in the line config is always addressable.
 */
import { Group, InstancedMesh, Matrix4, Quaternion, Vector3, type Mesh, type Texture } from "three";

import type { LineMachine } from "../../api/types.ts";
import type { Footprint } from "../config.ts";
import { PartKit, type BuildContext, type PartMaterial } from "./shared.ts";

/** Advances the machine's motion; `rate` 1 = nominal speed, 0 = stopped. */
export type Animator = (dt: number, rate: number) => void;

export interface BuiltMachine {
  /** Model root (local frame); the engine places and rotates it. */
  root: Group;
  parts: Map<string, Group>;
  materials: PartMaterial[];
  animate: Animator;
  /** Top of the model (m), for beacons and labels. */
  height: number;
  /** Belt geometry for moving items (conveyors only). */
  belt: { length: number; top: number } | null;
}

export function buildMachine(ctx: BuildContext, machine: LineMachine, fp: Footprint): BuiltMachine {
  const kit = new PartKit(ctx);
  const root = new Group();
  root.name = machine.id;
  let built: Omit<BuiltMachine, "root" | "parts" | "materials">;
  switch (machine.kind) {
    case "CONVEYOR":
      built = conveyor(kit, root, fp);
      break;
    case "CNC":
      built = cnc(kit, root);
      break;
    case "INSPECTION":
      built = inspection(kit, root);
      break;
    default:
      built = generic(kit, root, fp);
  }
  ensureParts(kit, root, machine.parts, fp);
  return { root, parts: kit.parts, materials: kit.materials, ...built };
}

/** A small housing for every config part the builder did not create. */
export function ensureParts(kit: PartKit, root: Group, parts: readonly string[], fp: Footprint): void {
  const missing = parts.filter((p) => !kit.parts.has(p));
  missing.forEach((name, i) => {
    const group = kit.part(name, root);
    const mat = kit.material(kit.ctx.cfg.colors.accent, name);
    const x = -fp.length / 2 + 0.2 + i * 0.32;
    kit.box(mat, [0.24, 0.24, 0.24], [x, 0.12, fp.width / 2 - 0.15], group);
  });
}

// ---------------------------------------------------------------- conveyor
function conveyor(kit: PartKit, root: Group, fp: Footprint): Omit<BuiltMachine, "root" | "parts" | "materials"> {
  const { cfg } = kit.ctx;
  const L = fp.length;
  const W = cfg.conveyor.beltWidth;
  const H = cfg.conveyor.height;
  const frame = kit.material(cfg.colors.frame, null, { metalness: 0.6, roughness: 0.4 });
  const side = W / 2 + 0.05;

  for (const z of [-side, side]) {
    kit.box(frame, [L, 0.14, 0.06], [0, H - 0.07, z], root);
    for (const x of [-(L / 2 - 0.2), L / 2 - 0.2]) {
      kit.box(frame, [0.07, H - 0.1, 0.07], [x, (H - 0.1) / 2, z], root);
    }
  }

  // Rollers: one InstancedMesh per conveyor.
  const rollerMat = kit.material(cfg.colors.accent, null, { metalness: 0.7, roughness: 0.3 });
  const count = Math.max(1, Math.floor(L / cfg.conveyor.rollerPitch));
  const rollers = new InstancedMesh(kit.ctx.cache.cylinder, rollerMat, count);
  rollers.name = "rollers";
  const q = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), Math.PI / 2);
  const s = new Vector3(0.09, W, 0.09);
  const m = new Matrix4();
  for (let i = 0; i < count; i++) {
    const x = -L / 2 + (L / count) * (i + 0.5);
    rollers.setMatrixAt(i, m.compose(new Vector3(x, H - 0.06, 0), q, s));
  }
  rollers.instanceMatrix.needsUpdate = true;
  rollers.receiveShadow = true;
  kit.ctx.tracker.track(rollers);
  root.add(rollers);

  // Belt: a textured plane whose offset scrolls with the belt speed, plus the end drums.
  const repeatLength = 0.5;
  const tex: Texture = kit.ctx.tracker.track(kit.ctx.beltTexture.clone());
  tex.repeat.set((L - 0.1) / repeatLength, 1);
  tex.needsUpdate = true;
  const belt = kit.part("belt", root);
  const beltMat = kit.material(cfg.colors.belt, "belt", { metalness: 0.05, roughness: 0.9, map: tex });
  // The map carries the colour; keep the base white so the stripes are not darkened twice.
  beltMat.color.set(0xffffff);
  const last = kit.materials[kit.materials.length - 1];
  if (last !== undefined) last.base.set(0xffffff);
  const plane = kit.mesh(kit.ctx.cache.plane, beltMat, [L - 0.1, W, 1], [0, H + 0.005, 0], belt);
  plane.rotation.x = -Math.PI / 2;
  const drumMat = kit.material(cfg.colors.accent, "belt", { metalness: 0.7, roughness: 0.3 });
  const drums: Mesh[] = [-1, 1].map((d) =>
    kit.cylinder(drumMat, 0.07, W, [d * (L / 2 - 0.05), H - 0.065, 0], belt, "z"),
  );

  const motor = kit.part("motor", root);
  const motorMat = kit.material(0x37474f, "motor", { metalness: 0.5 });
  const mx = L / 2 - 0.25;
  kit.cylinder(motorMat, 0.14, 0.36, [mx, H - 0.25, -(side + 0.32)], motor, "z");
  kit.box(motorMat, [0.24, 0.24, 0.2], [mx, H - 0.25, -(side + 0.08)], motor);

  const bearing = kit.part("bearing", root);
  const bearingMat = kit.material(0x8d6e63, "bearing", { metalness: 0.6 });
  for (const x of [-(L / 2 - 0.05), L / 2 - 0.05]) {
    for (const z of [-(side + 0.04), side + 0.04]) {
      kit.box(bearingMat, [0.13, 0.13, 0.08], [x, H - 0.065, z], bearing);
    }
  }

  const travelPerS = cfg.beltSpeed;
  return {
    height: H + 0.15,
    belt: { length: L - 0.3, top: H + 0.01 },
    animate: (dt, rate) => {
      if (rate <= 0) return;
      const metres = travelPerS * rate * dt;
      tex.offset.x = (tex.offset.x - metres / repeatLength) % 1;
      for (const drum of drums) drum.rotation.y -= metres / 0.07;
    },
  };
}

// ---------------------------------------------------------------- CNC
function cnc(kit: PartKit, root: Group): Omit<BuiltMachine, "root" | "parts" | "materials"> {
  const { cfg } = kit.ctx;
  const body = kit.material(cfg.colors.body, null);
  const accent = kit.material(cfg.colors.accent, null, { metalness: 0.6 });
  kit.box(body, [2.0, 0.8, 1.6], [0, 0.4, 0], root); // bed
  kit.box(accent, [1.4, 0.1, 0.9], [0, 0.85, 0.1], root); // table
  kit.box(body, [0.55, 1.7, 0.55], [0, 1.65, -0.55], root); // column

  const head = new Group();
  head.position.set(0, 1.95, -0.12);
  root.add(head);
  const motor = kit.part("spindle_motor", head);
  kit.box(kit.material(0x455a64, "spindle_motor", { metalness: 0.5 }), [0.45, 0.55, 0.45], [0, 0.22, 0], motor);
  const bearing = kit.part("spindle_bearing", head);
  kit.cylinder(kit.material(0x8d6e63, "spindle_bearing", { metalness: 0.6 }), 0.13, 0.2, [0, -0.15, 0], bearing);
  const spindle = kit.part("spindle", head);
  const spindleMat = kit.material(0xcfd8dc, "spindle", { metalness: 0.9, roughness: 0.2 });
  kit.cylinder(spindleMat, 0.05, 0.4, [0, -0.45, 0], spindle);
  const tool = kit.mesh(kit.ctx.cache.cone, spindleMat, [0.08, 0.14, 0.08], [0, -0.72, 0], spindle);
  tool.rotation.x = Math.PI; // tip down

  // Enclosure: thin frame and glass panels (see-through).
  const glass = kit.material(cfg.colors.glass, null, { opacity: 0.16, metalness: 0.1, roughness: 0.1 });
  const ex = 1.25;
  const ez = 1.1;
  const eh = 2.55;
  for (const x of [-ex, ex]) {
    for (const z of [-ez, ez]) kit.box(accent, [0.05, eh, 0.05], [x, eh / 2, z], root);
  }
  // Roof: thin frame edges and a glass panel, so the spindle stays visible from above.
  for (const z of [-ez, ez]) kit.box(accent, [ex * 2, 0.05, 0.05], [0, eh, z], root);
  for (const x of [-ex, ex]) kit.box(accent, [0.05, 0.05, ez * 2], [x, eh, 0], root);
  kit.box(glass, [ex * 2, 0.02, ez * 2], [0, eh, 0], root);
  kit.box(glass, [ex * 2, eh, 0.02], [0, eh / 2, ez], root);
  kit.box(glass, [ex * 2, eh, 0.02], [0, eh / 2, -ez], root);
  kit.box(glass, [0.02, eh, ez * 2], [ex, eh / 2, 0], root);
  kit.box(glass, [0.02, eh, ez * 2], [-ex, eh / 2, 0], root);

  const spin = cfg.spindleTurnsPerS * 2 * Math.PI;
  let phase = 0;
  return {
    height: eh,
    belt: null,
    animate: (dt, rate) => {
      if (rate <= 0) return;
      spindle.rotation.y += spin * rate * dt;
      phase += rate * dt;
      head.position.y = 1.95 - 0.08 * (0.5 + 0.5 * Math.sin(phase * 1.3));
    },
  };
}

// ---------------------------------------------------------------- inspection
function inspection(kit: PartKit, root: Group): Omit<BuiltMachine, "root" | "parts" | "materials"> {
  const { cfg } = kit.ctx;
  const body = kit.material(cfg.colors.body, null);
  const accent = kit.material(cfg.colors.accent, null, { metalness: 0.6 });
  kit.box(body, [1.8, 0.75, 1.6], [0, 0.375, 0], root); // base

  const bearing = kit.part("bearing", root);
  kit.cylinder(kit.material(0x8d6e63, "bearing", { metalness: 0.6 }), 0.35, 0.08, [0, 0.79, 0], bearing);
  const indexer = kit.part("indexer", root);
  indexer.position.set(0, 0.87, 0);
  const indexerMat = kit.material(0x90a4ae, "indexer", { metalness: 0.6 });
  kit.cylinder(indexerMat, 0.65, 0.08, [0, 0, 0], indexer);
  for (let i = 0; i < 4; i++) {
    const a = (i * Math.PI) / 2;
    kit.box(indexerMat, [0.16, 0.12, 0.16], [Math.cos(a) * 0.45, 0.1, Math.sin(a) * 0.45], indexer);
  }
  const motor = kit.part("motor", root);
  kit.box(kit.material(0x37474f, "motor", { metalness: 0.5 }), [0.36, 0.3, 0.3], [0.62, 0.45, 0.95], motor);

  // Gantry with a sweeping camera head.
  for (const x of [-0.85, 0.85]) kit.box(accent, [0.1, 1.45, 0.1], [x, 0.75 + 0.725, -0.1], root);
  kit.box(accent, [1.8, 0.12, 0.12], [0, 2.2, -0.1], root);
  const camera = new Group();
  camera.position.set(0, 2.0, -0.1);
  root.add(camera);
  kit.box(body, [0.22, 0.26, 0.22], [0, 0, 0], camera);
  kit.cylinder(kit.material(0x212121, null, { metalness: 0.2, roughness: 0.3 }), 0.05, 0.14, [0, -0.19, 0], camera);

  const spin = cfg.indexerTurnsPerS * 2 * Math.PI;
  const sweep = cfg.cameraSweepHz * 2 * Math.PI;
  let phase = 0;
  return {
    height: 2.3,
    belt: null,
    animate: (dt, rate) => {
      if (rate <= 0) return;
      indexer.rotation.y += spin * rate * dt;
      phase += rate * dt;
      camera.position.x = 0.6 * Math.sin(phase * sweep);
    },
  };
}

// ---------------------------------------------------------------- unknown kind
function generic(kit: PartKit, root: Group, fp: Footprint): Omit<BuiltMachine, "root" | "parts" | "materials"> {
  const body = kit.material(kit.ctx.cfg.colors.body, null);
  kit.box(body, [fp.length * 0.8, 1.0, fp.width * 0.8], [0, 0.5, 0], root);
  return { height: 1.0, belt: null, animate: () => undefined };
}

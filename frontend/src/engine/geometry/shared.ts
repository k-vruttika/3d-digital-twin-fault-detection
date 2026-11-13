/**
 * Shared building blocks: unit geometries reused by every machine (scaled per mesh), a
 * material/mesh kit that records which part every material belongs to, and procedural
 * textures. Nothing is loaded from files.
 */
import {
  BoxGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DataTexture,
  Group,
  Mesh,
  MeshStandardMaterial,
  OctahedronGeometry,
  PlaneGeometry,
  RepeatWrapping,
  RGBAFormat,
  RingGeometry,
  SphereGeometry,
  SRGBColorSpace,
  TorusGeometry,
  type BufferGeometry,
  type Material,
  type Object3D,
  type Texture,
} from "three";

import type { VisualConfig } from "../config.ts";
import type { ResourceTracker } from "../resources.ts";

/** Unit geometries (size 1, cylinders of diameter 1), created once per engine. */
export class GeometryCache {
  readonly box: BoxGeometry;
  readonly cylinder: CylinderGeometry;
  readonly cone: ConeGeometry;
  readonly sphere: SphereGeometry;
  readonly octahedron: OctahedronGeometry;
  readonly plane: PlaneGeometry;
  readonly ring: RingGeometry;
  readonly torus: TorusGeometry;

  constructor(tracker: ResourceTracker) {
    const t = <G extends BufferGeometry>(g: G): G => tracker.track(g);
    this.box = t(new BoxGeometry(1, 1, 1));
    this.cylinder = t(new CylinderGeometry(0.5, 0.5, 1, 24));
    this.cone = t(new ConeGeometry(0.5, 1, 16));
    this.sphere = t(new SphereGeometry(0.5, 16, 12));
    this.octahedron = t(new OctahedronGeometry(0.5));
    this.plane = t(new PlaneGeometry(1, 1));
    this.ring = t(new RingGeometry(0.9, 1, 48));
    this.torus = t(new TorusGeometry(1, 0.025, 8, 64));
  }
}

/** A material of a machine, with its as-built colour and the config part it belongs to. */
export interface PartMaterial {
  material: MeshStandardMaterial;
  base: Color;
  baseOpacity: number;
  /** Config part name, or null for structural parts. */
  part: string | null;
}

export interface BuildContext {
  cfg: VisualConfig;
  cache: GeometryCache;
  /** Owns everything created for the current layout. */
  tracker: ResourceTracker;
  /** Shared belt texture source (cloned per conveyor so each scrolls on its own). */
  beltTexture: Texture;
  /** Invisible material of the pick boxes (one box per machine / buffer). */
  pickMaterial: Material;
}

type Vec3 = readonly [number, number, number];

/** Creates meshes and per-part materials for one machine. */
export class PartKit {
  readonly materials: PartMaterial[] = [];
  readonly parts = new Map<string, Group>();

  constructor(readonly ctx: BuildContext) {}

  material(
    color: number,
    part: string | null,
    opts: { opacity?: number; metalness?: number; roughness?: number; map?: Texture } = {},
  ): MeshStandardMaterial {
    const opacity = opts.opacity ?? 1;
    const mat = this.ctx.tracker.track(
      new MeshStandardMaterial({
        color,
        metalness: opts.metalness ?? 0.35,
        roughness: opts.roughness ?? 0.55,
        transparent: opacity < 1,
        opacity,
        depthWrite: opacity >= 1,
        ...(opts.map === undefined ? {} : { map: opts.map }),
      }),
    );
    this.materials.push({ material: mat, base: new Color(color), baseOpacity: opacity, part });
    return mat;
  }

  /** The named group of a config part (created on first use). */
  part(name: string, parent: Object3D): Group {
    let group = this.parts.get(name);
    if (group === undefined) {
      group = new Group();
      group.name = name;
      group.userData.part = name;
      parent.add(group);
      this.parts.set(name, group);
    }
    return group;
  }

  box(mat: MeshStandardMaterial, size: Vec3, pos: Vec3, parent: Object3D): Mesh {
    return this.mesh(this.ctx.cache.box, mat, size, pos, parent);
  }

  /** A cylinder of the given radius and length along an axis. */
  cylinder(
    mat: MeshStandardMaterial,
    radius: number,
    length: number,
    pos: Vec3,
    parent: Object3D,
    axis: "x" | "y" | "z" = "y",
  ): Mesh {
    const m = this.mesh(this.ctx.cache.cylinder, mat, [radius * 2, length, radius * 2], pos, parent);
    if (axis === "x") m.rotation.z = Math.PI / 2;
    if (axis === "z") m.rotation.x = Math.PI / 2;
    return m;
  }

  mesh(geometry: BufferGeometry, mat: MeshStandardMaterial, size: Vec3, pos: Vec3, parent: Object3D): Mesh {
    const m = new Mesh(geometry, mat);
    m.scale.set(size[0], size[1], size[2]);
    m.position.set(pos[0], pos[1], pos[2]);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  }
}

/** A stripe texture for belts (DataTexture: works without a DOM). */
export function makeBeltTexture(cfg: VisualConfig, tracker: ResourceTracker): DataTexture {
  // Stripes run across the belt (they vary along u = the flow direction), so scrolling the
  // texture offset along u makes the belt visibly move.
  const w = 32;
  const h = 4;
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const hex = x % 16 < 3 ? cfg.colors.beltStripe : cfg.colors.belt; // sRGB bytes
      const i = (y * w + x) * 4;
      data[i] = (hex >> 16) & 0xff;
      data[i + 1] = (hex >> 8) & 0xff;
      data[i + 2] = hex & 0xff;
      data[i + 3] = 255;
    }
  }
  const tex = new DataTexture(data, w, h, RGBAFormat);
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.colorSpace = SRGBColorSpace;
  tex.needsUpdate = true;
  return tracker.track(tex);
}

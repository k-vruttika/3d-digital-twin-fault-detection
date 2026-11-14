/**
 * Runtime model of one machine in the scene: applies a VisualSpec to its materials and
 * indicators, animates it at the smoothed rate, and composes part highlights.
 */
import {
  BackSide,
  Color,
  Group,
  InstancedMesh,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Sprite,
  SpriteMaterial,
  Vector3,
  type Material,
  type Object3D,
} from "three";

import type { AlertSeverity } from "../api/types.ts";
import type { BuiltMachine } from "./geometry/machines.ts";
import type { BuildContext, PartMaterial } from "./geometry/shared.ts";
import { labelScreenScale, MIN_VISIBLE_OPACITY } from "./labelPolicy.ts";
import type { LabelTextureFactory } from "./labels.ts";
import type { Placement } from "./layout.ts";
import { smoothRate, toneColor, type VisualSpec } from "./visuals.ts";

const GREY = new Color();
const TMP = new Color();
const BLACK = new Color(0x000000);
const LABEL_POS = new Vector3();

/** Luminance-preserving grey of a colour. */
function greyOf(c: Color, out: Color): Color {
  const l = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  return out.setRGB(l, l, l);
}

export class MachineModel {
  readonly id: string;
  readonly root: Group;
  readonly parts: Map<string, Group>;
  readonly height: number;
  readonly belt: BuiltMachine["belt"];
  /** Meshes used for picking (no outlines, no indicators). */
  readonly pickables: Object3D[] = [];
  /** Shown rate (smoothed) and the rate the reported speed asks for. */
  rate = 0;
  targetRate = 1;
  spec: VisualSpec | null = null;

  private readonly materials: PartMaterial[];
  private readonly animateFn: BuiltMachine["animate"];
  private readonly outlines = new Map<string, Mesh[]>();
  private readonly outlineMaterial: MeshBasicMaterial;
  private highlights = new Map<string, number>();
  private selected = false;
  private labelKey = "";
  /** Texture aspect of the current label; null = no label texture. */
  private labelAspect: number | null = null;
  private flashPhase = 0;

  // indicators
  private readonly beacon: Group;
  private readonly lampMaterial: MeshStandardMaterial;
  private readonly halo: Mesh;
  private readonly marker: Mesh;
  private readonly markerMaterial: MeshStandardMaterial;
  private readonly selectionRing: Mesh;
  private readonly label: Sprite;
  private readonly labelMaterial: SpriteMaterial;

  constructor(
    built: BuiltMachine,
    placement: Placement,
    private readonly ctx: BuildContext,
    private readonly labelFactory: LabelTextureFactory | null,
  ) {
    const { cfg, cache, tracker } = ctx;
    this.id = placement.id;
    this.root = built.root;
    this.parts = built.parts;
    this.materials = built.materials;
    this.height = built.height;
    this.belt = built.belt;
    this.animateFn = built.animate;
    this.root.position.set(placement.x, placement.y, placement.z);
    this.root.rotation.y = placement.rotationY;
    this.root.userData.machineId = this.id;

    this.root.traverse((o) => {
      if (o instanceof Mesh) o.userData.machineId = this.id;
    });
    // One invisible box over footprint and height: open structures (gantries, glass) are easy
    // to hover, and picking tests one box per machine instead of every mesh.
    const pick = new Mesh(cache.box, ctx.pickMaterial);
    pick.scale.set(placement.footprint.length, this.height, placement.footprint.width);
    pick.position.y = this.height / 2;
    pick.userData.machineId = this.id;
    this.root.add(pick);
    this.pickables.push(pick);

    // Inverted-hull outlines per config part (hidden until a WARNING names the part).
    this.outlineMaterial = tracker.track(
      new MeshBasicMaterial({ color: cfg.colors.warning, side: BackSide }),
    );
    const t = cfg.outlineThickness;
    const grow = (size: number): number => 1 + (2 * t) / Math.max(size, 1e-3);
    for (const [name, group] of this.parts) {
      const meshes: Mesh[] = [];
      group.traverse((o) => {
        if (o instanceof Mesh && !(o instanceof InstancedMesh)) meshes.push(o as Mesh);
      });
      // Hulls are children of their mesh, so they inherit its scale: grow each axis by 2t.
      const hulls = meshes.map((mesh) => {
        const hull = new Mesh(mesh.geometry, this.outlineMaterial);
        hull.scale.set(grow(mesh.scale.x), grow(mesh.scale.y), grow(mesh.scale.z));
        hull.visible = false;
        hull.userData.noPick = true;
        mesh.add(hull);
        return hull;
      });
      this.outlines.set(name, hulls);
    }

    const fp = placement.footprint;
    const reach = Math.max(fp.length, fp.width) / 2;

    // Beacon (STARVED / BLOCKED yellow, MAINTENANCE blue).
    this.beacon = new Group();
    this.beacon.position.set(0, this.height, 0);
    const postMat = tracker.track(new MeshStandardMaterial({ color: cfg.colors.accent }));
    const post = new Mesh(cache.cylinder, postMat);
    post.scale.set(0.04, 0.3, 0.04);
    post.position.y = 0.15;
    this.lampMaterial = tracker.track(new MeshStandardMaterial({ color: cfg.colors.waiting, emissiveIntensity: 1.2 }));
    const lamp = new Mesh(cache.sphere, this.lampMaterial);
    lamp.scale.setScalar(0.2);
    lamp.position.y = 0.38;
    this.beacon.add(post, lamp);
    this.beacon.visible = false;
    this.root.add(this.beacon);

    // Data-quality halo on the floor.
    const haloMat = tracker.track(
      new MeshBasicMaterial({ color: cfg.colors.dataQuality, transparent: true, opacity: 0.45, depthWrite: false }),
    );
    this.halo = new Mesh(cache.ring, haloMat);
    this.halo.rotation.x = -Math.PI / 2;
    this.halo.position.y = 0.015;
    this.halo.scale.setScalar(reach + 0.35);
    this.halo.visible = false;
    this.root.add(this.halo);

    // Alert marker.
    this.markerMaterial = tracker.track(new MeshStandardMaterial({ color: cfg.colors.alert.WARNING, emissiveIntensity: 0.9 }));
    this.marker = new Mesh(cache.octahedron, this.markerMaterial);
    this.marker.scale.set(0.26, 0.36, 0.26);
    this.marker.position.set(0, this.height + 1.25, 0);
    this.marker.visible = false;
    this.root.add(this.marker);

    // Selection ring.
    const ringMat = tracker.track(new MeshBasicMaterial({ color: cfg.colors.selection, transparent: true, opacity: 0.9 }));
    this.selectionRing = new Mesh(cache.torus, ringMat);
    this.selectionRing.rotation.x = -Math.PI / 2;
    this.selectionRing.position.y = 0.03;
    this.selectionRing.scale.setScalar(reach + 0.2);
    this.selectionRing.visible = false;
    this.root.add(this.selectionRing);

    // State label (camera-facing).
    this.labelMaterial = tracker.track(new SpriteMaterial({ transparent: true, depthWrite: false, depthTest: false }));
    this.label = new Sprite(this.labelMaterial);
    this.label.position.set(0, this.height + 0.75, 0);
    this.label.renderOrder = 10;
    this.label.visible = false;
    this.root.add(this.label);

    for (const o of [this.beacon, this.halo, this.marker, this.selectionRing, this.label]) {
      o.traverse((c) => (c.userData.noPick = true));
    }
  }

  get labelText(): string | null {
    return this.spec?.label?.text ?? null;
  }

  get isSelected(): boolean {
    return this.selected;
  }

  /** Static part of a visual spec: colours, opacity, indicators, label. */
  applyVisual(spec: VisualSpec): void {
    this.spec = spec;
    const cfg = this.ctx.cfg;
    for (const pm of this.materials) {
      TMP.copy(pm.base);
      if (spec.grey > 0) TMP.lerp(greyOf(pm.base, GREY), spec.grey).multiplyScalar(1 - 0.35 * spec.grey);
      if (spec.tint !== null && spec.tintAmount > 0) TMP.lerp(GREY.set(spec.tint), spec.tintAmount);
      pm.material.color.copy(TMP);
      const opacity = pm.baseOpacity * spec.opacity;
      pm.material.opacity = opacity;
      const transparent = opacity < 1;
      if (pm.material.transparent !== transparent) {
        pm.material.transparent = transparent;
        pm.material.depthWrite = !transparent;
        pm.material.needsUpdate = true; // blending changed: the program must be re-evaluated
      }
    }

    const outlined = new Set(spec.outlineParts);
    if (spec.outlineColor !== null) this.outlineMaterial.color.set(spec.outlineColor);
    for (const [part, hulls] of this.outlines) {
      for (const h of hulls) h.visible = outlined.has(part);
    }

    this.beacon.visible = spec.beacon !== "none";
    if (spec.beacon !== "none") {
      const c = spec.beacon === "waiting" ? cfg.colors.waiting : cfg.colors.maintenance;
      this.lampMaterial.color.set(c);
      this.lampMaterial.emissive.set(c);
    }
    this.halo.visible = spec.dqBadge;
    this.setAlertMarker(spec.alertMarker);
    this.setLabel(spec);
  }

  setHighlights(highlights: ReadonlyMap<string, number>): void {
    this.highlights = new Map(highlights);
  }

  get highlightMap(): ReadonlyMap<string, number> {
    return this.highlights;
  }

  setSelected(selected: boolean): void {
    this.selected = selected;
    this.selectionRing.visible = selected;
  }

  /** Per frame: rate smoothing, motion, flashing parts and highlights, marker spin. */
  update(dt: number): void {
    const spec = this.spec;
    const animate = spec?.animate ?? false;
    this.rate = smoothRate(this.rate, animate ? this.targetRate : 0, dt, this.ctx.cfg.rateResponse);
    if (this.rate < 1e-3 && !animate) this.rate = 0;
    this.animateFn(dt, this.rate);

    this.flashPhase += dt * this.ctx.cfg.flashHz * 2 * Math.PI;
    const flash = new Set(spec?.flashParts ?? []);
    const pulse = 0.5 + 0.5 * Math.sin(this.flashPhase);
    for (const pm of this.materials) {
      const e = pm.material.emissive;
      const part = pm.part;
      if (part !== null && flash.has(part) && spec?.flashColor != null) {
        e.set(spec.flashColor).multiplyScalar(0.25 + 0.75 * pulse);
        continue;
      }
      const h = part === null ? 0 : (this.highlights.get(part) ?? 0);
      if (h > 0) e.set(this.ctx.cfg.colors.highlight).multiplyScalar(0.8 * h);
      else e.copy(BLACK);
    }
    if (this.marker.visible) this.marker.rotation.y += dt * 1.5;
  }

  /** Every material this model owns (tests). */
  ownedMaterials(): Material[] {
    return this.materials.map((m) => m.material);
  }

  /** The PartMaterials of one config part (tests and highlight checks). */
  partMaterials(part: string): PartMaterial[] {
    return this.materials.filter((m) => m.part === part);
  }

  outlineVisible(part: string): boolean {
    return (this.outlines.get(part) ?? []).some((h) => h.visible);
  }

  get indicators(): { beacon: boolean; halo: boolean; marker: boolean; label: boolean } {
    return {
      beacon: this.beacon.visible,
      halo: this.halo.visible,
      marker: this.marker.visible,
      label: this.label.visible,
    };
  }

  private setAlertMarker(severity: AlertSeverity | null): void {
    this.marker.visible = severity !== null;
    if (severity === null) return;
    const c = this.ctx.cfg.colors.alert[severity];
    this.markerMaterial.color.set(c);
    this.markerMaterial.emissive.set(c);
  }

  private setLabel(spec: VisualSpec): void {
    const label = spec.label;
    const key = label === null ? "" : `${label.tone}|${label.text}`;
    if (key === this.labelKey) return;
    this.labelKey = key;
    const old = this.labelMaterial.map;
    this.labelMaterial.map = null;
    this.ctx.tracker.release(old);
    if (label === null || this.labelFactory === null) {
      this.labelAspect = null;
      this.label.visible = false;
      return;
    }
    const cfg = this.ctx.cfg;
    const made = this.labelFactory(label.text, {
      text: cfg.colors.label.text,
      background: cfg.colors.label.background,
      accent: toneColor(label.tone, cfg),
      fontPx: cfg.labels.fontPx,
    });
    if (made === null) {
      this.labelAspect = null;
      this.label.visible = false;
      return;
    }
    this.labelMaterial.map = this.ctx.tracker.track(made.texture);
    this.labelMaterial.needsUpdate = true;
    this.labelAspect = made.aspect;
  }

  /**
   * Per frame: show the label if the policy says so (`show`), at a constant on-screen size
   * and faded by distance (labelPolicy.ts); `exempt` (hovered / selected) never fades.
   */
  updateLabel(cameraPosition: Vector3, fovDeg: number, viewportPx: number, show: boolean, exempt: boolean): void {
    if (!show || this.labelAspect === null) {
      this.label.visible = false;
      return;
    }
    this.label.getWorldPosition(LABEL_POS);
    const { worldHeight, opacity } = labelScreenScale(
      LABEL_POS.distanceTo(cameraPosition),
      fovDeg,
      viewportPx,
      this.ctx.cfg.labels,
      exempt,
    );
    this.label.visible = opacity >= MIN_VISIBLE_OPACITY;
    this.labelMaterial.opacity = opacity;
    this.label.scale.set(worldHeight * this.labelAspect, worldHeight, 1);
  }
}

/**
 * Buffers between machines, shown qualitatively (ADR 0016): the line reports no buffer level,
 * so a buffer is FULL, EMPTY, FLOWING or UNKNOWN as inferred from its neighbours' states. FULL
 * shows a stack of `capacity` parts (that count is then known), EMPTY none, FLOWING a few parts
 * passing through (no count), UNKNOWN a greyed tray. A text label names the status.
 */
import {
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Quaternion,
  Sprite,
  SpriteMaterial,
  Vector3,
} from "three";

import type { BuildContext } from "./geometry/shared.ts";
import { labelScreenScale, MIN_VISIBLE_OPACITY } from "./labelPolicy.ts";
import type { LabelTextureFactory } from "./labels.ts";
import type { BufferPlacement } from "./layout.ts";
import { toneColor, type BufferStatus, type Tone } from "./visuals.ts";

const HIDDEN = new Matrix4().makeScale(0, 0, 0);
const TRAY_LENGTH = 1.2;
const TRAY_WIDTH = 0.8;
const TRAY_HEIGHT = 0.55;
const FLOWING_ITEMS = 2;
/** Buffer labels are drawn a little smaller than machine labels. */
const BUFFER_LABEL_SCALE = 0.8;
const LABEL_POS = new Vector3();

const STATUS_TONE: Record<BufferStatus, Tone> = {
  FULL: "waiting",
  EMPTY: "neutral",
  FLOWING: "running",
  UNKNOWN: "stale",
};

export class BufferModel {
  readonly id: string;
  readonly placement: BufferPlacement;
  readonly root = new Group();
  readonly pickables: Mesh[] = [];
  status: BufferStatus = "UNKNOWN";
  why = "";

  private readonly stack: InstancedMesh;
  private readonly trayMaterial: MeshStandardMaterial;
  private readonly stackCount: number;
  private readonly label: Sprite;
  private readonly labelMaterial: SpriteMaterial;
  private phase = 0;
  private dirty = true;
  private labelAspect: number | null = null;

  constructor(
    placement: BufferPlacement,
    private readonly ctx: BuildContext,
    private readonly labelFactory: LabelTextureFactory | null,
  ) {
    const { cfg, cache, tracker } = ctx;
    this.id = placement.id;
    this.placement = placement;
    this.root.name = `buffer:${placement.id}`;
    this.root.position.set(placement.x, placement.y, placement.z);
    this.root.rotation.y = placement.rotationY;
    this.root.userData.bufferId = placement.id;

    this.trayMaterial = tracker.track(new MeshStandardMaterial({ color: cfg.colors.frame, metalness: 0.5, roughness: 0.5 }));
    const tray = new Mesh(cache.box, this.trayMaterial);
    tray.scale.set(TRAY_LENGTH, 0.06, TRAY_WIDTH);
    tray.position.y = TRAY_HEIGHT;
    tray.receiveShadow = true;
    const legs = [-1, 1].flatMap((sx) =>
      [-1, 1].map((sz) => {
        const leg = new Mesh(cache.box, this.trayMaterial);
        leg.scale.set(0.05, TRAY_HEIGHT, 0.05);
        leg.position.set((sx * (TRAY_LENGTH - 0.1)) / 2, TRAY_HEIGHT / 2, (sz * (TRAY_WIDTH - 0.1)) / 2);
        return leg;
      }),
    );
    this.root.add(tray, ...legs);
    const pick = new Mesh(cache.box, ctx.pickMaterial);
    pick.scale.set(TRAY_LENGTH, TRAY_HEIGHT + 0.5, TRAY_WIDTH);
    pick.position.y = (TRAY_HEIGHT + 0.5) / 2;
    pick.userData.bufferId = placement.id;
    this.root.add(pick);
    this.pickables.push(pick);

    const itemMat = tracker.track(new MeshStandardMaterial({ color: cfg.colors.item, roughness: 0.8 }));
    this.stackCount = Math.max(FLOWING_ITEMS, Math.min(placement.capacity, cfg.buffer.maxStack));
    this.stack = tracker.track(new InstancedMesh(cache.box, itemMat, this.stackCount));
    this.stack.castShadow = true;
    this.stack.userData.bufferId = placement.id;
    this.root.add(this.stack);

    this.labelMaterial = tracker.track(new SpriteMaterial({ transparent: true, depthWrite: false, depthTest: false }));
    this.label = new Sprite(this.labelMaterial);
    this.label.position.y = TRAY_HEIGHT + 0.9;
    this.label.renderOrder = 10;
    this.label.visible = false;
    this.label.userData.noPick = true;
    this.root.add(this.label);
    this.layoutStack();
  }

  setStatus(status: BufferStatus, why: string): void {
    this.why = why;
    if (status === this.status && !this.dirty) return;
    this.status = status;
    this.dirty = false;
    const cfg = this.ctx.cfg;
    this.trayMaterial.color.set(status === "UNKNOWN" ? cfg.colors.offline : cfg.colors.frame);
    this.trayMaterial.opacity = status === "UNKNOWN" ? cfg.staleOpacity : 1;
    const transparent = status === "UNKNOWN";
    if (this.trayMaterial.transparent !== transparent) {
      this.trayMaterial.transparent = transparent;
      this.trayMaterial.needsUpdate = true;
    }
    this.layoutStack();
    this.setLabel(status);
  }

  /** Moves FLOWING parts along the tray; `rate` follows the feeding machine. */
  update(dt: number, rate: number): void {
    if (this.status !== "FLOWING" || rate <= 0) return;
    this.phase = (this.phase + dt * rate * this.ctx.cfg.beltSpeed) % TRAY_LENGTH;
    this.layoutStack();
  }

  /** Visible stacked parts (tests). */
  get visibleItems(): number {
    switch (this.status) {
      case "FULL":
        return Math.min(this.placement.capacity, this.stackCount);
      case "FLOWING":
        return FLOWING_ITEMS;
      default:
        return 0;
    }
  }

  private layoutStack(): void {
    const { boxSize, perLayer } = this.ctx.cfg.buffer;
    const visible = this.visibleItems;
    const m = new Matrix4();
    const q = new Quaternion();
    const s = new Vector3(boxSize, boxSize, boxSize);
    const cols = Math.ceil(perLayer / 2);
    for (let i = 0; i < this.stackCount; i++) {
      if (i >= visible) {
        this.stack.setMatrixAt(i, HIDDEN);
        continue;
      }
      let x: number;
      let y: number;
      let z: number;
      if (this.status === "FLOWING") {
        x = ((this.phase + (i * TRAY_LENGTH) / FLOWING_ITEMS) % TRAY_LENGTH) - TRAY_LENGTH / 2;
        y = TRAY_HEIGHT + 0.03 + boxSize / 2;
        z = 0;
      } else {
        const layer = Math.floor(i / perLayer);
        const k = i % perLayer;
        x = ((k % cols) - (cols - 1) / 2) * (boxSize + 0.04);
        z = (Math.floor(k / cols) - 0.5) * (boxSize + 0.04);
        y = TRAY_HEIGHT + 0.03 + boxSize / 2 + layer * (boxSize + 0.01);
      }
      this.stack.setMatrixAt(i, m.compose(new Vector3(x, y, z), q, s));
    }
    this.stack.instanceMatrix.needsUpdate = true;
  }

  private setLabel(status: BufferStatus): void {
    const old = this.labelMaterial.map;
    this.labelMaterial.map = null;
    this.ctx.tracker.release(old);
    const cfg = this.ctx.cfg;
    if (!cfg.labels.enabled || this.labelFactory === null) {
      this.labelAspect = null;
      this.label.visible = false;
      return;
    }
    const made = this.labelFactory(status === "UNKNOWN" ? "BUFFER ?" : status, {
      text: cfg.colors.label.text,
      background: cfg.colors.label.background,
      accent: toneColor(STATUS_TONE[status], cfg),
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

  /** Per frame, like MachineModel.updateLabel. */
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
    const h = worldHeight * BUFFER_LABEL_SCALE;
    this.label.visible = opacity >= MIN_VISIBLE_OPACITY;
    this.labelMaterial.opacity = opacity;
    this.label.scale.set(h * this.labelAspect, h, 1);
  }

  get labelVisible(): boolean {
    return this.label.visible;
  }
}

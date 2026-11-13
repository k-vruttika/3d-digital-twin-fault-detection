/**
 * TwinEngine: the framework-agnostic 3D digital twin of the line (Three.js, no React; ADR 0016).
 *
 * Feed it a layout (`GET /line`) and live data (WS snapshot and messages, or a replay) and it
 * renders every machine's state. All data is applied through the public methods below; the
 * engine never talks to the network itself (see api/bind.ts).
 *
 * Renderer, controls, label textures, clocks and the frame loop are injectable, so the engine
 * runs headless in tests.
 */
import {
  Color,
  DirectionalLight,
  GridHelper,
  Group,
  HemisphereLight,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PCFShadowMap,
  PerspectiveCamera,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  ACESFilmicToneMapping,
  WebGLRenderer,
  type Camera,
  type Object3D,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type {
  Alert,
  DataQualityReport,
  LineMachine,
  LineResponse,
  LineViewModel,
  MachineState,
  MachineView,
  Prediction,
  Reading,
  RootCauseView,
  StateChange,
} from "../api/types.ts";
import { DEFAULT_CONFIG, type VisualConfig } from "./config.ts";
import { buildMachine } from "./geometry/machines.ts";
import { GeometryCache, makeBeltTexture, type BuildContext } from "./geometry/shared.ts";
import { BufferModel } from "./buffers.ts";
import { FpsMeter } from "./fps.ts";
import { ItemFlow } from "./items.ts";
import { canvasLabelFactory, type LabelTextureFactory } from "./labels.ts";
import { computeLayout, footprintOf, type LineLayout } from "./layout.ts";
import { MachineModel } from "./machineModel.ts";
import { pickTarget, type PickTarget } from "./picking.ts";
import { ListenerRegistry, ResourceTracker } from "./resources.ts";
import { bufferTooltip, machineTooltip, type TooltipModel } from "./tooltip.ts";
import { TooltipView } from "./tooltipView.ts";
import { animationRate, bufferStatus, machineVisual, stateLabel, type VisualInput } from "./visuals.ts";

export type EngineMode = "live" | "replay";

export interface SelectEvent {
  machineId: string | null;
}

export interface HoverEvent {
  target: PickTarget | null;
  tooltip: TooltipModel | null;
  /** Pointer position relative to the container (px). */
  x: number;
  y: number;
}

interface EngineEvents {
  select: (e: SelectEvent) => void;
  hover: (e: HoverEvent) => void;
}

export interface RendererLike {
  readonly domElement: HTMLElement;
  shadowMap: { enabled: boolean };
  setSize(width: number, height: number, updateStyle?: boolean): void;
  setPixelRatio(ratio: number): void;
  setClearColor?(color: number): void;
  render(scene: Scene, camera: Camera): void;
  dispose(): void;
  forceContextLoss?(): void;
}

export interface ControlsLike {
  target: Vector3;
  enabled: boolean;
  update(): unknown;
  dispose(): void;
}

export interface TwinEngineOptions {
  config?: VisualConfig;
  createRenderer?: () => RendererLike;
  createControls?: (camera: PerspectiveCamera, domElement: HTMLElement) => ControlsLike;
  /** null disables labels entirely; default draws them on a canvas. */
  labelTextureFactory?: LabelTextureFactory | null;
  /** Built-in DOM tooltip on hover (default true). */
  tooltip?: boolean;
  /** FPS counter overlay (default false). */
  fps?: boolean;
  /** Frame clock in ms (default performance.now). */
  now?: () => number;
  /** Wall clock in ms for reading ages (default Date.now). */
  wallClock?: () => number;
  requestFrame?: (cb: (t: number) => void) => number;
  cancelFrame?: (id: number) => void;
  /** Start the frame loop immediately (default true). */
  autoStart?: boolean;
  devicePixelRatio?: number;
}

/** Anything snapshot-shaped: the WS snapshot or a /replay LineState. */
export interface SnapshotLike {
  machines?: Record<string, MachineView>;
  line?: LineViewModel | null;
  dataQuality?: Record<string, DataQualityReport>;
  alerts?: Alert[];
  predictions?: Record<string, Record<string, Prediction>>;
}

interface MachineData {
  cfg: LineMachine;
  state: MachineState | null;
  reason: string | null;
  reading: Reading | null;
  dataQuality: DataQualityReport | null;
  alerts: Map<string, Alert>;
  predictions: Map<string, Prediction>;
  rootCause: RootCauseView | null;
}

const MAX_DT = 0.1;
const NO_ITEMS: readonly (MachineState | null)[] = [null, "STARVED", "STALE", "OFFLINE"];

export class TwinEngine {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly cfg: VisualConfig;

  private readonly container: HTMLElement;
  private readonly renderer: RendererLike;
  private readonly controls: ControlsLike;
  private readonly labelFactory: LabelTextureFactory | null;
  private readonly now: () => number;
  private readonly wallClock: () => number;
  private readonly requestFrame: ((cb: (t: number) => void) => number) | null;
  private readonly cancelFrame: ((id: number) => void) | null;

  private readonly shared = new ResourceTracker();
  private layoutTracker = new ResourceTracker();
  private readonly listeners = new ListenerRegistry();
  private readonly events: { [K in keyof EngineEvents]: Set<EngineEvents[K]> } = {
    select: new Set(),
    hover: new Set(),
  };
  private readonly cache: GeometryCache;
  private readonly beltTexture;
  private readonly lights: { hemi: HemisphereLight; sun: DirectionalLight };
  private readonly raycaster = new Raycaster();
  private readonly tooltipView: TooltipView | null;
  private readonly fpsMeter: FpsMeter | null;

  private line: LineResponse | null = null;
  private layout: LineLayout | null = null;
  private world = new Group();
  private readonly machines = new Map<string, MachineModel>();
  private readonly buffers = new Map<string, BufferModel>();
  private items: ItemFlow | null = null;
  private readonly data = new Map<string, MachineData>();
  private readonly dirty = new Set<string>();
  private pickables: Object3D[] = [];
  private modeValue: EngineMode = "live";
  private connected = true;
  private selectedId: string | null = null;
  private hovered: PickTarget | null = null;
  private pointer: { x: number; y: number } | null = null;
  private frameId: number | null = null;
  private lastFrame: number | null = null;
  private disposed = false;

  constructor(container: HTMLElement, opts: TwinEngineOptions = {}) {
    this.container = container;
    this.cfg = opts.config ?? DEFAULT_CONFIG;
    this.now = opts.now ?? (() => performance.now());
    this.wallClock = opts.wallClock ?? (() => Date.now());
    this.requestFrame =
      opts.requestFrame ?? (typeof requestAnimationFrame === "function" ? (cb) => requestAnimationFrame(cb) : null);
    this.cancelFrame =
      opts.cancelFrame ?? (typeof cancelAnimationFrame === "function" ? (id) => { cancelAnimationFrame(id); } : null);
    this.labelFactory = opts.labelTextureFactory === undefined ? canvasLabelFactory : opts.labelTextureFactory;

    this.renderer = (opts.createRenderer ?? defaultRenderer)();
    this.renderer.setPixelRatio(
      Math.min(opts.devicePixelRatio ?? (typeof devicePixelRatio === "number" ? devicePixelRatio : 1), this.cfg.render.maxPixelRatio),
    );
    container.appendChild(this.renderer.domElement);

    this.camera = new PerspectiveCamera(this.cfg.render.fov, 1, 0.1, 500);
    this.camera.position.set(-6, 8, 12);
    this.controls = (opts.createControls ?? defaultControls)(this.camera, this.renderer.domElement);

    this.scene.background = new Color(this.cfg.colors.background);
    const hemi = new HemisphereLight(0xffffff, 0x404850, 1.6);
    const sun = new DirectionalLight(0xffffff, 2.2);
    sun.position.set(-8, 14, 10);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    this.scene.add(hemi, sun, sun.target);
    this.shared.track(sun);
    this.lights = { hemi, sun };
    this.scene.add(this.world);

    this.cache = new GeometryCache(this.shared);
    this.beltTexture = makeBeltTexture(this.cfg, this.shared);

    this.tooltipView = opts.tooltip === false ? null : TooltipView.create(container);
    this.fpsMeter = opts.fps === true ? new FpsMeter(container) : null;

    this.listeners.add(this.renderer.domElement, "pointermove", (e: PointerEvent) => {
      this.onPointerMove(e);
    });
    this.listeners.add(this.renderer.domElement, "pointerleave", () => {
      this.pointer = null;
      this.setHover(null);
    });
    this.listeners.add(this.renderer.domElement, "pointerdown", (e: PointerEvent) => {
      this.downAt = { x: e.clientX, y: e.clientY };
    });
    this.listeners.add(this.renderer.domElement, "pointerup", (e: PointerEvent) => {
      this.onPointerUp(e);
    });
    if (typeof ResizeObserver === "function") {
      const ro = new ResizeObserver(() => {
        this.resize();
      });
      ro.observe(container);
      this.listeners.addCleanup(() => {
        ro.disconnect();
      });
    }
    this.resize();
    if (opts.autoStart !== false) this.start();
  }

  private downAt: { x: number; y: number } | null = null;

  // ------------------------------------------------------------------ public API
  get mode(): EngineMode {
    return this.modeValue;
  }

  /** Build the scene for a line (replaces the previous one; known machine data is kept). */
  loadLayout(line: LineResponse): void {
    this.assertAlive();
    this.clearScene();
    this.line = line;
    const layout = computeLayout(line, this.cfg.layout);
    this.layout = layout;
    const ctx = this.buildContext();

    const ids = new Set(line.machines.map((m) => m.id));
    for (const id of [...this.data.keys()]) if (!ids.has(id)) this.data.delete(id);
    for (const m of line.machines) {
      const existing = this.data.get(m.id);
      this.data.set(m.id, existing === undefined ? emptyData(m) : { ...existing, cfg: m });
      const placement = layout.machines.get(m.id);
      if (placement === undefined) continue;
      const built = buildMachine(ctx, m, footprintOf(m.kind, this.cfg.layout));
      const model = new MachineModel(built, placement, ctx, this.labelFactory);
      this.machines.set(m.id, model);
      this.world.add(model.root);
      this.dirty.add(m.id);
    }
    for (const b of layout.buffers) {
      const model = new BufferModel(b, ctx, this.labelFactory);
      this.buffers.set(b.id, model);
      this.world.add(model.root);
    }
    this.items = new ItemFlow(ctx, [...this.machines.values()]);
    if (this.items.mesh !== null) this.world.add(this.items.mesh);
    this.pickables = [
      ...[...this.machines.values()].flatMap((m) => m.pickables),
      ...[...this.buffers.values()].flatMap((b) => b.pickables),
    ];
    if (this.selectedId !== null && !this.machines.has(this.selectedId)) this.selectedId = null;
    this.machines.get(this.selectedId ?? "")?.setSelected(true);

    this.addFloor(ctx, layout);
    this.renderer.shadowMap.enabled = line.machines.length <= this.cfg.render.shadowMaxMachines;
    this.lights.sun.castShadow = this.renderer.shadowMap.enabled;
    this.frame(layout);
    this.refreshBuffers();
  }

  /** Apply a full state (WS snapshot or /replay result). */
  applySnapshot(snapshot: SnapshotLike): void {
    for (const [id, d] of this.data) {
      const view = snapshot.machines?.[id];
      if (view !== undefined) {
        d.state = view.state ?? null;
        d.reading = view.lastReading ?? null;
        d.reason = view.lastStateChange?.reason ?? null;
      }
      const dq = snapshot.dataQuality?.[id];
      if (dq !== undefined) d.dataQuality = dq;
      const rc = snapshot.line?.rootCauses[id];
      if (snapshot.line) d.rootCause = rc ?? null;
      const preds = snapshot.predictions?.[id];
      if (preds !== undefined) d.predictions = new Map(Object.entries(preds));
      if (snapshot.alerts !== undefined) d.alerts.clear();
      this.dirty.add(id);
    }
    for (const a of snapshot.alerts ?? []) this.applyAlert(a);
  }

  applyReading(reading: Reading): void {
    const d = this.data.get(reading.machineId);
    if (d === undefined) return;
    if (d.reading !== null && Date.parse(reading.ts) < Date.parse(d.reading.ts)) return;
    d.reading = reading;
    if (reading.quality !== "LATE" && reading.state !== d.state) {
      // The same fold as the backend's live state: a reading carries the machine's state.
      d.state = reading.state;
      d.reason = null;
    }
    this.dirty.add(reading.machineId);
  }

  applyStateChange(change: StateChange): void {
    const d = this.data.get(change.machineId);
    if (d === undefined) return;
    d.state = change.state;
    d.reason = change.reason ?? null;
    this.dirty.add(change.machineId);
  }

  applyDataQuality(report: DataQualityReport): void {
    const d = this.data.get(report.machineId);
    if (d === undefined) return;
    d.dataQuality = report;
    this.dirty.add(report.machineId);
  }

  /** One alert lifecycle event; resolved alerts (cleared and acknowledged) disappear. */
  applyAlert(alert: Alert): void {
    const d = this.data.get(alert.machineId);
    if (d === undefined) return;
    const current = d.alerts.get(alert.alertId);
    if (current !== undefined && current.rev > alert.rev) return;
    if (!alert.active && alert.acknowledged) d.alerts.delete(alert.alertId);
    else d.alerts.set(alert.alertId, alert);
    this.dirty.add(alert.machineId);
  }

  applyLine(view: LineViewModel): void {
    for (const [id, d] of this.data) {
      d.rootCause = view.rootCauses[id] ?? null;
      this.dirty.add(id);
    }
  }

  setPrediction(prediction: Prediction): void {
    const d = this.data.get(prediction.machineId);
    if (d === undefined) return;
    d.predictions.set(String(prediction.horizonS), prediction);
    this.dirty.add(prediction.machineId);
  }

  /** Emphasise parts (intensity 0..1, 0 or an empty list clears). Returns unknown part names. */
  highlightParts(machineId: string, parts: readonly { part: string; intensity: number }[]): string[] {
    const model = this.machines.get(machineId);
    if (model === undefined) return parts.map((p) => p.part);
    const map = new Map<string, number>();
    const unknown: string[] = [];
    for (const { part, intensity } of parts) {
      if (!model.parts.has(part)) {
        unknown.push(part);
        continue;
      }
      const v = Math.min(1, Math.max(0, Number.isFinite(intensity) ? intensity : 0));
      if (v > 0) map.set(part, v);
    }
    model.setHighlights(map);
    return unknown;
  }

  setMode(mode: EngineMode): void {
    this.modeValue = mode;
    const bg = mode === "replay" ? this.cfg.colors.replayBackground : this.cfg.colors.background;
    (this.scene.background as Color).set(bg);
    this.container.dataset.twinMode = mode;
  }

  /** The stream is up (true) or down (false: every machine shows "link lost"). */
  setConnected(connected: boolean): void {
    if (connected === this.connected) return;
    this.connected = connected;
    for (const id of this.data.keys()) this.dirty.add(id);
  }

  select(machineId: string | null): void {
    const id = machineId !== null && this.machines.has(machineId) ? machineId : null;
    if (id === this.selectedId) return;
    this.machines.get(this.selectedId ?? "")?.setSelected(false);
    this.selectedId = id;
    this.machines.get(id ?? "")?.setSelected(true);
    for (const cb of this.events.select) cb({ machineId: id });
  }

  get selected(): string | null {
    return this.selectedId;
  }

  on<K extends keyof EngineEvents>(event: K, cb: EngineEvents[K]): () => void {
    const set = this.events[event] as Set<EngineEvents[K]>;
    set.add(cb);
    return () => set.delete(cb);
  }

  resize(width?: number, height?: number): void {
    if (this.disposed) return;
    const w = Math.max(1, Math.floor(width ?? this.container.clientWidth));
    const h = Math.max(1, Math.floor(height ?? this.container.clientHeight));
    this.renderer.setSize(w, h, true);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  start(): void {
    if (this.disposed || this.frameId !== null || this.requestFrame === null) return;
    const loop = (): void => {
      this.frameId = this.requestFrame?.(loop) ?? null;
      this.tick(this.now());
    };
    this.frameId = this.requestFrame(loop);
  }

  stop(): void {
    if (this.frameId !== null) this.cancelFrame?.(this.frameId);
    this.frameId = null;
  }

  /** Advance and render one frame at `nowMs` (the loop calls it; tests may call it directly). */
  tick(nowMs: number): void {
    if (this.disposed) return;
    const dt = this.lastFrame === null ? 0 : Math.min(MAX_DT, Math.max(0, (nowMs - this.lastFrame) / 1000));
    this.lastFrame = nowMs;
    if (this.dirty.size > 0) {
      for (const id of this.dirty) this.refreshMachine(id);
      this.dirty.clear();
      this.refreshBuffers();
    }
    for (const m of this.machines.values()) m.update(dt);
    for (const b of this.buffers.values()) b.update(dt, this.machines.get(b.placement.from)?.rate ?? 0);
    const { itemSpacing, itemSize } = this.cfg.conveyor;
    this.items?.update(dt, itemSpacing, itemSize, this.cfg.beltSpeed, (m) => this.itemsVisible(m.id));
    if (this.pointer !== null) this.updateHover();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.fpsMeter?.frame(nowMs);
  }

  /** Free everything: frame loop, listeners, controls, GPU resources, DOM. Safe to call twice. */
  dispose(): void {
    if (this.disposed) return;
    this.stop();
    this.disposed = true;
    this.listeners.removeAll();
    this.controls.dispose();
    this.clearScene();
    this.shared.disposeAll();
    this.renderer.dispose();
    this.renderer.forceContextLoss?.();
    this.renderer.domElement.remove();
    this.tooltipView?.dispose();
    this.fpsMeter?.dispose();
    this.events.select.clear();
    this.events.hover.clear();
    this.data.clear();
  }

  // ------------------------------------------------------------------ diagnostics (tests, dev page)
  /** Live counts of what the engine still holds. */
  resourceCounts(): { geometries: number; materials: number; textures: number; other: number; listeners: number; handlers: number; frames: number } {
    const a = this.shared.counts();
    const b = this.layoutTracker.counts();
    return {
      geometries: a.geometry + b.geometry,
      materials: a.material + b.material,
      textures: a.texture + b.texture,
      other: a.other + b.other,
      listeners: this.listeners.count,
      handlers: this.events.select.size + this.events.hover.size,
      frames: this.frameId === null ? 0 : 1,
    };
  }

  machineModel(id: string): MachineModel | undefined {
    return this.machines.get(id);
  }

  bufferModel(id: string): BufferModel | undefined {
    return this.buffers.get(id);
  }

  get itemFlow(): ItemFlow | null {
    return this.items;
  }

  get currentLayout(): LineLayout | null {
    return this.layout;
  }

  /** Screen position (px, container-relative) of a machine's label anchor, or null. */
  screenPosition(machineId: string): { x: number; y: number } | null {
    const m = this.machines.get(machineId);
    if (m === undefined) return null;
    this.scene.updateMatrixWorld();
    this.camera.updateMatrixWorld();
    const p = new Vector3(0, m.height * 0.6, 0);
    m.root.localToWorld(p);
    p.project(this.camera);
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    return { x: ((p.x + 1) / 2) * w, y: ((1 - p.y) / 2) * h };
  }

  /** Point the camera at one machine (dev page framings). */
  focus(machineId: string, distance = 8): void {
    const m = this.machines.get(machineId);
    if (m === undefined) return;
    const target = m.root.position.clone().add(new Vector3(0, m.height * 0.6, 0));
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(new Vector3(-0.55 * distance, 0.5 * distance, 0.8 * distance));
    this.camera.lookAt(target);
    this.controls.update();
  }

  /** The machine or buffer under a container-relative point, if any. */
  pickAt(x: number, y: number): PickTarget | null {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    const ndc = new Vector2((x / w) * 2 - 1, -(y / h) * 2 + 1);
    this.scene.updateMatrixWorld();
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(ndc, this.camera);
    return pickTarget(this.raycaster, this.pickables);
  }

  // ------------------------------------------------------------------ internals
  private visualInput(d: MachineData): VisualInput {
    const latest = [...d.predictions.values()].sort((a, b) => Date.parse(b.ts) - Date.parse(a.ts))[0];
    return {
      state: d.state,
      rootCause: d.rootCause,
      reason: d.reason,
      dataQuality: d.dataQuality?.status ?? null,
      alerts: [...d.alerts.values()],
      attributions: latest?.attributions ?? null,
      parts: d.cfg.parts,
      connected: this.connected,
    };
  }

  private refreshMachine(id: string): void {
    const d = this.data.get(id);
    const model = this.machines.get(id);
    if (d === undefined || model === undefined) return;
    model.applyVisual(machineVisual(this.visualInput(d), this.cfg));
    const rate = animationRate(d.reading?.speed, d.cfg.kind, this.cfg);
    if (rate !== null) model.targetRate = rate;
  }

  private refreshBuffers(): void {
    for (const b of this.buffers.values()) {
      const from = this.connected ? (this.data.get(b.placement.from)?.state ?? null) : null;
      const to = this.connected ? (this.data.get(b.placement.to)?.state ?? null) : null;
      const s = bufferStatus(b.placement.from, from, b.placement.to, to);
      b.setStatus(s.status, s.why);
    }
  }

  private itemsVisible(id: string): boolean {
    const state = this.data.get(id)?.state ?? null;
    return this.connected && !NO_ITEMS.includes(state);
  }

  private buildContext(): BuildContext {
    const pickMaterial = this.layoutTracker.track(new MeshBasicMaterial({ visible: false }));
    return { cfg: this.cfg, cache: this.cache, tracker: this.layoutTracker, beltTexture: this.beltTexture, pickMaterial };
  }

  private addFloor(ctx: BuildContext, layout: LineLayout): void {
    const { minX, maxX, minZ, maxZ } = layout.bounds;
    const margin = 4;
    const sx = maxX - minX + margin * 2;
    const sz = maxZ - minZ + margin * 2;
    const floorMat = ctx.tracker.track(new MeshStandardMaterial({ color: this.cfg.colors.floor, roughness: 0.95 }));
    const floor = new Mesh(this.cache.plane, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.scale.set(sx, sz, 1);
    floor.position.set((minX + maxX) / 2, 0, (minZ + maxZ) / 2);
    floor.receiveShadow = true;
    floor.userData.noPick = true;
    const size = Math.ceil(Math.max(sx, sz));
    const grid = new GridHelper(size, size, this.cfg.colors.grid, this.cfg.colors.grid);
    ctx.tracker.track(grid.geometry);
    const gridMats = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const m of gridMats) ctx.tracker.track(m);
    grid.position.set(floor.position.x, 0.005, floor.position.z);
    grid.userData.noPick = true;
    this.world.add(floor, grid);
  }

  private frame(layout: LineLayout): void {
    const { minX, maxX, minZ, maxZ } = layout.bounds;
    const cx = (minX + maxX) / 2;
    const cz = (minZ + maxZ) / 2;
    const extent = Math.max(maxX - minX, maxZ - minZ, 4);
    // Distance that fits the footprint into both the vertical and the horizontal field of view.
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
    const tanH = tanV * Math.max(this.camera.aspect, 0.1);
    const fit = Math.max((maxX - minX) / 2 / tanH, (maxZ - minZ) / 2 / tanV, 4);
    const d = fit * 1.15 + 2;
    this.controls.target.set(cx, 0.8, cz);
    this.camera.position.set(cx - d * 0.05, d * 0.5, cz + d * 0.9);
    this.camera.far = Math.max(500, d * 10);
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(this.controls.target);
    const sun = this.lights.sun;
    sun.position.set(cx - extent * 0.4, extent * 0.9 + 8, cz + extent * 0.6);
    sun.target.position.set(cx, 0, cz);
    const half = extent / 2 + 3;
    const cam = sun.shadow.camera;
    cam.left = -half;
    cam.right = half;
    cam.top = half;
    cam.bottom = -half;
    cam.far = extent * 3 + 40;
    cam.updateProjectionMatrix();
    this.controls.update();
  }

  private clearScene(): void {
    this.scene.remove(this.world);
    this.world = new Group();
    this.scene.add(this.world);
    this.machines.clear();
    this.buffers.clear();
    this.items = null;
    this.pickables = [];
    this.hovered = null;
    this.layoutTracker.disposeAll();
    this.layoutTracker = new ResourceTracker();
  }

  // ------------------------------------------------------------------ pointer
  private onPointerMove(e: PointerEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  private onPointerUp(e: PointerEvent): void {
    const down = this.downAt;
    this.downAt = null;
    // A drag (orbiting) is not a click.
    if (down !== null && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const target = this.pickAt(e.clientX - rect.left, e.clientY - rect.top);
    this.select(target?.kind === "machine" ? target.id : null);
  }

  private updateHover(): void {
    const p = this.pointer;
    if (p === null) return;
    this.setHover(this.pickAt(p.x, p.y));
  }

  /** Tooltip for a machine or buffer (also used by the dev page). */
  tooltipFor(target: PickTarget): TooltipModel | null {
    if (target.kind === "machine") {
      const d = this.data.get(target.id);
      const line = this.line;
      if (d === undefined || line === null) return null;
      return machineTooltip(
        {
          machine: d.cfg,
          stateLabel: stateLabel(this.visualInput(d)),
          reading: d.reading,
          sensors: line.sensors,
          dataQuality: d.dataQuality,
          alerts: [...d.alerts.values()],
          nowMs: this.wallClock(),
        },
        this.cfg,
      );
    }
    const b = this.buffers.get(target.id);
    if (b === undefined) return null;
    return bufferTooltip(b.placement, { status: b.status, why: b.why });
  }

  private setHover(target: PickTarget | null): void {
    const same =
      (target === null && this.hovered === null) ||
      (target !== null && this.hovered !== null && target.kind === this.hovered.kind && target.id === this.hovered.id);
    this.hovered = target;
    const tooltip = target === null ? null : this.tooltipFor(target);
    const p = this.pointer ?? { x: 0, y: 0 };
    this.tooltipView?.show(tooltip, p.x, p.y);
    if (!same) for (const cb of this.events.hover) cb({ target, tooltip, x: p.x, y: p.y });
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error("TwinEngine is disposed");
  }
}

function emptyData(cfg: LineMachine): MachineData {
  return {
    cfg,
    state: null,
    reason: null,
    reading: null,
    dataQuality: null,
    alerts: new Map(),
    predictions: new Map(),
    rootCause: null,
  };
}

function defaultRenderer(): RendererLike {
  const r = new WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  r.shadowMap.type = PCFShadowMap;
  r.toneMapping = ACESFilmicToneMapping;
  r.domElement.style.display = "block";
  r.domElement.style.width = "100%";
  r.domElement.style.height = "100%";
  return r;
}

function defaultControls(camera: PerspectiveCamera, domElement: HTMLElement): ControlsLike {
  const c = new OrbitControls(camera, domElement);
  c.enableDamping = true;
  c.dampingFactor = 0.08;
  c.maxPolarAngle = Math.PI * 0.49;
  c.minDistance = 2;
  c.maxDistance = 200;
  return c;
}

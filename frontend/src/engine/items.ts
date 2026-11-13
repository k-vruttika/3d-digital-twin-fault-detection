/**
 * Parts riding on conveyors: one InstancedMesh for the whole line. Each conveyor is a lane
 * whose parts advance with the belt, so their motion follows the reported speed; a STARVED,
 * stale or offline conveyor shows no parts (nothing is known to be on it).
 */
import { InstancedMesh, Matrix4, MeshStandardMaterial, Quaternion, Vector3 } from "three";

import type { BuildContext } from "./geometry/shared.ts";
import type { MachineModel } from "./machineModel.ts";

const HIDDEN = new Matrix4().makeScale(0, 0, 0);

interface Lane {
  machine: MachineModel;
  start: number;
  count: number;
  length: number;
  top: number;
  phase: number;
  visible: boolean;
  dirty: boolean;
}

export class ItemFlow {
  readonly mesh: InstancedMesh | null;
  private readonly lanes: Lane[] = [];

  constructor(ctx: BuildContext, machines: readonly MachineModel[]) {
    const { cfg, cache, tracker } = ctx;
    let total = 0;
    for (const m of machines) {
      if (m.belt === null) continue;
      const count = Math.max(1, Math.floor(m.belt.length / cfg.conveyor.itemSpacing));
      this.lanes.push({
        machine: m,
        start: total,
        count,
        length: m.belt.length,
        top: m.belt.top,
        phase: 0,
        visible: false,
        dirty: true,
      });
      total += count;
    }
    if (total === 0) {
      this.mesh = null;
      return;
    }
    const mat = tracker.track(new MeshStandardMaterial({ color: cfg.colors.item, roughness: 0.8 }));
    this.mesh = tracker.track(new InstancedMesh(cache.box, mat, total));
    this.mesh.name = "items";
    this.mesh.castShadow = true;
    this.mesh.userData.noPick = true;
    for (let i = 0; i < total; i++) this.mesh.setMatrixAt(i, HIDDEN);
  }

  /** Visible parts per conveyor (tests). */
  visibleCount(machineId: string): number {
    const lane = this.lanes.find((l) => l.machine.id === machineId);
    return lane?.visible === true ? lane.count : 0;
  }

  update(dt: number, spacing: number, size: number, beltSpeed: number, visibleFor: (m: MachineModel) => boolean): void {
    const mesh = this.mesh;
    if (mesh === null) return;
    const m = new Matrix4();
    const p = new Vector3();
    const q = new Quaternion();
    const s = new Vector3(size, size, size);
    let changed = false;
    for (const lane of this.lanes) {
      const visible = visibleFor(lane.machine);
      const moving = visible && lane.machine.rate > 0;
      if (visible !== lane.visible) {
        lane.visible = visible;
        lane.dirty = true;
      }
      if (moving) {
        lane.phase = (lane.phase + dt * lane.machine.rate * beltSpeed) % spacing;
        lane.dirty = true;
      }
      if (!lane.dirty) continue;
      lane.dirty = false;
      changed = true;
      const root = lane.machine.root;
      root.updateMatrixWorld();
      q.setFromRotationMatrix(root.matrixWorld);
      for (let i = 0; i < lane.count; i++) {
        if (!lane.visible) {
          mesh.setMatrixAt(lane.start + i, HIDDEN);
          continue;
        }
        const x = -lane.length / 2 + ((lane.phase + i * spacing) % lane.length);
        p.set(x, lane.top + size / 2, 0).applyMatrix4(root.matrixWorld);
        mesh.setMatrixAt(lane.start + i, m.compose(p, q, s));
      }
    }
    if (changed) mesh.instanceMatrix.needsUpdate = true;
  }
}

/** Ray picking of machines and buffers (works headless: no DOM involved). */
import type { Object3D, Raycaster } from "three";

export type PickTarget = { kind: "machine"; id: string } | { kind: "buffer"; id: string };

/** The machine or buffer owning an object (walks up the parents), or null. */
export function ownerOf(object: Object3D | null): PickTarget | null {
  let o: Object3D | null = object;
  while (o !== null) {
    const ud = o.userData as { machineId?: unknown; bufferId?: unknown };
    if (typeof ud.machineId === "string") return { kind: "machine", id: ud.machineId };
    if (typeof ud.bufferId === "string") return { kind: "buffer", id: ud.bufferId };
    o = o.parent;
  }
  return null;
}

/** The nearest pickable hit; outlines, labels and other indicators are skipped. */
export function pickTarget(raycaster: Raycaster, objects: readonly Object3D[]): PickTarget | null {
  const hits = raycaster.intersectObjects(objects as Object3D[], false);
  for (const hit of hits) {
    if ((hit.object.userData as { noPick?: unknown }).noPick === true) continue;
    if (!hit.object.visible) continue;
    const owner = ownerOf(hit.object);
    if (owner !== null) return owner;
  }
  return null;
}

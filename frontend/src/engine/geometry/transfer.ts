/**
 * Transfer track: the visible connection where the line turns into the next row (or wherever an
 * override puts two linked machines out of line). A low track along the link path with flow
 * chevrons, so it is clear which row continues into which (layout.ts `linkPath`).
 */
import { Group, Mesh, MeshStandardMaterial } from "three";

import type { BufferPlacement } from "../layout.ts";
import type { BuildContext } from "./shared.ts";

/** Chevron arm length (m) and the angle of each arm to the flow direction. */
const CHEVRON_ARM = 0.26;
const CHEVRON_ANGLE = Math.PI / 5;

export function buildTransfer(ctx: BuildContext, link: BufferPlacement): Group {
  const { cfg, cache, tracker } = ctx;
  const t = cfg.transfer;
  const group = new Group();
  group.name = `transfer:${link.id}`;
  group.userData.noPick = true;
  const trackMat = tracker.track(new MeshStandardMaterial({ color: cfg.colors.belt, roughness: 0.85 }));
  const railMat = tracker.track(new MeshStandardMaterial({ color: cfg.colors.frame, metalness: 0.5, roughness: 0.5 }));
  const chevronMat = tracker.track(
    new MeshStandardMaterial({ color: cfg.colors.beltStripe, emissive: cfg.colors.beltStripe, emissiveIntensity: 0.35 }),
  );
  for (let i = 1; i < link.path.length; i++) {
    const a = link.path[i - 1];
    const b = link.path[i];
    if (a === undefined || b === undefined) continue;
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1e-6) continue;
    const seg = new Group();
    seg.position.set((a[0] + b[0]) / 2, 0, (a[1] + b[1]) / 2);
    seg.rotation.y = Math.atan2(-(b[1] - a[1]), b[0] - a[0]);
    // Extend each segment by half the width so the corners close.
    const full = len + t.width;
    const track = new Mesh(cache.box, trackMat);
    track.scale.set(full, t.height, t.width);
    track.position.y = t.height / 2;
    track.receiveShadow = true;
    seg.add(track);
    for (const side of [-1, 1]) {
      const rail = new Mesh(cache.box, railMat);
      rail.scale.set(full, t.height * 2, t.railWidth);
      rail.position.set(0, t.height, side * (t.width / 2 - t.railWidth / 2));
      seg.add(rail);
    }
    const n = Math.max(1, Math.floor(len / t.chevronSpacing));
    for (let k = 0; k < n; k++) {
      const x = -len / 2 + ((k + 0.5) * len) / n;
      for (const side of [-1, 1]) {
        const arm = new Mesh(cache.box, chevronMat);
        arm.scale.set(CHEVRON_ARM, t.height * 0.6, t.railWidth);
        arm.rotation.y = side * CHEVRON_ANGLE;
        // the arms meet at the tip, pointing along the flow (+x of the segment)
        const back = (CHEVRON_ARM / 2) * Math.cos(CHEVRON_ANGLE);
        const off = (CHEVRON_ARM / 2) * Math.sin(CHEVRON_ANGLE);
        arm.position.set(x - back, t.height * 1.15, side * off);
        seg.add(arm);
      }
    }
    group.add(seg);
  }
  group.traverse((o) => (o.userData.noPick = true));
  return group;
}

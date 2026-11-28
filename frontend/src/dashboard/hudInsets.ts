/**
 * How far HUD elements cover the 3D canvas from its top and bottom edge (DESIGN.md §5): the
 * annunciator floats over the sky, the line strip over the foreground. An element that does
 * not overlap the canvas (the docked strip below 1024 px) contributes nothing.
 */
export interface Box {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

/** Breathing room between a HUD element and the framed line (px). */
export const HUD_GAP_PX = 16;

export function hudInsets(canvas: Box, top: readonly Box[], bottom: readonly Box[]): { top: number; right: number; bottom: number; left: number } {
  const overlaps = (b: Box) => b.bottom > canvas.top && b.top < canvas.bottom && b.right > canvas.left && b.left < canvas.right;
  const h = canvas.bottom - canvas.top;
  let t = 0;
  for (const b of top) if (overlaps(b)) t = Math.max(t, b.bottom - canvas.top + HUD_GAP_PX);
  let btm = 0;
  for (const b of bottom) if (overlaps(b)) btm = Math.max(btm, canvas.bottom - b.top + HUD_GAP_PX);
  // never hand the engine less than a third of the canvas
  const limit = (h * 2) / 3;
  if (t + btm > limit) {
    const k = limit / (t + btm);
    t *= k;
    btm *= k;
  }
  return { top: Math.round(t), right: 0, bottom: Math.round(btm), left: 0 };
}

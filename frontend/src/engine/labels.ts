/** Text label textures for camera-facing state labels (canvas in the browser, injectable). */
import { CanvasTexture, SRGBColorSpace, type Texture } from "three";

export interface LabelStyle {
  text: number;
  background: number;
  /** Bar on the left in the tone's colour (the label never relies on it alone). */
  accent: number;
  fontPx: number;
}

export interface LabelTexture {
  texture: Texture;
  /** width / height */
  aspect: number;
}

/** Returns null when no texture can be made (no DOM): the label is then not shown. */
export type LabelTextureFactory = (text: string, style: LabelStyle) => LabelTexture | null;

const hex = (c: number): string => `#${c.toString(16).padStart(6, "0")}`;

export const canvasLabelFactory: LabelTextureFactory = (text, style) => {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (ctx === null) return null;
  const font = `600 ${String(style.fontPx)}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  ctx.font = font;
  const pad = Math.round(style.fontPx * 0.45);
  const bar = Math.round(style.fontPx * 0.22);
  const width = Math.ceil(ctx.measureText(text).width) + pad * 2 + bar;
  const height = Math.round(style.fontPx * 1.5);
  canvas.width = width;
  canvas.height = height;
  ctx.font = font; // resizing the canvas resets the context
  ctx.fillStyle = hex(style.background);
  ctx.globalAlpha = 0.82;
  ctx.fillRect(0, 0, width, height);
  ctx.globalAlpha = 1;
  ctx.fillStyle = hex(style.accent);
  ctx.fillRect(0, 0, bar, height);
  ctx.fillStyle = hex(style.text);
  ctx.textBaseline = "middle";
  ctx.fillText(text, bar + pad, height / 2 + 1);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return { texture, aspect: width / height };
};

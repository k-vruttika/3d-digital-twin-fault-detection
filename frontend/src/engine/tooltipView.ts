/** The engine's built-in hover tooltip (plain DOM, text only: no HTML from data is ever parsed). */
import type { TooltipModel } from "./tooltip.ts";

const STYLE: Partial<CSSStyleDeclaration> = {
  position: "absolute",
  pointerEvents: "none",
  zIndex: "10",
  minWidth: "180px",
  maxWidth: "320px",
  padding: "8px 10px",
  borderRadius: "6px",
  background: "rgba(20, 24, 29, 0.92)",
  color: "#e8edf2",
  font: "12px/1.45 system-ui, -apple-system, 'Segoe UI', sans-serif",
  boxShadow: "0 4px 14px rgba(0, 0, 0, 0.35)",
  display: "none",
};

export class TooltipView {
  private constructor(
    private readonly root: HTMLDivElement,
    private readonly container: HTMLElement,
  ) {}

  /** null when there is no DOM (tests, workers). */
  static create(container: HTMLElement): TooltipView | null {
    if (typeof document === "undefined") return null;
    const root = document.createElement("div");
    root.className = "twin-tooltip";
    Object.assign(root.style, STYLE);
    container.appendChild(root);
    return new TooltipView(root, container);
  }

  show(model: TooltipModel | null, x: number, y: number): void {
    if (model === null) {
      this.root.style.display = "none";
      return;
    }
    this.root.replaceChildren(...render(model));
    this.root.style.display = "block";
    const pad = 14;
    const w = this.root.offsetWidth;
    const h = this.root.offsetHeight;
    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;
    const left = x + pad + w > cw ? x - pad - w : x + pad;
    const top = y + pad + h > ch ? y - pad - h : y + pad;
    this.root.style.left = `${String(Math.max(0, left))}px`;
    this.root.style.top = `${String(Math.max(0, top))}px`;
  }

  dispose(): void {
    this.root.remove();
  }
}

function el(tag: string, text: string, style: Partial<CSSStyleDeclaration> = {}): HTMLElement {
  const e = document.createElement(tag);
  e.textContent = text;
  Object.assign(e.style, style);
  return e;
}

function render(model: TooltipModel): HTMLElement[] {
  const out = [
    el("div", model.title, { fontWeight: "600", fontSize: "13px" }),
    el("div", model.subtitle, { margin: "2px 0 6px", fontWeight: "600", letterSpacing: "0.02em" }),
  ];
  const table = document.createElement("table");
  table.style.borderCollapse = "collapse";
  for (const row of model.rows) {
    const tr = document.createElement("tr");
    tr.append(
      el("td", row.label, { opacity: "0.7", paddingRight: "12px" }),
      el("td", row.value, { textAlign: "right", fontVariantNumeric: "tabular-nums" }),
    );
    table.append(tr);
  }
  out.push(table);
  for (const note of model.notes) out.push(el("div", note, { marginTop: "6px", opacity: "0.85" }));
  return out;
}

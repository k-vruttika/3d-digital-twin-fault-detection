/** Frame-rate measurement (pure) and a small dev overlay. */

/** Frames per second over a sliding window of frame timestamps (ms). */
export class FpsCounter {
  private readonly times: number[] = [];

  constructor(private readonly windowMs = 1000) {}

  frame(nowMs: number): void {
    this.times.push(nowMs);
    const cutoff = nowMs - this.windowMs;
    while (this.times.length > 0 && (this.times[0] ?? nowMs) < cutoff) this.times.shift();
  }

  /** Frames per second, or 0 before two frames arrived. */
  get fps(): number {
    const n = this.times.length;
    if (n < 2) return 0;
    const first = this.times[0] ?? 0;
    const last = this.times[n - 1] ?? 0;
    return last > first ? ((n - 1) * 1000) / (last - first) : 0;
  }

  /** Longest gap between two frames in the window (ms): spikes show up here. */
  get worstFrameMs(): number {
    let worst = 0;
    for (let i = 1; i < this.times.length; i++) {
      worst = Math.max(worst, (this.times[i] ?? 0) - (this.times[i - 1] ?? 0));
    }
    return worst;
  }
}

export class FpsMeter {
  readonly counter = new FpsCounter();
  private readonly el: HTMLDivElement | null;
  private lastPaint = 0;

  constructor(container: HTMLElement, private readonly paintEveryMs = 500) {
    if (typeof document === "undefined") {
      this.el = null;
      return;
    }
    const el = document.createElement("div");
    el.className = "twin-fps";
    Object.assign(el.style, {
      position: "absolute",
      right: "8px",
      bottom: "8px",
      padding: "2px 6px",
      borderRadius: "4px",
      background: "rgba(0, 0, 0, 0.55)",
      color: "#9effa6",
      font: "11px/1.4 ui-monospace, Consolas, monospace",
      pointerEvents: "none",
      zIndex: "11",
    } satisfies Partial<CSSStyleDeclaration>);
    container.appendChild(el);
    this.el = el;
  }

  frame(nowMs: number): void {
    this.counter.frame(nowMs);
    if (this.el === null || nowMs - this.lastPaint < this.paintEveryMs) return;
    this.lastPaint = nowMs;
    this.el.textContent = `${this.counter.fps.toFixed(0)} fps · worst ${this.counter.worstFrameMs.toFixed(0)} ms`;
  }

  dispose(): void {
    this.el?.remove();
  }
}

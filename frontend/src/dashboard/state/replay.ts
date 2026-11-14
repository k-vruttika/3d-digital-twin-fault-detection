/**
 * Replay mode (unit tested with fake timers).
 *
 * Entering replay switches the model and engine to "replay" (live stream data is then ignored
 * by bindStream), loads the stored range from `/replay/range` and shows `/replay?t=` for the
 * chosen time: machines, root causes and alerts as of t. Play advances event time by
 * speed x `replayTickMs` every tick; only one request is in flight, later ticks just move the
 * target (coalesced). Returning to live resubscribes the stream, whose fresh snapshot replaces
 * the replayed state.
 */
import type { RestClient } from "../../api/rest.ts";
import type { DashboardConfig } from "../config.ts";
import type { DashboardModel } from "./model.ts";
import { createStore, type Store } from "./store.ts";

export interface ReplayState {
  active: boolean;
  start: number | null;
  end: number | null;
  /** Shown event time (ms) once loaded; the target while a request is in flight. */
  t: number | null;
  playing: boolean;
  speed: number;
  loading: boolean;
  error: string | null;
}

export const IDLE_REPLAY: ReplayState = {
  active: false,
  start: null,
  end: null,
  t: null,
  playing: false,
  speed: 1,
  loading: false,
  error: null,
};

export interface ReplayDeps {
  model: DashboardModel;
  rest: () => RestClient;
  /** Ask the live stream for a fresh snapshot (Session.resubscribe). */
  resubscribe: () => void;
  cfg: DashboardConfig;
}

export class ReplayController {
  readonly state: Store<ReplayState> = createStore(IDLE_REPLAY);
  private timer: ReturnType<typeof setInterval> | null = null;
  private inFlight = false;
  private pending: number | null = null;
  /** Requests made (tests check the coalescing). */
  requests = 0;

  constructor(private readonly deps: ReplayDeps) {}

  async enter(): Promise<void> {
    const { model } = this.deps;
    model.setMode("replay");
    this.state.set({ ...IDLE_REPLAY, active: true, loading: true, speed: this.state.get().speed });
    try {
      const r = await this.deps.rest().getReplayRange();
      if (!this.state.get().active) return;
      const start = r.start ? Date.parse(r.start) : null;
      const end = r.end ? Date.parse(r.end) : null;
      if (start === null || end === null) {
        this.state.update((s) => ({ ...s, loading: false, error: "Nothing is stored yet: there is no history to replay." }));
        return;
      }
      this.state.update((s) => ({ ...s, start, end }));
      const asOf = model.asOf.get();
      const t0 = asOf === null ? end : Math.min(end, Math.max(start, Date.parse(asOf)));
      this.seek(t0);
    } catch (e) {
      this.state.update((s) => ({ ...s, loading: false, error: `Replay unavailable: ${String(e)}` }));
    }
  }

  /** Show the line at event time `t` (clamped to the stored range). */
  seek(t: number): void {
    const s = this.state.get();
    if (!s.active) return;
    const clamped = s.start !== null && s.end !== null ? Math.min(s.end, Math.max(s.start, t)) : t;
    this.state.set({ ...s, t: clamped });
    this.pending = clamped;
    this.pump();
  }

  play(): void {
    const s = this.state.get();
    if (!s.active || s.playing || s.t === null) return;
    if (s.end !== null && s.t >= s.end && s.start !== null) this.seek(s.start); // replay again from the start
    this.state.update((x) => ({ ...x, playing: true }));
    this.timer = setInterval(() => {
      this.tick();
    }, this.deps.cfg.replayTickMs);
  }

  pause(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    this.state.update((s) => ({ ...s, playing: false }));
  }

  setSpeed(speed: number): void {
    this.state.update((s) => ({ ...s, speed }));
  }

  exit(): void {
    this.pause();
    this.pending = null;
    this.state.set({ ...IDLE_REPLAY, speed: this.state.get().speed });
    this.deps.model.setMode("live");
    this.deps.resubscribe();
  }

  dispose(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }

  private tick(): void {
    const s = this.state.get();
    if (s.t === null) return;
    const next = s.t + s.speed * this.deps.cfg.replayTickMs;
    if (s.end !== null && next >= s.end) {
      this.seek(s.end);
      this.pause();
      return;
    }
    this.seek(next);
  }

  private pump(): void {
    if (this.inFlight || this.pending === null) return;
    const t = this.pending;
    this.pending = null;
    this.inFlight = true;
    this.requests++;
    const iso = new Date(t).toISOString();
    this.state.update((s) => ({ ...s, loading: true }));
    this.deps
      .rest()
      .getReplay(iso)
      .then((r) => {
        if (this.state.get().active) {
          this.deps.model.applyReplay(r, iso);
          this.state.update((s) => ({ ...s, error: null }));
        }
      })
      .catch((e: unknown) => {
        this.state.update((s) => ({ ...s, error: `Replay failed: ${String(e)}` }));
      })
      .finally(() => {
        this.inFlight = false;
        this.state.update((s) => ({ ...s, loading: this.pending !== null }));
        this.pump();
      });
  }
}

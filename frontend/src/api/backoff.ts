/** Exponential reconnect backoff with seeded jitter (all randomness reproducible). */

export interface BackoffOptions {
  /** First delay in ms. */
  baseMs: number;
  /** Multiplier per attempt. */
  factor: number;
  /** Upper bound in ms. */
  maxMs: number;
  /** Fraction of the delay that is randomised (0 = none, 1 = full jitter). */
  jitter: number;
  /** Seed of the jitter generator. */
  seed: number;
}

export const DEFAULT_BACKOFF: BackoffOptions = {
  baseMs: 500,
  factor: 2,
  maxMs: 15_000,
  jitter: 0.3,
  seed: 1,
};

/** Small seeded PRNG (mulberry32): same seed, same sequence. Returns values in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Backoff {
  private attemptCount = 0;
  private readonly random: () => number;

  constructor(private readonly opts: BackoffOptions = DEFAULT_BACKOFF) {
    if (opts.baseMs <= 0 || opts.factor < 1 || opts.maxMs < opts.baseMs) {
      throw new Error("backoff needs baseMs > 0, factor >= 1 and maxMs >= baseMs");
    }
    if (opts.jitter < 0 || opts.jitter > 1) {
      throw new Error("backoff jitter must be within [0, 1]");
    }
    this.random = mulberry32(opts.seed);
  }

  /** Attempts since the last reset. */
  get attempt(): number {
    return this.attemptCount;
  }

  /** The delay before the next attempt; grows until maxMs. */
  next(): number {
    const { baseMs, factor, maxMs, jitter } = this.opts;
    const ceiling = Math.min(maxMs, baseMs * factor ** this.attemptCount);
    this.attemptCount += 1;
    return Math.round(ceiling * (1 - jitter) + ceiling * jitter * this.random());
  }

  reset(): void {
    this.attemptCount = 0;
  }
}

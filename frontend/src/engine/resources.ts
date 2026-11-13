/**
 * Bookkeeping for GPU-backed resources and DOM listeners, so `dispose()` provably frees
 * everything it created (counters are asserted in the tests).
 */

export interface Disposable {
  dispose(): void;
}

export type ResourceKind = "geometry" | "material" | "texture" | "other";

function kindOf(r: Disposable): ResourceKind {
  const flags = r as { isBufferGeometry?: boolean; isMaterial?: boolean; isTexture?: boolean };
  if (flags.isBufferGeometry === true) return "geometry";
  if (flags.isMaterial === true) return "material";
  if (flags.isTexture === true) return "texture";
  return "other";
}

export class ResourceTracker {
  private readonly items = new Map<Disposable, ResourceKind>();

  track<T extends Disposable>(resource: T): T {
    this.items.set(resource, kindOf(resource));
    return resource;
  }

  /** Dispose one resource now and stop tracking it. */
  release(resource: Disposable | null | undefined): void {
    if (resource === null || resource === undefined) return;
    if (this.items.delete(resource)) resource.dispose();
  }

  has(resource: Disposable): boolean {
    return this.items.has(resource);
  }

  counts(): Record<ResourceKind | "total", number> {
    const out = { geometry: 0, material: 0, texture: 0, other: 0, total: 0 };
    for (const kind of this.items.values()) {
      out[kind] += 1;
      out.total += 1;
    }
    return out;
  }

  disposeAll(): void {
    const items = [...this.items.keys()];
    this.items.clear();
    for (const r of items) r.dispose();
  }
}

/** The part of EventTarget the registry needs (DOM elements, window, test fakes). */
export interface ListenerTarget {
  addEventListener(type: string, listener: EventListener, options?: boolean | AddEventListenerOptions): void;
  removeEventListener(type: string, listener: EventListener, options?: boolean | EventListenerOptions): void;
}

/** Registers listeners (and other cleanups) so they can all be removed at once. */
export class ListenerRegistry {
  private readonly cleanups = new Set<() => void>();

  add(
    target: ListenerTarget,
    type: string,
    listener: (ev: never) => void,
    options?: AddEventListenerOptions,
  ): () => void {
    const fn = listener as EventListener;
    target.addEventListener(type, fn, options);
    return this.addCleanup(() => {
      target.removeEventListener(type, fn, options);
    });
  }

  /** Any other teardown (ResizeObserver.disconnect, ...); counted like a listener. */
  addCleanup(cleanup: () => void): () => void {
    let done = false;
    const wrapped = (): void => {
      if (done) return;
      done = true;
      this.cleanups.delete(wrapped);
      cleanup();
    };
    this.cleanups.add(wrapped);
    return wrapped;
  }

  get count(): number {
    return this.cleanups.size;
  }

  removeAll(): void {
    for (const cleanup of [...this.cleanups]) cleanup();
  }
}

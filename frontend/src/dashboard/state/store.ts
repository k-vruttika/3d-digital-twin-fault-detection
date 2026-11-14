/**
 * A minimal typed external store, read by React through `useSyncExternalStore` (ADR 0018).
 *
 * State is replaced, never mutated, so `getSnapshot` identity tells React when to re-render.
 * High-rate data (readings for charts) does not go through stores: charts subscribe to the
 * model's reading listeners directly.
 */
import { useSyncExternalStore } from "react";

export interface Store<T> {
  readonly get: () => T;
  readonly set: (next: T) => void;
  readonly update: (fn: (current: T) => T) => void;
  readonly subscribe: (listener: () => void) => () => void;
}

export function createStore<T>(initial: T): Store<T> {
  let value = initial;
  const listeners = new Set<() => void>();
  const set = (next: T): void => {
    if (Object.is(next, value)) return;
    value = next;
    for (const l of [...listeners]) l();
  };
  return {
    get: () => value,
    set,
    update: (fn) => {
      set(fn(value));
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

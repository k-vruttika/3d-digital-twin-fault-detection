/**
 * Hosts the 3D TwinEngine. The engine (and three.js) load as their own chunk on mount; until
 * then the panels already work. Selection syncs both ways with the model.
 */
import { useEffect, useRef, useState } from "react";

import { useServices } from "../context.ts";
import type { EngineHandle } from "../engineFactory.ts";
import { useStore } from "../state/store.ts";
import { ENGINE_THEME, themeStore, useTheme } from "../theme.ts";

export function TwinView() {
  const { model, engineFactory } = useServices();
  const ref = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<EngineHandle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ready = useStore(model.engineReady);
  const theme = useTheme();

  useEffect(() => {
    const el = ref.current;
    if (el === null) return;
    let disposed = false;
    let cleanup: (() => void) | null = null;
    engineFactory(el)
      .then((engine) => {
        if (disposed) {
          engine.dispose();
          return;
        }
        engineRef.current = engine;
        engine.setTheme(ENGINE_THEME[themeStore.get()]);
        const detach = model.attachEngine(engine);
        const off = engine.on("select", ({ machineId }) => {
          model.select(machineId);
        });
        if (import.meta.env.DEV) {
          (window as unknown as { __linetwin?: unknown }).__linetwin = { model, engine };
        }
        cleanup = () => {
          off();
          detach();
          engine.dispose();
          engineRef.current = null;
        };
      })
      .catch((e: unknown) => {
        setError(`The 3D view could not start: ${String(e)}`);
      });
    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [model, engineFactory]);

  useEffect(() => {
    engineRef.current?.setTheme(ENGINE_THEME[theme]);
  }, [theme]);

  return (
    <div className="twin" ref={ref} data-testid="twin-view" aria-label="3D view of the line" role="region">
      {!ready && error === null && <div className="twin-overlay">Loading 3D view…</div>}
      {error !== null && (
        <div className="twin-overlay error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}

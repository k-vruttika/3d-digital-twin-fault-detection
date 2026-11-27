/**
 * Hosts the 3D TwinEngine. The engine (and three.js) load as their own chunk on mount; until
 * then the panels already work. Selection syncs both ways with the model.
 */
import { useEffect, useRef, useState } from "react";

import { useServices } from "../context.ts";
import type { EngineHandle } from "../engineFactory.ts";
import { hudInsets } from "../hudInsets.ts";
import { useStore } from "../state/store.ts";
import { ENGINE_THEME, themeStore, useTheme } from "../theme.ts";

export function TwinView() {
  const { model, engineFactory } = useServices();
  const ref = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<EngineHandle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ready = useStore(model.engineReady);
  const theme = useTheme();
  const mode = useStore(model.modeStore);

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

  // Keep the framed line clear of the HUD (annunciator above, line strip below): measure how far
  // they overlap the canvas whenever any of them changes size, and tell the engine.
  useEffect(() => {
    const el = ref.current;
    const stage = el?.parentElement ?? null;
    if (el === null || stage === null || !ready || typeof ResizeObserver !== "function") return;
    let raf: number | null = null;
    let last = "";
    const measure = (): void => {
      raf = null;
      const box = (n: Element | null) => (n === null ? [] : [n.getBoundingClientRect()]);
      const insets = hudInsets(el.getBoundingClientRect(), box(stage.querySelector(".annunciator")), box(stage.querySelector(".hud-strip")));
      const key = JSON.stringify(insets);
      if (key === last) return;
      last = key;
      engineRef.current?.setViewInsets?.(insets);
    };
    const schedule = (): void => {
      if (raf === null) raf = requestAnimationFrame(measure);
    };
    const ro = new ResizeObserver(schedule);
    ro.observe(el);
    for (const n of stage.querySelectorAll(".annunciator, .hud-strip")) ro.observe(n);
    // the annunciator is re-rendered as alerts change; watch the stage for new HUD nodes
    const mo = new MutationObserver(() => {
      for (const n of stage.querySelectorAll(".annunciator, .hud-strip")) ro.observe(n);
      schedule();
    });
    mo.observe(stage, { childList: true, subtree: true });
    schedule();
    return () => {
      ro.disconnect();
      mo.disconnect();
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, [ready]);

  return (
    <div className="twin" ref={ref} data-testid="twin-view" data-mode={mode} aria-label="3D view of the line" role="region">
      {!ready && error === null && <div className="twin-overlay">Loading 3D view…</div>}
      {error !== null && (
        <div className="twin-overlay error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}

/** Polls GET /sources into the model (sources panel, scenario capability). */
import { useEffect } from "react";

import { useModel, useRest } from "./context.ts";
import { useStore } from "./state/store.ts";

export function useSourcesPolling(): void {
  const model = useModel();
  const rest = useRest();
  const auth = useStore(model.auth);
  const ok = auth.status === "ok";
  useEffect(() => {
    if (!ok) return;
    let cancelled = false;
    const poll = (): void => {
      rest
        .getSources()
        .then((s) => {
          if (!cancelled) model.sources.set(s);
        })
        .catch(() => undefined);
    };
    poll();
    const timer = setInterval(poll, model.cfg.sourcesPollMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [model, rest, ok]);
}

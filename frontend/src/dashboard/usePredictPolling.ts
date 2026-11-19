/** Polls GET /predict into the model while live (inference state, bundles, per-machine gate). */
import { useEffect } from "react";

import { useModel, useRest } from "./context.ts";
import { useStore } from "./state/store.ts";

export function usePredictPolling(): void {
  const model = useModel();
  const rest = useRest();
  const auth = useStore(model.auth);
  const mode = useStore(model.modeStore);
  const ok = auth.status === "ok";
  useEffect(() => {
    if (!ok || mode !== "live") return;
    let cancelled = false;
    const poll = (): void => {
      rest
        .getPredict()
        .then((r) => {
          if (cancelled || typeof r !== "object" || !("inference" in r)) return;
          model.inference.set(r);
          model.refreshHighlights(); // the stale time may have changed
        })
        .catch(() => undefined); // an older backend or a hiccup: the panel says "not known"
    };
    poll();
    const timer = setInterval(poll, model.cfg.predictPollMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [model, rest, ok, mode]);
}

/**
 * Raise times need an alert's first event (rev 0). The stream delivers every transition from
 * the moment we connect, but an alert raised earlier (a snapshot after connecting, a restart, a
 * replay) arrives at a later rev. For those, fetch that machine's lifecycle events up to the
 * alert's current transition from `/alerts/history`, once per alert.
 */
import { useEffect, useRef } from "react";

import { useModel, useRest } from "./context.ts";
import { useStore } from "./state/store.ts";

export function useAlertBackfill(): void {
  const model = useModel();
  const rest = useRest();
  const alerts = useStore(model.alerts);
  const auth = useStore(model.auth);
  const requested = useRef(new Set<string>());

  useEffect(() => {
    if (auth.status !== "ok") return;
    const events = model.alertEvents.get();
    // newest transition per machine among alerts whose raise event is missing
    const upTo = new Map<string, number>();
    for (const a of alerts.values()) {
      if (requested.current.has(a.alertId)) continue;
      if ((events.get(a.alertId) ?? []).some((e) => e.rev === 0)) continue;
      requested.current.add(a.alertId);
      upTo.set(a.machineId, Math.max(upTo.get(a.machineId) ?? 0, Date.parse(a.ts)));
    }
    for (const [machine, t] of upTo) {
      rest
        .getAlertHistory({ machine, to: new Date(t + 1).toISOString() })
        .then((r) => {
          model.addAlertEvents(Array.isArray(r.alerts) ? r.alerts : []);
        })
        .catch(() => undefined); // the times then show as unknown
    }
  }, [alerts, auth.status, model, rest]);
}

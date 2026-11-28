/** Alert markers for one machine's charts, from the alerts and alert events in the model. */
import { useMemo } from "react";

import { useModel } from "../context.ts";
import { useStore } from "../state/store.ts";
import { alertMarkers, type ChartMarker } from "./annotations.ts";

export function useAlertMarkers(machineId: string | null): readonly ChartMarker[] {
  const model = useModel();
  const alerts = useStore(model.alerts);
  const events = useStore(model.alertEvents);
  return useMemo(() => (machineId === null ? [] : alertMarkers(alerts.values(), events, machineId)), [alerts, events, machineId]);
}

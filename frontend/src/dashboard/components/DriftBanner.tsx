/**
 * Model drift (Phase 11 Part C, ADR 0031): a banner while any machine's operating conditions
 * (speed, load) differ from the data the models were trained on. Predictions of that machine
 * are marked DEGRADED. The banner comes from the active `model_drift` alerts.
 */
import { useModel } from "../context.ts";
import { activeDrift } from "../state/predictions.ts";
import { useStore } from "../state/store.ts";

export function DriftBanner() {
  const model = useModel();
  const alerts = useStore(model.alerts);
  const notices = activeDrift(alerts.values());
  if (notices.length === 0) return null;
  const machines = notices.map((n) => n.machineId).join(", ");
  return (
    <div className="banner banner-drift" role="status" aria-live="polite" data-testid="drift-banner">
      <strong>Model drift on {machines}.</strong> The operating conditions differ from the data the models were trained on.
      Predictions for {notices.length === 1 ? "this machine are" : "these machines are"} marked DEGRADED: their confidence may
      not hold.
      <ul className="small drift-notes">
        {notices.map((n) => (
          <li key={n.machineId} data-testid={`drift-note-${n.machineId}`}>
            {n.message}
          </li>
        ))}
      </ul>
    </div>
  );
}

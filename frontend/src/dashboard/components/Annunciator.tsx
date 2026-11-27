/**
 * Annunciator (DESIGN.md §5, after ref-1's incident banner): the most urgent alert at the top of
 * the stage with its ISA-18.2 lifecycle Raised → Acknowledged → Cleared. Urgency: active and
 * unacknowledged first, then active and acknowledged, then cleared but unacknowledged; severity
 * and recency within a group (groupAlerts). It pulses only while the alert is active and
 * unacknowledged. A click opens the Alerts tab.
 */
import type { Alert } from "../../api/types.ts";
import { useModel } from "../context.ts";
import { formatTime } from "../format.ts";
import { alertTimes, GROUP_ORDER, groupAlerts } from "../state/alerts.ts";
import { alertKindView } from "../state/predictions.ts";
import { useStore } from "../state/store.ts";

const SEVERITY_TONE: Record<Alert["severity"], string> = { CRITICAL: "tone-fault", WARNING: "tone-warning", INFO: "tone-maintenance" };

/** Badge classes for an alert: its kind's colour where the kind has one, else its severity's. */
function toneOf(a: Alert): string {
  const kind = alertKindView(a.kind).className;
  if (kind === "kind-unusual" || kind === "kind-drift") return kind;
  return SEVERITY_TONE[a.severity];
}

export function Annunciator({ onOpen }: { onOpen: () => void }) {
  const model = useModel();
  const alerts = useStore(model.alerts);
  const events = useStore(model.alertEvents);
  const groups = groupAlerts(alerts.values());
  const ordered = GROUP_ORDER.flatMap((g) => groups[g]);
  const top = ordered[0];

  if (top === undefined) {
    return (
      <div className="hud annunciator quiet" role="status" aria-live="polite" data-testid="annunciator">
        <span className="micro">Alarms</span> No active alerts
      </div>
    );
  }

  const t = alertTimes(top, events.get(top.alertId) ?? []);
  const unacked = top.active && !top.acknowledged;
  const predicted = alertKindView(top.kind).className === "kind-predicted";
  const steps = [
    { label: "Raised", done: true, at: t.raisedAt },
    { label: "Acknowledged", done: top.acknowledged, at: t.ackedAt },
    { label: "Cleared", done: !top.active, at: t.clearedAt },
  ];
  const more = ordered.length - 1;

  return (
    <div className="hud annunciator" role="status" aria-live="polite" data-testid="annunciator" data-unacked={unacked}>
      <button
        type="button"
        className="annunciator-main"
        onClick={onOpen}
        title="Open the Alerts tab"
        data-testid="annunciator-open"
      >
        <span className={`badge ${toneOf(top)}${unacked && !predicted ? " solid" : ""}${predicted ? " kind-predicted" : ""}`}>
          {top.severity}
        </span>
        <span className="annunciator-message">{top.message}</span>
        <time className="annunciator-time" dateTime={t.raisedAt ?? top.ts}>
          {formatTime(t.raisedAt ?? top.ts)}
        </time>
        {more > 0 && <span className="micro annunciator-more">+{more} more</span>}
      </button>
      <ol className="annunciator-steps" aria-label="Alert lifecycle">
        {steps.map((s) => (
          <li key={s.label} data-done={s.done}>
            <span className="micro">{s.label}</span>
            {s.done && s.at !== null && <time dateTime={s.at}>{formatTime(s.at)}</time>}
          </li>
        ))}
      </ol>
    </div>
  );
}

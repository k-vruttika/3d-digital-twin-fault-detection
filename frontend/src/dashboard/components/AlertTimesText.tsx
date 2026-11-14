/** "raised 00:12:01 · cleared 00:12:57 · acked 00:13:10 by alice" for one alert. */
import type { Alert } from "../../api/types.ts";
import { useModel } from "../context.ts";
import { formatDateTime, formatTime } from "../format.ts";
import { alertTimes } from "../state/alerts.ts";
import { useStore } from "../state/store.ts";

export function AlertTimesText({ alert }: { alert: Alert }) {
  const events = useStore(useModel().alertEvents);
  const t = alertTimes(alert, events.get(alert.alertId) ?? []);
  const part = (label: string, iso: string | null, suffix = "") =>
    iso === null ? null : (
      <span title={formatDateTime(iso)}>
        {label} {formatTime(iso)}
        {suffix}
      </span>
    );
  const parts = [
    t.raisedAt === null ? <span title="the raise event is not available">raised –</span> : part("raised", t.raisedAt),
    part("cleared", t.clearedAt),
    part("acked", t.ackedAt, alert.ackedBy ? ` by ${alert.ackedBy}` : ""),
  ].filter((p) => p !== null);
  return (
    <span className="alert-times" data-testid={`alert-times-${alert.alertId}`}>
      {parts.map((p, i) => (
        <span key={i}>
          {i > 0 && " · "}
          {p}
        </span>
      ))}
    </span>
  );
}

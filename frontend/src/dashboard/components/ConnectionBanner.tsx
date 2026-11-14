/** Connection pill (header) and banner (shown while not connected). */
import { useModel } from "../context.ts";
import { formatAge } from "../format.ts";
import { CONNECTION_TEXT } from "../state/connection.ts";
import { useStore } from "../state/store.ts";

export function ConnectionPill() {
  const conn = useStore(useModel().connection);
  return (
    <span className={`pill conn-${conn.status}`} data-testid="connection-pill" title={conn.detail ?? undefined}>
      <span className="dot" aria-hidden="true" />
      {CONNECTION_TEXT[conn.status]}
    </span>
  );
}

export function ConnectionBanner() {
  const conn = useStore(useModel().connection);
  if (conn.status === "connected") return null;
  let text: string;
  switch (conn.status) {
    case "connecting":
      text = "Connecting to the backend…";
      break;
    case "reconnecting":
      text = "Connection dropped: reconnecting automatically. The data shown may be out of date.";
      break;
    case "linkLost":
      text = `Link lost for ${formatAge(conn.downForMs)}: the data shown is out of date. Still trying to reconnect.`;
      break;
    case "unauthorized":
      text = "The backend refused the API key. Enter a valid key above to connect.";
      break;
  }
  return (
    <div className={`banner banner-${conn.status}`} role="status" aria-live="polite" data-testid="connection-banner">
      <strong>{CONNECTION_TEXT[conn.status]}.</strong> {text} Commands and acknowledgements are disabled.
      {conn.detail !== null && <span className="banner-detail"> ({conn.detail})</span>}
    </div>
  );
}

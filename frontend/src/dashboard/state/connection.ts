/**
 * What the dashboard tells the operator about the stream (pure function).
 *
 * - connected: the stream is live;
 * - connecting: the first connection is being set up (never live yet, not for long);
 * - reconnecting: the stream dropped and StreamClient is reconnecting (shorter than
 *   `linkLostAfterMs`);
 * - linkLost: not live for longer than that (or the client gave up): data on screen is old;
 * - unauthorized: the backend refused the key (4401); no reconnect until a new key is applied.
 */
import type { StreamStatus } from "../../api/ws.ts";

export type ConnectionStatus = "connecting" | "connected" | "reconnecting" | "linkLost" | "unauthorized";

export interface ConnectionView {
  status: ConnectionStatus;
  /** Milliseconds since the stream stopped being live (0 while live). */
  downForMs: number;
  detail: string | null;
}

export function connectionView(
  stream: StreamStatus,
  everLive: boolean,
  notLiveSinceMs: number | null,
  nowMs: number,
  linkLostAfterMs: number,
  detail: string | null = null,
): ConnectionView {
  if (stream === "live") return { status: "connected", downForMs: 0, detail };
  const down = notLiveSinceMs === null ? 0 : Math.max(0, nowMs - notLiveSinceMs);
  if (stream === "unauthorized") return { status: "unauthorized", downForMs: down, detail };
  if (stream === "closed") return { status: "linkLost", downForMs: down, detail };
  if (down >= linkLostAfterMs) return { status: "linkLost", downForMs: down, detail };
  return { status: everLive ? "reconnecting" : "connecting", downForMs: down, detail };
}

export const CONNECTION_TEXT: Record<ConnectionStatus, string> = {
  connecting: "Connecting…",
  connected: "Connected",
  reconnecting: "Reconnecting…",
  linkLost: "Link lost",
  unauthorized: "Key rejected",
};

/**
 * API key handling: the key lives in memory, or in sessionStorage when the operator ticks
 * "remember in this tab". It is never written to localStorage and never put in a URL (REST
 * sends it as the X-API-Key header, the WebSocket in its first message).
 */
import type { Role } from "../../api/types.ts";

export const SESSION_KEY = "linetwin.apiKey";

export type AuthStatus = "checking" | "ok" | "needsKey" | "rejected" | "error";

export interface AuthState {
  key: string | null;
  remember: boolean;
  status: AuthStatus;
  role: Role | null;
  principal: string | null;
  anonymous: boolean;
  error: string | null;
}

export const INITIAL_AUTH: AuthState = {
  key: null,
  remember: false,
  status: "checking",
  role: null,
  principal: null,
  anonymous: false,
  error: null,
};

/** Minimal Storage view (sessionStorage in the browser, a fake in tests). */
export interface KeyStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function sessionStore(): KeyStorage | null {
  try {
    return typeof sessionStorage === "undefined" ? null : sessionStorage;
  } catch {
    return null; // storage disabled
  }
}

export function loadRememberedKey(storage: KeyStorage | null = sessionStore()): string | null {
  try {
    const v = storage?.getItem(SESSION_KEY) ?? null;
    return v === "" ? null : v;
  } catch {
    return null;
  }
}

/** Keep the key for this tab (remember) or forget it; memory always holds the current key. */
export function persistKey(key: string | null, remember: boolean, storage: KeyStorage | null = sessionStore()): void {
  try {
    if (remember && key !== null && key !== "") storage?.setItem(SESSION_KEY, key);
    else storage?.removeItem(SESSION_KEY);
  } catch {
    // storage full or disabled: the key stays in memory only
  }
}

export function isOperator(auth: AuthState): boolean {
  return auth.status === "ok" && auth.role === "operator";
}

export function roleText(auth: AuthState): string {
  if (auth.status === "checking") return "checking key…";
  if (auth.status === "needsKey") return "no key (required)";
  if (auth.status === "rejected") return "key rejected";
  if (auth.status === "error") return "role unknown";
  if (auth.anonymous) return "anonymous viewer";
  return auth.role === "operator" ? `operator (${auth.principal ?? ""})` : `viewer (${auth.principal ?? ""})`;
}

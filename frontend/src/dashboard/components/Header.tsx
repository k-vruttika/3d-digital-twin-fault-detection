/** Top bar: wordmark and line, connection, data time, API key and role, label and theme toggles. */
import type { ReactNode } from "react";

import { useModel } from "../context.ts";
import { formatTime } from "../format.ts";
import { useStore } from "../state/store.ts";
import { setTheme, useTheme } from "../theme.ts";
import { ApiKeyField } from "./ApiKeyField.tsx";
import { ConnectionPill } from "./ConnectionBanner.tsx";

/** The event time the screen shows (newest data, or the replay position), in UTC. */
function DataTime() {
  const asOf = useStore(useModel().asOf);
  if (asOf === null) return null;
  return (
    <span className="data-time" data-testid="data-time" title="Event time of the newest data shown (UTC)">
      <span className="micro">Data</span>
      <time dateTime={asOf}>{formatTime(asOf)}</time>
      <span className="micro">UTC</span>
    </span>
  );
}

export function Header({ children }: { children?: ReactNode }) {
  const model = useModel();
  const line = useStore(model.line);
  const labelMode = useStore(model.labelMode);
  const theme = useTheme();
  return (
    <header className="header">
      <h1 className="wordmark">
        LineTwin <span className="line-id">{line?.lineId ?? ""}</span>
      </h1>
      <ConnectionPill />
      <DataTime />
      {children}
      <span className="spacer" />
      <ApiKeyField />
      <div className="header-tools">
        <button
          type="button"
          aria-pressed={labelMode === "all"}
          onClick={() => {
            model.setLabelMode(labelMode === "all" ? "auto" : "all");
          }}
          title="Show every 3D label (default: only machines and buffers that need attention)"
        >
          {labelMode === "all" ? "All labels" : "Key labels"}
        </button>
        <button
          type="button"
          onClick={() => {
            setTheme(theme === "dark" ? "light" : "dark");
          }}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          data-testid="theme-toggle"
        >
          {theme === "dark" ? "Light theme" : "Dark theme"}
        </button>
      </div>
    </header>
  );
}

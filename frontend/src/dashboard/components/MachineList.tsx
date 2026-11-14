/**
 * Machine list with state badges. A listbox: arrow keys / Home / End move the selection, which
 * is shared with the 3D view and the details panel.
 */
import { useMemo, type KeyboardEvent } from "react";

import { stateLabel } from "../../engine/visuals.ts";
import { useModel } from "../context.ts";
import { rootCauseOf } from "../state/model.ts";
import { useStore } from "../state/store.ts";
import { StateBadge } from "./StateBadge.tsx";

export function MachineList() {
  const model = useModel();
  const line = useStore(model.line);
  const machines = useStore(model.machines);
  const lineView = useStore(model.lineView);
  const alerts = useStore(model.alerts);
  const selected = useStore(model.selection);
  const conn = useStore(model.connection);
  const connected = conn.status === "connected";

  const alertCount = useMemo(() => {
    const n = new Map<string, number>();
    for (const a of alerts.values()) if (a.active) n.set(a.machineId, (n.get(a.machineId) ?? 0) + 1);
    return n;
  }, [alerts]);

  const ids = line?.topology.order ?? [];
  const onKey = (e: KeyboardEvent<HTMLUListElement>): void => {
    if (ids.length === 0) return;
    const i = selected === null ? -1 : ids.indexOf(selected);
    let next: number | null = null;
    if (e.key === "ArrowDown") next = Math.min(ids.length - 1, i + 1);
    else if (e.key === "ArrowUp") next = Math.max(0, i <= 0 ? 0 : i - 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = ids.length - 1;
    else if (e.key === "Escape") {
      model.select(null);
      e.preventDefault();
      return;
    }
    if (next === null) return;
    e.preventDefault();
    model.select(ids[next] ?? null);
  };

  if (line === null) {
    return (
      <section className="panel machine-list" aria-label="Machines">
        <h2>Machines</h2>
        <p className="muted">Waiting for the line layout…</p>
      </section>
    );
  }

  return (
    <section className="panel machine-list" aria-label="Machines">
      <h2>
        Machines <span className="muted">· {line.lineId}</span>
      </h2>
      <ul
        role="listbox"
        aria-label="Machines (arrow keys to select)"
        tabIndex={0}
        aria-activedescendant={selected === null ? undefined : `machine-${selected}`}
        onKeyDown={onKey}
        data-testid="machine-list"
      >
        {ids.map((id) => {
          const cfg = line.machines.find((m) => m.id === id);
          const m = machines[id];
          const state = m?.state ?? null;
          const label = stateLabel({
            state,
            rootCause: rootCauseOf(lineView, id),
            reason: m?.reason ?? null,
            parts: [],
            connected,
          });
          const n = alertCount.get(id) ?? 0;
          return (
            <li
              key={id}
              id={`machine-${id}`}
              role="option"
              aria-selected={selected === id}
              className={selected === id ? "selected" : undefined}
              onClick={() => {
                model.select(id);
              }}
              data-testid={`machine-item-${id}`}
            >
              <div className="row">
                <span className="mid">{id}</span>
                <StateBadge state={state} />
              </div>
              <div className="row small">
                <span className="muted">{cfg?.name ?? ""}</span>
                <span className="flags">
                  {m?.dataQuality?.status === "DEGRADED" && (
                    <span className="flag dq" title="data quality degraded">
                      DQ
                    </span>
                  )}
                  {n > 0 && (
                    <span className="flag alert" title={`${String(n)} active alert(s)`}>
                      {n} alert{n === 1 ? "" : "s"}
                    </span>
                  )}
                </span>
              </div>
              {label !== state && <div className="small cause">{label}</div>}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/**
 * The line strip (DESIGN.md §5): the machines in material-flow order along the bottom of the 3D
 * stage, joined by flow connectors that show what the data says about each buffer (FULL / EMPTY /
 * UNKNOWN; never a count, ADR 0016). A listbox: arrow keys (left/right or up/down), Home / End
 * move the selection, Escape clears it; the selection is shared with the 3D view and the inspector.
 */
import { useMemo, type KeyboardEvent } from "react";

import { bufferStatus, stateLabel } from "../../engine/visuals.ts";
import { useModel } from "../context.ts";
import { stateTone } from "../format.ts";
import { rootCauseOf } from "../state/model.ts";
import { useStore } from "../state/store.ts";
import { StateBadge } from "./StateBadge.tsx";

const NEXT_KEYS = new Set(["ArrowDown", "ArrowRight"]);
const PREV_KEYS = new Set(["ArrowUp", "ArrowLeft"]);

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
    if (NEXT_KEYS.has(e.key)) next = Math.min(ids.length - 1, i + 1);
    else if (PREV_KEYS.has(e.key)) next = Math.max(0, i <= 0 ? 0 : i - 1);
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
      <section className="line-strip machine-list" aria-label="Machines">
        <h2 className="micro">Machines</h2>
        <p className="muted">Waiting for the line layout…</p>
      </section>
    );
  }

  return (
    <section className="line-strip machine-list" aria-label="Machines">
      <h2 className="micro strip-title">
        Line <span className="mono">{line.lineId}</span>
      </h2>
      <ul
        role="listbox"
        aria-label="Machines (arrow keys to select)"
        aria-orientation="horizontal"
        tabIndex={0}
        aria-activedescendant={selected === null ? undefined : `machine-${selected}`}
        onKeyDown={onKey}
        data-testid="machine-list"
      >
        {ids.map((id, idx) => {
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
          const nextId = ids[idx + 1];
          const buffer = nextId === undefined ? undefined : line.buffers.find((b) => b.from === id && b.to === nextId);
          const flow = nextId === undefined ? null : bufferStatus(id, state, nextId, machines[nextId]?.state ?? null);
          return (
            <li
              key={id}
              id={`machine-${id}`}
              role="option"
              aria-selected={selected === id}
              className={selected === id ? "selected" : undefined}
              data-tone={stateTone(state)}
              onClick={() => {
                model.select(id);
              }}
              data-testid={`machine-item-${id}`}
            >
              <div className="strip-cell">
                <div className="row">
                  <span className="mid">{id}</span>
                  <StateBadge state={state} />
                </div>
                <div className="row small">
                  <span className="muted strip-name">{cfg?.name ?? ""}</span>
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
              </div>
              {flow !== null && (
                <span
                  className="strip-flow"
                  data-buffer={flow.status}
                  title={`${buffer?.id ?? "buffer"}${buffer ? ` (capacity ${String(buffer.capacity)})` : ""}: ${flow.why}`}
                  aria-hidden="true"
                >
                  <span className="strip-arrow">→</span>
                  {flow.status !== "FLOWING" && <span className="micro">{flow.status}</span>}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

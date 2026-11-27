/** Accessible tabs (arrow keys move between tabs; panels keep their own state while hidden). */
import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

export interface TabSpec {
  id: string;
  label: ReactNode;
  content: ReactNode;
}

interface TabsProps {
  tabs: TabSpec[];
  label: string;
  initial?: string;
  /** Controlled mode: the active tab id, changed through onChange (e.g. the annunciator opens Alerts). */
  value?: string;
  onChange?: (id: string) => void;
}

export function Tabs({ tabs, label, initial, value, onChange }: TabsProps) {
  const [own, setOwn] = useState(initial ?? tabs[0]?.id ?? "");
  const active = value ?? own;
  const setActive = (id: string): void => {
    setOwn(id);
    onChange?.(id);
  };
  const base = useId();
  const refs = useRef(new Map<string, HTMLButtonElement>());
  const current = tabs.some((t) => t.id === active) ? active : (tabs[0]?.id ?? "");

  const onKey = (e: KeyboardEvent): void => {
    const i = tabs.findIndex((t) => t.id === current);
    let next: number;
    if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    else return;
    e.preventDefault();
    const t = tabs[next];
    if (t === undefined) return;
    setActive(t.id);
    refs.current.get(t.id)?.focus();
  };

  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} onKeyDown={onKey}>
        {tabs.map((t) => (
          <button
            key={t.id}
            ref={(el) => {
              if (el === null) refs.current.delete(t.id);
              else refs.current.set(t.id, el);
            }}
            role="tab"
            type="button"
            id={`${base}-tab-${t.id}`}
            aria-selected={t.id === current}
            aria-controls={`${base}-panel-${t.id}`}
            tabIndex={t.id === current ? 0 : -1}
            onClick={() => {
              setActive(t.id);
            }}
            data-testid={`tab-${t.id}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tabs.map((t) => (
        <div
          key={t.id}
          role="tabpanel"
          id={`${base}-panel-${t.id}`}
          aria-labelledby={`${base}-tab-${t.id}`}
          hidden={t.id !== current}
          className="tabpanel"
        >
          {t.id === current ? t.content : null}
        </div>
      ))}
    </div>
  );
}

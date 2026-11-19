/** Pure dashboard state: connection view, alert groups, auth key storage, series, model routing. */
import { describe, expect, it } from "vitest";

import { routeMessage } from "../../api/bind.ts";
import type { Reading } from "../../api/types.ts";
import { LINE } from "../../test/fakes.ts";
import { alert, reading, snapshot } from "../../test/dashboard.tsx";
import { BUCKET_S, bucketsToSeries, chooseResolution, readingsToSeries, trimWindow } from "../charts/series.ts";
import { DASHBOARD_CONFIG } from "../config.ts";
import { alertTimes, applyAlertEvent, groupAlerts, mergeAlertEvents } from "./alerts.ts";
import { loadRememberedKey, persistKey, SESSION_KEY, type KeyStorage } from "./auth.ts";
import { connectionView } from "./connection.ts";
import { DashboardModel } from "./model.ts";

describe("connectionView", () => {
  const L = 5000;
  it("maps stream status and time down to what the operator sees", () => {
    expect(connectionView("live", true, null, 0, L).status).toBe("connected");
    expect(connectionView("connecting", false, 0, 100, L).status).toBe("connecting");
    expect(connectionView("reconnecting", true, 0, 100, L).status).toBe("reconnecting");
    expect(connectionView("reconnecting", true, 0, L, L).status).toBe("linkLost");
    expect(connectionView("connecting", false, 0, L + 1, L).status).toBe("linkLost");
    expect(connectionView("closed", true, 0, 1, L).status).toBe("linkLost");
    expect(connectionView("unauthorized", false, 0, 1, L).status).toBe("unauthorized");
  });
});

describe("alerts", () => {
  it("keeps the highest rev and drops resolved alerts", () => {
    let m = applyAlertEvent(new Map(), alert({ alertId: "a", machineId: "M", rev: 1, acknowledged: true }));
    m = applyAlertEvent(m, alert({ alertId: "a", machineId: "M", rev: 0 })); // older: ignored
    expect(m.get("a")?.rev).toBe(1);
    m = applyAlertEvent(m, alert({ alertId: "a", machineId: "M", rev: 2, active: false, acknowledged: true }));
    expect(m.has("a")).toBe(false);
  });

  it("orders by severity, then newest first", () => {
    const g = groupAlerts([
      alert({ alertId: "w", machineId: "M", severity: "WARNING", ts: "2026-01-01T00:00:09Z" }),
      alert({ alertId: "c1", machineId: "M", severity: "CRITICAL", ts: "2026-01-01T00:00:01Z" }),
      alert({ alertId: "c2", machineId: "M", severity: "CRITICAL", ts: "2026-01-01T00:00:02Z" }),
    ]);
    expect(g.activeUnacked.map((a) => a.alertId)).toEqual(["c2", "c1", "w"]);
  });
});

describe("alert times", () => {
  const ev = (rev: number, ts: string, active: boolean, acknowledged = false) =>
    alert({ alertId: "w", machineId: "INSP-01", severity: "WARNING", kind: "machine_warning", rev, ts, active, acknowledged, ackedAt: acknowledged ? ts : null, ackedBy: acknowledged ? "alice" : null });
  const raised = ev(0, "2026-01-01T00:09:10Z", true);
  const cleared = ev(1, "2026-01-01T00:12:57Z", false);
  const acked = ev(2, "2026-01-01T00:14:00Z", false, true);

  it("a cleared alert shows when it was raised, not its latest transition", () => {
    expect(alertTimes(cleared, [raised, cleared])).toEqual({
      raisedAt: "2026-01-01T00:09:10Z",
      clearedAt: "2026-01-01T00:12:57Z",
      ackedAt: null,
    });
  });

  it("acked after clearing; events in any order and duplicated", () => {
    expect(alertTimes(acked, [acked, cleared, raised, raised])).toEqual({
      raisedAt: "2026-01-01T00:09:10Z",
      clearedAt: "2026-01-01T00:12:57Z",
      ackedAt: "2026-01-01T00:14:00Z",
    });
  });

  it("only events up to the current rev count (replay), and a re-raise clears the clear time", () => {
    expect(alertTimes(raised, [raised, cleared, acked])).toEqual({ raisedAt: raised.ts, clearedAt: null, ackedAt: null });
    const again = ev(2, "2026-01-01T00:13:30Z", true);
    expect(alertTimes(again, [raised, cleared, again]).clearedAt).toBeNull();
  });

  it("without the raise event the raise time is unknown", () => {
    expect(alertTimes(cleared, []).raisedAt).toBeNull();
  });

  it("mergeAlertEvents keeps one event per rev, sorted", () => {
    const m = mergeAlertEvents(new Map(), [cleared, raised, cleared]);
    expect(m.get("w")?.map((e) => e.rev)).toEqual([0, 1]);
  });
});

class MemoryStorage implements KeyStorage {
  readonly data = new Map<string, string>();
  getItem(k: string): string | null {
    return this.data.get(k) ?? null;
  }
  setItem(k: string, v: string): void {
    this.data.set(k, v);
  }
  removeItem(k: string): void {
    this.data.delete(k);
  }
}

describe("API key storage", () => {
  it("keeps the key in sessionStorage only when remembered, and forgets it otherwise", () => {
    const s = new MemoryStorage();
    persistKey("secret", true, s);
    expect(s.getItem(SESSION_KEY)).toBe("secret");
    expect(loadRememberedKey(s)).toBe("secret");
    persistKey("secret", false, s);
    expect(s.getItem(SESSION_KEY)).toBeNull();
    persistKey(null, true, s);
    expect(loadRememberedKey(s)).toBeNull();
  });

  it("survives a storage that throws (disabled storage)", () => {
    const broken: KeyStorage = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
      removeItem: () => {
        throw new Error("denied");
      },
    };
    expect(() => {
      persistKey("k", true, broken);
    }).not.toThrow();
    expect(loadRememberedKey(broken)).toBeNull();
  });
});

describe("series", () => {
  const cfg = DASHBOARD_CONFIG;
  it("missing values become null and long time gaps break the line", () => {
    const rs: Reading[] = [
      reading("M", "RUNNING", 1),
      reading("M", "RUNNING", 2, { temp: null }),
      reading("M", "RUNNING", 3),
      reading("M", "RUNNING", 4),
      reading("M", "RUNNING", 20), // 16 s gap at a 1 s interval
    ];
    const pts = readingsToSeries(rs, "temp", cfg);
    expect(pts.map((p) => p.y)).toEqual([30, null, 30, 30, null, 30]);
    expect(pts.some((p) => p.y === 0)).toBe(false);
  });

  it("an absent sensor key is a gap too (the API drops nulls from JSON)", () => {
    const noVib: Reading = reading("M", "RUNNING", 1);
    delete noVib.vib;
    const pts = readingsToSeries([reading("M", "RUNNING", 0), noVib, reading("M", "RUNNING", 2)], "vib", cfg);
    expect(pts.map((p) => p.y)).toEqual([1, null, 1]);
  });

  it("empty and missing buckets are gaps; the band has min and max", () => {
    const b = (minute: number, count: number) => ({
      bucket: new Date(Date.parse("2026-01-01T00:00:00Z") + minute * 60_000).toISOString(),
      machineId: "M",
      n: count,
      nLate: 0,
      nSuspect: 0,
      sensors: { temp: count > 0 ? { count, mean: 30, min: 29, max: 31 } : { count: 0 } },
    });
    const s = bucketsToSeries([b(0, 60), b(1, 0), b(2, 60), b(5, 60)], "temp", BUCKET_S["1m"]);
    expect(s.mean.map((p) => p.y)).toEqual([30, null, 30, null, 30]);
    expect(s.min.find((p) => p.y !== null)?.y).toBe(29);
    expect(s.max.find((p) => p.y !== null)?.y).toBe(31);
  });

  it("chooses raw, 1 min or 10 min by the visible span", () => {
    expect(chooseResolution(15 * 60, cfg)).toBe("raw");
    expect(chooseResolution(6 * 3600, cfg)).toBe("1m");
    expect(chooseResolution(3 * 86400, cfg)).toBe("10m");
  });

  it("trimWindow keeps the last window of event time", () => {
    const rs = [1, 2, 300, 400].map((s) => reading("M", "RUNNING", s));
    expect(trimWindow(rs, 300).map((r) => r.seq)).toEqual([300, 400]);
  });
});

describe("DashboardModel", () => {
  it("folds snapshots, readings, state changes and alerts like the engine", () => {
    const m = new DashboardModel();
    m.loadLayout(LINE);
    m.applySnapshot(snapshot({ "CNC-01": "RUNNING" }, [alert({ alertId: "a", machineId: "CNC-01" })]));
    expect(m.machines.get()["CNC-01"]?.state).toBe("RUNNING");
    expect(m.alerts.get().size).toBe(1);
    m.applyReading(reading("CNC-01", "WARNING", 9));
    expect(m.machines.get()["CNC-01"]?.state).toBe("WARNING");
    m.applyReading(reading("CNC-01", "RUNNING", 3)); // older: ignored
    expect(m.machines.get()["CNC-01"]?.reading?.seq).toBe(9);
    m.applyReading(reading("CNC-01", "FAULT", 10, { quality: "LATE" })); // LATE: no state change
    expect(m.machines.get()["CNC-01"]?.state).toBe("WARNING");
    m.applyStateChange({ machineId: "CNV-01", state: "BLOCKED", previousState: "RUNNING", ts: "2026-01-01T00:00:12Z", reason: "BLOCKED by CNC-01" });
    expect(m.machines.get()["CNV-01"]?.reason).toBe("BLOCKED by CNC-01");
    expect(m.asOf.get()).toBe(reading("CNC-01", "FAULT", 10).ts);
  });

  it("routes command and ack frames to listeners and replays state into a late engine", () => {
    const m = new DashboardModel();
    m.loadLayout(LINE);
    const events: string[] = [];
    m.onCommandEvent((e) => events.push(e.command ? "command" : "ack"));
    routeMessage(m, {
      type: "command",
      data: { commandId: "c1", idempotencyKey: "k", machineId: "CNC-01", type: "STOP", params: {}, issuedBy: "alice", ts: "2026-01-01T00:00:00Z" },
    });
    routeMessage(m, { type: "ack", data: { commandId: "c1", status: "ACCEPTED", reason: null, ts: "2026-01-01T00:00:00Z" } });
    expect(events).toEqual(["command", "ack"]);

    m.applyReading(reading("CNC-01", "FAULT", 4));
    const calls: string[] = [];
    const fake = {
      mode: "live" as const,
      loadLayout: () => calls.push("loadLayout"),
      applySnapshot: (s: { machines?: Record<string, { state?: string | null }> }) => {
        calls.push(`snapshot:${String(s.machines?.["CNC-01"]?.state)}`);
      },
      applyReading: () => undefined,
      applyStateChange: () => undefined,
      applyDataQuality: () => undefined,
      applyAlert: () => undefined,
      applyLine: () => undefined,
      setPrediction: () => undefined,
      highlightParts: () => [],
      setConnected: () => undefined,
      setMode: () => undefined,
      select: () => undefined,
      setLabelMode: () => undefined,
    };
    m.attachEngine(fake);
    expect(calls).toEqual(["loadLayout", "snapshot:FAULT"]);
  });
});

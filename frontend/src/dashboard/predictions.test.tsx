// @vitest-environment jsdom
/**
 * Phase 11 Part B: live predictions in the dashboard (ADR 0030).
 * Pure helpers, the engine's unusual-behaviour visual, part highlights (supervised only),
 * the prediction panel (gate texts, sets in words, the honest note), replay of stored
 * predictions, and the distinct unusual-behaviour alert.
 */
import { act, cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Prediction, PredictResponse } from "../api/types.ts";
import { DEFAULT_CONFIG } from "../engine/config.ts";
import { machineVisual } from "../engine/visuals.ts";
import { OPERATOR, alert, pathOf, reading, renderDashboard, type Route } from "../test/dashboard.tsx";
import {
  activeDrift,
  alertKindView,
  confidenceText,
  resolveHighlight,
  unusualView,
  gateText,
  highlightsFor,
  localGate,
  setText,
  topClass,
  ttfText,
} from "./state/predictions.ts";

afterEach(() => {
  cleanup();
});

const T = "2026-01-01T00:00:05Z";

function pred(over: Partial<Prediction> = {}): Prediction {
  return {
    machineId: "CNC-01",
    ts: T,
    horizonS: 30,
    probs: { NORMAL: 0.1, WARNING: 0.7, FAULT: 0.2 },
    calibrated: true,
    conformalSet: ["WARNING", "FAULT"],
    ttfS: 140,
    ttfInterval: [60, 220],
    anomalyScore: 0.01,
    attributions: [
      { sensor: "vib", part: "spindle_bearing", weight: 0.7 },
      { sensor: "temp", part: "spindle_motor", weight: 0.2 },
      { sensor: "load", part: "spindle", weight: -0.1 },
    ],
    reliability: "OK",
    modelVersion: "linetwin-gru v1 (gru seed 0) + linetwin-autoencoder v1 (autoencoder seed 0)",
    alarm: true,
    anomalyFlag: false,
    anomalyThreshold: 0.18,
    attributionsTs: "2026-01-01T00:00:04Z",
    ...over,
  };
}

function bundle(kind: "supervised" | "anomaly") {
  return {
    registered: kind === "supervised" ? "linetwin-gru" : "linetwin-autoencoder",
    version: "1",
    label: kind === "supervised" ? "linetwin-gru v1 (gru seed 0)" : "linetwin-autoencoder v1 (autoencoder seed 0)",
    kind,
    modelType: kind === "supervised" ? "gru" : "autoencoder",
    seed: 0,
    datasetKey: "944580196604b618",
    builtAt: "2026-11-19T13:00:00+00:00",
    checks: { calibration: "reproduced" },
    card: {
      trainingData: "Trained and validated on simulated data only.",
      realData: "Phase 10 did not demonstrate early warning on real machines.",
      limitations: ["Conformal coverage is approximate."],
    },
    horizons: kind === "supervised" ? [10, 30] : null,
    ttfCapS: kind === "supervised" ? 600 : null,
    sustainS: kind === "supervised" ? 3 : null,
    alpha: kind === "supervised" ? 0.1 : null,
    percentile: kind === "anomaly" ? 99.9 : null,
  };
}

function predictBody(gates: Record<string, { status: string; runLen: number }>): PredictResponse {
  return {
    inference: {
      enabled: true,
      state: "ready",
      model: "gru",
      modelVersion: "linetwin-gru v1 (gru seed 0) + linetwin-autoencoder v1 (autoencoder seed 0)",
      windowS: 30,
      primaryHorizonS: 30,
      predictionStaleS: 5,
      supervised: bundle("supervised"),
      anomaly: bundle("anomaly"),
    },
    machines: Object.fromEntries(
      Object.entries(gates).map(([id, g]) => [
        id,
        {
          gate: { ...g, window: 30, alarm: false, predictedFault: false, unusual: false },
          predictions: {},
        },
      ]),
    ),
  } as PredictResponse;
}

const predictRoute =
  (gates: Record<string, { status: string; runLen: number }>): Route =>
  (c) =>
    pathOf(c) === "/predict" ? { body: predictBody(gates) } : undefined;

// ---------------------------------------------------------------------------- pure helpers
describe("prediction helpers", () => {
  it("highlights only parts with positive supervised evidence, strongest first", () => {
    expect(highlightsFor(pred(), T, 5000, 2)).toEqual([
      { part: "spindle_bearing", intensity: 1 },
      { part: "spindle_motor", intensity: 0.2 / 0.7 },
    ]);
    expect(highlightsFor(pred({ attributions: [] }), T, 5000, 2)).toEqual([]); // NORMAL: none
    expect(highlightsFor(pred(), "2026-01-01T00:00:20Z", 5000, 2)).toEqual([]); // stale
    expect(highlightsFor(null, T, 5000, 2)).toEqual([]);
  });

  it("words: sets, time to fault, top class, gate", () => {
    expect(setText(["WARNING", "FAULT"])).toBe("WARNING or FAULT");
    expect(setText(["NORMAL"])).toBe("NORMAL");
    expect(setText(["NORMAL", "WARNING", "FAULT"])).toBe("any state");
    expect(setText([])).toBe("empty set");
    expect(ttfText(pred(), 600)).toBe("about 140 s (interval 60–220 s)");
    expect(ttfText(pred({ ttfS: 600 }), 600)).toContain("beyond 600 s");
    expect(topClass(pred())).toEqual({ cls: "WARNING", prob: 0.7 });
    expect(gateText("warming_up", 12, 30)).toBe("Warming up (12/30 s of running data)");
    expect(gateText("stopped")).toBe("No prediction (machine stopped)");
    expect(gateText("waiting")).toBe("No prediction (machine waiting)");
  });

  it("local gate: stopped and waiting states never show a prediction", () => {
    for (const s of ["IDLE", "FAULT", "MAINTENANCE", "STALE", "OFFLINE"] as const) {
      expect(localGate(s, pred(), T, 5000)).toBe("stopped");
    }
    expect(localGate("STARVED", pred(), T, 5000)).toBe("waiting");
    expect(localGate("RUNNING", pred(), T, 5000)).toBe("predict");
    expect(localGate("RUNNING", pred(), "2026-01-01T00:01:00Z", 5000)).toBe("no_recent");
  });

  it("unusual behaviour has its own look and says cause unknown", () => {
    const v = alertKindView("unusual_behaviour");
    expect(v.className).toBe("kind-unusual");
    expect(v.note).toContain("cause unknown");
    expect(alertKindView("predicted_fault").label).toBe("predicted FAULT");
    expect(alertKindView("machine_fault").className).toBe("");
  });
});

// ---------------------------------------------------------------------------- engine visual
describe("unusual-behaviour visual", () => {
  const parts = ["spindle", "spindle_motor", "spindle_bearing"];
  const unusual = [{ severity: "WARNING" as const, active: true, kind: "unusual_behaviour" }];

  it("outlines the whole machine in its own colour, never one part", () => {
    const spec = machineVisual({ state: "RUNNING", parts, alerts: unusual }, DEFAULT_CONFIG);
    expect(spec.unusual).toBe(true);
    expect(spec.outlineParts).toEqual(parts);
    expect(spec.outlineColor).toBe(DEFAULT_CONFIG.colors.unusual);
  });

  it("never over a state outline, nor on a stale machine; not without the alert", () => {
    const warn = machineVisual(
      { state: "WARNING", parts, alerts: unusual, attributions: [{ part: "spindle_bearing", weight: 1 }] },
      DEFAULT_CONFIG,
    );
    expect(warn.outlineParts).toEqual(["spindle_bearing"]);
    expect(warn.outlineColor).toBe(DEFAULT_CONFIG.colors.warning);
    expect(machineVisual({ state: "STALE", parts, alerts: unusual }, DEFAULT_CONFIG).outlineParts).toEqual([]);
    const cleared = [{ severity: "WARNING" as const, active: false, kind: "unusual_behaviour" }];
    expect(machineVisual({ state: "RUNNING", parts, alerts: cleared }, DEFAULT_CONFIG).outlineParts).toEqual([]);
    // display hold: the outline stays briefly after the alert cleared
    expect(machineVisual({ state: "RUNNING", parts, alerts: cleared, unusualHeld: true }, DEFAULT_CONFIG).unusual).toBe(true);
    expect(machineVisual({ state: "RUNNING", parts, alerts: cleared }, DEFAULT_CONFIG).unusual).toBe(false);
  });
});

// ---------------------------------------------------------------------------- model + engine
describe("part highlights", () => {
  it("come from supervised attributions and clear on NORMAL or when stale", async () => {
    const h = renderDashboard({ auth: OPERATOR });
    await waitFor(() => {
      expect(h.model.engineReady.get()).toBe(true);
    });
    act(() => {
      h.model.setPrediction(pred());
    });
    expect(h.engine.highlights.get("CNC-01")?.[0]).toEqual({ part: "spindle_bearing", intensity: 1 });
    // the autoencoder flag alone never highlights anything
    act(() => {
      h.model.setPrediction(
        pred({ ts: "2026-01-01T00:00:06Z", attributions: [], alarm: false, anomalyFlag: true, probs: { NORMAL: 0.9, WARNING: 0.05, FAULT: 0.05 } }),
      );
    });
    expect(h.engine.highlights.get("CNC-01")).toEqual([]);
    act(() => {
      h.model.setPrediction(pred({ ts: "2026-01-01T00:00:07Z" }));
    });
    expect(h.engine.highlights.get("CNC-01")?.length).toBe(2);
    act(() => {
      h.model.applyReading(reading("CNC-01", "RUNNING", 30)); // 23 s later: stale
    });
    expect(h.engine.highlights.get("CNC-01")).toEqual([]);
  });
});

describe("Part C group 1: display fixes", () => {
  it("keeps the last highlight while the alarm stays on, labelled with its window, and clears with the alarm", async () => {
    const h = renderDashboard({ auth: OPERATOR });
    await waitFor(() => {
      expect(h.model.engineReady.get()).toBe(true);
    });
    act(() => {
      h.model.setPrediction(pred());
    });
    expect(h.model.highlights.get()["CNC-01"]).toMatchObject({ windowTs: "2026-01-01T00:00:04Z", held: false });
    // a later prediction, alarm still on, explanation late (none attached)
    act(() => {
      h.model.setPrediction(pred({ ts: "2026-01-01T00:00:07Z", attributions: [], attributionsTs: null }));
    });
    expect(h.model.highlights.get()["CNC-01"]).toMatchObject({ windowTs: "2026-01-01T00:00:04Z", held: true });
    expect(h.engine.highlights.get("CNC-01")?.[0]?.part).toBe("spindle_bearing");
    // alarm clears: highlight clears
    act(() => {
      h.model.setPrediction(pred({ ts: "2026-01-01T00:00:08Z", attributions: [], alarm: false }));
    });
    expect(h.model.highlights.get()["CNC-01"]).toBeUndefined();
    expect(h.engine.highlights.get("CNC-01")).toEqual([]);
  });

  it("resolveHighlight: stale prediction or time going backwards drops the hold", () => {
    const first = resolveHighlight(pred(), null, T, 5000, 2);
    expect(first?.held).toBe(false);
    const late = pred({ ts: "2026-01-01T00:00:07Z", attributions: [] });
    expect(resolveHighlight(late, first, "2026-01-01T00:00:08Z", 5000, 2)?.held).toBe(true);
    expect(resolveHighlight(late, first, "2026-01-01T00:00:30Z", 5000, 2)).toBeNull(); // stale
    expect(resolveHighlight(pred({ ts: "2026-01-01T00:00:02Z", attributions: [] }), first, "2026-01-01T00:00:03Z", 5000, 2)).toBeNull();
    expect(resolveHighlight(null, first, T, 5000, 2)).toBeNull();
  });

  it("conformal set wording: most likely first, then what cannot be ruled out", () => {
    const normalFault = pred({ probs: { NORMAL: 0.996, WARNING: 0.003, FAULT: 0.001 }, conformalSet: ["FAULT"] });
    expect(confidenceText(normalFault)).toBe("Most likely NORMAL (1.00) · cannot rule out FAULT");
    expect(confidenceText(pred({ conformalSet: ["WARNING"] }))).toBe("Most likely WARNING (0.70) · no other state in the set");
    expect(confidenceText(pred({ conformalSet: ["NORMAL", "WARNING", "FAULT"] }))).toBe(
      "Most likely WARNING (0.70) · cannot rule out NORMAL, FAULT",
    );
    expect(confidenceText(pred({ conformalSet: [] }))).toContain("empty set");
  });

  const ucfg = { flapWindowS: 300, flapMinSwitches: 4, holdS: 30 };
  const ev = (rev: number, ts: string, active: boolean, id = "u1") =>
    alert({ alertId: id, machineId: "CNC-01", kind: "unusual_behaviour", rev, ts, active, severity: "WARNING" });

  it("unusual alarm: intermittent after several switches, held briefly after it clears, events untouched", () => {
    const one = [ev(0, "2026-01-01T00:00:10Z", true)];
    expect(unusualView(one, "CNC-01", "2026-01-01T00:00:20Z", ucfg)).toMatchObject({ visible: true, active: true, intermittent: false, label: "unusual behaviour" });
    const flap = [
      ev(0, "2026-01-01T00:00:10Z", true),
      ev(1, "2026-01-01T00:00:20Z", false),
      ev(0, "2026-01-01T00:00:30Z", true, "u2"),
      ev(1, "2026-01-01T00:00:40Z", false, "u2"),
    ];
    const frozen = JSON.stringify(flap);
    const v = unusualView(flap, "CNC-01", "2026-01-01T00:00:50Z", ucfg);
    expect(v).toMatchObject({ visible: true, active: false, intermittent: true, label: "unusual behaviour (intermittent)" });
    expect(unusualView(flap, "CNC-01", "2026-01-01T00:01:20Z", ucfg).visible).toBe(false); // hold over
    expect(unusualView(flap, "CNC-01", "2026-01-01T00:10:00Z", ucfg).intermittent).toBe(false); // outside the period
    expect(unusualView(flap, "CNV-01", "2026-01-01T00:00:50Z", ucfg).visible).toBe(false);
    expect(JSON.stringify(flap)).toBe(frozen);
  });

  it("a supervised alarm supersedes the hold (no part highlight and unusual outline at once)", () => {
    const flap = [ev(0, "2026-01-01T00:00:10Z", true), ev(1, "2026-01-01T00:00:20Z", false)];
    expect(unusualView(flap, "CNC-01", "2026-01-01T00:00:30Z", ucfg).visible).toBe(true); // held
    expect(unusualView(flap, "CNC-01", "2026-01-01T00:00:30Z", ucfg, true).visible).toBe(false);
    const on = [ev(0, "2026-01-01T00:00:10Z", true)];
    expect(unusualView(on, "CNC-01", "2026-01-01T00:00:30Z", ucfg, true).active).toBe(true); // still active: shown
  });

  it("the model holds the 3D outline and shows the intermittent label", async () => {
    const h = renderDashboard({ auth: OPERATOR });
    await waitFor(() => {
      expect(h.model.engineReady.get()).toBe(true);
    });
    act(() => {
      for (const [rev, ts, active, id] of [
        [0, "2026-01-01T00:00:10Z", true, "u1"],
        [1, "2026-01-01T00:00:20Z", false, "u1"],
        [0, "2026-01-01T00:00:30Z", true, "u2"],
        [1, "2026-01-01T00:00:40Z", false, "u2"],
      ] as const) {
        h.model.applyAlert(ev(rev, ts, active, id));
      }
      h.model.applyReading(reading("CNC-01", "RUNNING", 41));
    });
    expect(h.model.unusual.get()["CNC-01"]).toMatchObject({ visible: true, active: false, intermittent: true });
    expect(h.engine.unusualHeld.get("CNC-01")).toBe(true);
    act(() => {
      h.model.applyReading(reading("CNC-01", "RUNNING", 75)); // 35 s after the clear
    });
    expect(h.model.unusual.get()["CNC-01"]?.visible).toBe(false);
    expect(h.engine.unusualHeld.get("CNC-01")).toBe(false);
  });
});

describe("Part C group 3: model drift", () => {
  const drift = (id: string, over: Record<string, unknown> = {}) =>
    alert({
      alertId: `d-${id}`,
      machineId: id,
      kind: "model_drift",
      severity: "WARNING",
      source: "inference",
      message: `${id}: operating conditions differ from the training data (load PSI 9.20 vs alert level 2.00)`,
      ...over,
    });

  it("lists machines with an active drift alert, newest first", () => {
    const list = activeDrift([
      drift("CNC-01", { ts: "2026-01-01T00:00:10Z" }),
      drift("CNV-01", { ts: "2026-01-01T00:00:20Z" }),
      drift("INSP-01", { active: false }),
      alert({ alertId: "x", machineId: "CNV-02", kind: "machine_fault" }),
    ]);
    expect(list.map((n) => n.machineId)).toEqual(["CNV-01", "CNC-01"]);
    expect(activeDrift([])).toEqual([]);
    expect(alertKindView("model_drift").note).toContain("DEGRADED");
  });

  it("shows a banner with the backend's message, and DEGRADED reliability, while drift lasts", async () => {
    const h = renderDashboard({ auth: OPERATOR, routes: [predictRoute({ "CNC-01": { status: "predict", runLen: 30 } })] });
    expect(screen.queryByTestId("drift-banner")).toBeNull();
    act(() => {
      h.model.setPrediction(pred({ reliability: "DEGRADED" }));
      h.model.applyAlert(drift("CNC-01"));
    });
    const banner = screen.getByTestId("drift-banner");
    expect(banner.textContent).toContain("Model drift on CNC-01");
    expect(banner.textContent).toContain("DEGRADED");
    expect(screen.getByTestId("drift-note-CNC-01").textContent).toContain("load PSI 9.20 vs alert level 2.00");
    fireEvent.click(screen.getByTestId("tab-prediction"));
    await waitFor(() => {
      expect(screen.getByTestId("prediction-reliability-CNC-01").textContent).toContain("DEGRADED");
    });
    expect(screen.getByTestId("prediction-about").textContent).toContain("Drift.");
    act(() => {
      h.model.applyAlert(drift("CNC-01", { active: false, rev: 1 }));
    });
    expect(screen.queryByTestId("drift-banner")).toBeNull();
  });
});

// ---------------------------------------------------------------------------- panel
describe("prediction panel", () => {
  it("shows current vs predicted state, the set in words, time to fault and the model", async () => {
    const h = renderDashboard({
      auth: OPERATOR,
      states: { "CNV-01": "IDLE", "CNC-01": "RUNNING", "INSP-01": "RUNNING", "CNV-02": "STARVED" },
      routes: [
        predictRoute({
          "CNV-01": { status: "stopped", runLen: 0 },
          "CNC-01": { status: "predict", runLen: 30 },
          "INSP-01": { status: "warming_up", runLen: 12 },
          "CNV-02": { status: "waiting", runLen: 0 },
        }),
      ],
    });
    act(() => {
      h.model.setPrediction(pred({ horizonS: 10, probs: { NORMAL: 0.8, WARNING: 0.15, FAULT: 0.05 } }));
      h.model.setPrediction(pred());
    });
    fireEvent.click(screen.getByTestId("tab-prediction"));
    await waitFor(() => {
      expect(screen.getByTestId("prediction-model").textContent).toContain("linetwin-gru v1 (gru seed 0)");
    });
    expect(screen.getByTestId("prediction-cell-CNC-01-30").textContent).toContain("WARNING");
    expect(screen.getByTestId("prediction-cell-CNC-01-30").textContent).toContain("0.70");
    expect(screen.getByTestId("prediction-cell-CNC-01-10").textContent).toContain("NORMAL");
    expect(screen.getByTestId("prediction-set-CNC-01").textContent).toBe(
      "Most likely WARNING (0.70) · cannot rule out FAULT",
    );
    expect(screen.getByTestId("prediction-ttf-CNC-01").textContent).toBe("about 140 s (interval 60–220 s)");
    expect(screen.getByTestId("prediction-reliability-CNC-01").textContent).toContain("OK");
    expect(screen.getByTestId("prediction-reliability-CNC-01").textContent).toContain("alarm");
    expect(screen.getByTestId("prediction-gate-CNV-01").textContent).toBe("No prediction (machine stopped)");
    expect(screen.getByTestId("prediction-gate-CNV-02").textContent).toBe("No prediction (machine waiting)");
    expect(screen.getByTestId("prediction-gate-INSP-01").textContent).toBe("Warming up (12/30 s of running data)");
    const about = screen.getByTestId("prediction-about").textContent;
    expect(about).toContain("Simulator data only");
    expect(about).toContain("did not demonstrate early warning on real machines");
    expect(about).toContain("names no cause and no part");
    expect(about).toContain("not a second prediction");
    expect(screen.getByTestId("prediction-card").textContent).toContain("simulated data only");
  });

  it("details of the selected machine: attributions with their window, the detector", () => {
    const h = renderDashboard({ auth: OPERATOR, routes: [predictRoute({ "CNC-01": { status: "predict", runLen: 30 } })] });
    act(() => {
      h.model.setPrediction(pred({ anomalyFlag: true, anomalyScore: 0.25 }));
      h.model.select("CNC-01");
    });
    fireEvent.click(screen.getByTestId("tab-prediction"));
    expect(screen.getByTestId("prediction-attributions").textContent).toContain("vib → spindle_bearing (+0.70)");
    expect(screen.getByTestId("prediction-attributions").textContent).toContain("explains the window ending 00:00:04");
    expect(screen.getByTestId("prediction-anomaly").textContent).toContain("cause unknown");
    expect(screen.getByTestId("prediction-alarm").textContent).toBe("on");
    expect(screen.getByTestId("details-prediction").textContent).toContain("WARNING");
  });

  it("details show no stale alarm or score once the machine has stopped", () => {
    const h = renderDashboard({ auth: OPERATOR, states: { "CNC-01": "FAULT" } });
    act(() => {
      h.model.setPrediction(pred({ anomalyFlag: true, anomalyScore: 16.2 }));
      h.model.select("CNC-01");
    });
    fireEvent.click(screen.getByTestId("tab-prediction"));
    expect(screen.getByTestId("prediction-details-gate").textContent).toBe("No prediction (machine stopped)");
    expect(screen.queryByTestId("prediction-alarm")).toBeNull();
    expect(screen.queryByTestId("prediction-anomaly")).toBeNull();
  });

  it("says when live inference is off", async () => {
    const off: Route = (c) =>
      pathOf(c) === "/predict"
        ? { body: { inference: { enabled: false, state: "off", reason: "live inference is off" }, machines: {} } }
        : undefined;
    renderDashboard({ auth: OPERATOR, routes: [off] });
    fireEvent.click(screen.getByTestId("tab-prediction"));
    await waitFor(() => {
      expect(screen.getByTestId("prediction-off").textContent).toContain("Live inference is off");
    });
  });

  it("replay shows the stored predictions as they were at time t", async () => {
    const h = renderDashboard({ auth: OPERATOR });
    await waitFor(() => {
      expect(h.model.engineReady.get()).toBe(true);
    });
    act(() => {
      h.model.setMode("replay");
      h.model.applyReplay(
        {
          asOf: "2026-01-01T00:00:10Z",
          machines: {},
          line: { status: "RUNNING", rootCauses: {}, impacted: {} },
          alerts: [],
          predictions: {
            "CNC-01": { "30": pred({ ts: "2026-01-01T00:00:09Z", probs: { NORMAL: 0.1, WARNING: 0.2, FAULT: 0.7 } }) },
            "INSP-01": { "30": pred({ machineId: "INSP-01", ts: "2026-01-01T00:00:01Z" }) },
          },
        } as never,
        "2026-01-01T00:00:10Z",
      );
    });
    fireEvent.click(screen.getByTestId("tab-prediction"));
    expect(screen.getByTestId("prediction-cell-CNC-01-30").textContent).toContain("FAULT");
    expect(screen.getByTestId("prediction-gate-INSP-01").textContent).toBe("No recent prediction"); // 9 s old
    expect(h.engine.highlights.get("CNC-01")?.[0]?.part).toBe("spindle_bearing");
  });
});

describe("alerts panel", () => {
  it("shows unusual behaviour distinctly, with cause unknown", () => {
    renderDashboard({
      auth: OPERATOR,
      alerts: [
        alert({
          alertId: "a1",
          machineId: "CNC-01",
          kind: "unusual_behaviour",
          severity: "WARNING",
          source: "inference",
          message: "CNC-01: unusual behaviour, cause unknown",
        }),
      ],
    });
    fireEvent.click(screen.getByTestId("tab-alerts"));
    const row = screen.getByTestId("alert-a1");
    expect(row.className).toContain("kind-unusual");
    expect(screen.getAllByTestId("kind-unusual_behaviour")[0]?.textContent).toContain("cause unknown");
  });
});

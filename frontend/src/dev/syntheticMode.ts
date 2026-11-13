/** Dev page without a backend: generated lines and readings (performance check, screenshots). */
import type { LineResponse, MachineState } from "../api/types.ts";
import lineFixture from "../engine/__fixtures__/line.json";
import singleFixture from "../engine/__fixtures__/line_single.json";
import {
  STATE_SCENE,
  SYNTHETIC_EPOCH_MS,
  SyntheticReadings,
  generateLine,
  stateSceneLine,
  stateSceneLineView,
} from "../engine/synthetic.ts";
import type { TwinEngine } from "../engine/TwinEngine.ts";

interface Options {
  /** Time is frozen (screenshots): apply one frame of data and stop. */
  frozen: boolean;
  setStatus: (status: string, text: string) => void;
}

/** Returns a cleanup function. */
export function startSynthetic(engine: TwinEngine, scene: string, opts: Options): () => void {
  const readings = new SyntheticReadings();
  let line: LineResponse;
  let stateOf: (id: string, tS: number) => MachineState;

  if (scene === "states") {
    line = stateSceneLine();
    const fixed = new Map(STATE_SCENE.map((s) => [s.id, s.state]));
    stateOf = (id) => fixed.get(id) ?? "RUNNING";
  } else if (scene === "line" || scene === "single") {
    line = (scene === "line" ? lineFixture : singleFixture) as unknown as LineResponse;
    stateOf = () => "RUNNING";
  } else {
    const n = Number.parseInt(scene, 10);
    line = generateLine(Number.isFinite(n) && n > 0 ? n : 20);
    // Every 8 s one machine shows WARNING for 4 s, so outlines and labels are exercised too.
    stateOf = (id, tS) => {
      const idx = line.machines.findIndex((m) => m.id === id);
      const cycle = Math.floor(tS / 8);
      return cycle % line.machines.length === idx && tS % 8 < 4 ? "WARNING" : "RUNNING";
    };
  }

  engine.loadLayout(line);
  if (scene === "states") {
    engine.applyLine(stateSceneLineView());
    engine.applyDataQuality({
      machineId: "S-WARN",
      status: "DEGRADED",
      samples: 60,
      latePct: 0,
      quarantinePct: 0,
      partialPct: 18,
      linkDown: false,
      reasons: ["partial readings 18.0 % > 10.0 %"],
    });
    engine.applyAlert({
      alertId: "S-FAULT:machine_fault:0",
      machineId: "S-FAULT",
      ts: new Date(SYNTHETIC_EPOCH_MS).toISOString(),
      severity: "CRITICAL",
      kind: "machine_fault",
      message: "S-FAULT is in FAULT",
      source: "synthetic",
      active: true,
      acknowledged: false,
      ackedBy: null,
      ackedAt: null,
      rev: 0,
    });
  }

  let tS = 0;
  const step = (): void => {
    for (const m of line.machines) engine.applyReading(readings.next(m, stateOf(m.id, tS), tS));
    tS += 1;
  };
  // Ten seconds of history, so a frozen frame looks like a running line.
  for (let i = 0; i < 10; i++) step();
  opts.setStatus("live", `synthetic · ${line.lineId} · ${String(line.machines.length)} machines`);
  if (opts.frozen) return () => undefined;
  const timer = setInterval(step, 1000);
  return () => {
    clearInterval(timer);
  };
}

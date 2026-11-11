# ADR 0003: Simulator design

- **Status:** Accepted
- **Date:** 2026-11-11
- **Phase:** 1

## Context

The simulator is the data source for every later phase: the ingest pipeline and adapters (Phase 2), what-if simulation, model training, and the robustness experiments E1–E5. It has to produce realistic, labelled data that is reproducible bit for bit. It also has to support snapshot, restore and fast-forward for what-if, and to model the line disruptions that the twin must explain (root cause). Several design choices affect the meaning of the labels and how valid the experiments are.

## Decision

1. **Discrete parts with integer buffers rather than a fluid-flow model.** Machines run cycles on individual parts, and buffer levels are integers bounded by the capacities in `line.yaml`. STARVED and BLOCKED require a grace period (15 s) without input or output. Otherwise the normal short gaps between parts would make the flow state flicker. Discrete parts also give the 3D twin something concrete to animate.
2. **States come from ground truth.** WARNING and FAULT are decided from the deterministic signals, as a controller's filtered value would be, with hysteresis. Measurement noise and sensor faults are not used. State priority is `MAINTENANCE > FAULT > IDLE > WARNING > STARVED/BLOCKED > RUNNING`. WARNING ranks above the flow states so that a degrading machine keeps its health label when the line starves it; under the §6 mapping, STARVED maps to NORMAL.
3. **Sensor faults are an observation layer.** Stuck, drift and dropout change only what is observed and are labelled in their own column. Machine labels are unaffected, which is what the robustness experiments need: the twin should still infer the true state.
4. **Independent seeded streams.** Run `i` uses `SeedSequence(seed, spawn_key=(i,))`, so output does not depend on the number of runs or workers. Each run splits into named streams: plan, sensor plan, process, measurement and operations. Enabling sensor faults or changing measurement noise therefore never changes the physics. RNG states are part of the snapshot, which makes restore bit-identical.
5. **Feasibility is validated with the config.** Fault parameters must guarantee two properties:
   - FAULT is reachable at the best case (low load, cold ambient, highest perturbed thresholds) within the progression time;
   - the lead time is at least `min_lead_s` at the worst case (high load, hot ambient, lowest thresholds, fastest progression).

   `twin_sim.feasibility` simulates the noise-free model and refuses configs that violate either property, so a parameter change cannot silently produce faults that never trigger, or that trigger without warning. Progress `w` is uncapped, which guarantees a FAULT is reached eventually; physical fractions such as cooling loss are clipped.
6. **Overheating uses a fault heat term, not only cooling loss.** Cooling loss alone did not reach the FAULT threshold at low load (CNC at 30 % load and 18 °C: about 47 °C against a FAULT range of 76.5–93.5 °C).
7. **The held-out fault has a distinct signature.** `lubrication_loss` is current-led, with stick-slip vibration bursts and speed jitter, and it trips on overcurrent. No training fault produces that combination, so E5 measures a genuinely unseen fault.
8. **A pure-Python step loop with multiprocessing across runs.** One simulated hour takes about 0.25 s. The full 200-run dataset takes about 10 s on 17 workers and about 1 min on one core, so vectorising the inner loop is not worth the loss in readability. numpy is used for RNG streams and label computation.

## Consequences

- Labels have a precise meaning (docs/simulator.md §8). `time_to_fault_s` is a right-censored target with an explicit censoring flag.
- Some gradual faults started late in a run, at low load, do not reach FAULT before the run ends. They show up as censored rows, and the manifest records planned versus realised faults.
- Random sampling yields fewer spindle_imbalance faults than other types, because only the CNC is eligible. The ML phases must handle the class imbalance, or a scenario can raise its share.
- `twin_sim.testing` is a pytest plugin registered by the root `conftest.py`, because test file names, including `conftest.py`, must be unique repo-wide (ADR 0001).

# Demo script: the digital twin with live predictions

This page has two parts. First, a **5–7 minute viva demo** in short steps. Then the **full scenarios** (a)–(e), about 40 minutes of real time, which explain each step and give the measured timings. Everything here runs on **simulated data**: the models were trained, calibrated and validated on the LineTwin simulator, and Phase 10 did not show early warning on real machines (`experiments/results/real_data.md`). Say so at the start of any demo.

The scenarios (a)–(d) follow the Part C plan; (e) is an extra. All five are also end-to-end tests (`tests/test_phase11_demo.py`, real-time simulator, real inference workers, real database). They are **opt-in** (`uv run poe demo-tests`, see the end of this page) and are not part of `uv run poe test`. The times quoted are **measured by those tests** and stored in [`experiments/results/demo_scenarios/`](../experiments/results/demo_scenarios); they are event times of one run on one laptop, not guarantees.

> **Which run the numbers come from.** The numbers below were measured on 2026-11-20, before the scenarios were relabelled to the plan. That run's files use the old labels: `demo_c_known_fault.json` is now (a), `demo_b_unseen_fault.json` is still (b), `demo_d_load_drift.json` is now (c) and `demo_a_healthy_line.json` is now (e). The 2026-11-20 run of (c) did not yet record the ordering check, and **(d) has not been run yet**. `uv run poe demo-tests` writes the new files (`demo_a_known_fault`, `demo_b_unseen_fault`, `demo_c_load_drift`, `demo_d_whatif_during_fault`, `demo_e_healthy_line`). After that run, delete the three files that still have the old names, and update the numbers here from the new files.

## Set up (once)

```text
uv run poe up                  # TimescaleDB + Mosquitto
uv run poe deploy-models       # the three bundles: GRU, XGBoost, autoencoder (about 13 min; fresh checkout: README.md)
uv run poe demo                # = up, then backend + simulator with live inference, and the dashboard (Ctrl+C stops)
```

`uv run poe demo` is the same as running `uv run poe dev --source mock --inference` in one terminal and `uv run poe frontend` in another. The dashboard is on http://127.0.0.1:5173.

Enter an operator key in the header (`LINETWIN_API_KEYS_OPERATOR` in `.env`, see `README.md`) to send commands and inject faults. The simulator runs in real time: a demo needs patience, which is the point.

## Viva demo (5–7 minutes)

The predicted warning comes several minutes after a fault is injected: 444 s in the measured run (a) below. Inject the fault **before** the viva starts, so it arrives while you talk. The timings are from one run on one laptop. If something takes longer, show the matching screenshot from `experiments/results/frontend_screenshots/dashboard/` instead.

**Before (about 10 minutes ahead)**

1. Start Docker Desktop, then run `uv run poe demo`. Open http://127.0.0.1:5173 and paste the operator key into the header.
2. Wait until the Prediction tab shows a prediction for every machine (after 30 s of running data).
3. **About 4 minutes before you start:** in the **Scenario (sim)** tab, inject `bearing_wear` on CNC-01, progression 720 s.

**During the viva**

1. **(0:00–0:45) What it is.** Say that this is simulated data only, and that the models showed no early warning on real data (Phase 10). Point at the 3D line, the machine list with state badges, and the connection pill.
2. **(0:45–1:30) Data in.** Open the **Sources** tab: the active source (mock server), ingestion counters (late, quarantined, duplicates, latency) and per-machine data quality. Mention that the same twin runs from a plain CSV with no Docker (`uv run poe minimal`).
3. **(1:30–2:30) Predictions you can check.** Open the **Prediction** tab. Healthy machines show NORMAL with a calibrated confidence. Show "cannot rule out …" (the conformal set), the reliability badge and the alarm column. Open **About these predictions**.
4. **(2:30–4:00) The injected fault on CNC-01.** Point out three signals, in this order:
   - the purple **unusual behaviour** outline (cause unknown) while the model still says NORMAL;
   - the **predicted WARNING** alarm, with the spindle bearing highlighted in 3D;
   - shortly after, the simulator's own **WARNING**.

   The **Alerts** tab shows the matching badges. If an event has not happened yet, say what comes next and show `prediction_unusual.png` / `prediction_alarm.png`.
5. **(4:00–5:15) What-if.** In the **What-if** tab select CNC-01, change **Set load** to 40 %, project 300 s, **Run what-if**. Compare "no change" with "with change" (ghost states, dashed lines, outcome text). Then **Apply as command…**, confirm, and show the command result.
6. **(5:15–6:00) History.** Click **Replay** and scrub back to before the injection to watch CNC-01 change state in 3D. Then click **Return to live**.
7. **(6:00–6:45) Limits.**
   - Unusual behaviour gives about 0.57 false alarms per hour on clean data.
   - A drift on speed and load marks predictions DEGRADED (show `prediction_drift.png`).
   - A what-if is the simulator's future, not a forecast.

**Afterwards:** Ctrl+C in the demo terminal, then `uv run poe down`.

# Full scenarios (reference)

Reset between scenarios by restarting `uv run poe dev ...` (an injected fault stays until the simulator restarts).

Before you start, open the Prediction tab. After 30 s of running data every machine gets a prediction: NORMAL with confidence about 1.00, reliability OK, no alarm. Explain what the confidence text means: "Most likely NORMAL (1.00) · cannot rule out …" lists the rest of a class-conditional set. Such a set guarantees coverage per class, so a rare class can appear at low probability.

## (a) A known fault: warned before the simulator's WARNING

**Do:** in the Scenario panel inject `bearing_wear` on CNC-01 (progression 720 s).

**Show, in this order:**

1. **Unusual behaviour** (purple, whole machine) while the supervised model still says NORMAL and the twin shows RUNNING.
2. **Predicted WARNING** (the evaluated alarm rule: +30 s argmax WARNING or FAULT for 3 s in a row), still while the twin shows RUNNING. The Prediction tab shows the explanation: the spindle bearing is highlighted in the 3D view, labelled with the time of the window it explains. If a newer explanation is late, the last one stays while the alarm is on.
3. **The simulator's own WARNING**: the bearing outlined in the warning colour.

**Measured (2026-11-20 run, file `demo_c_known_fault.json`; test now `test_demo_a_known_fault_warns_before_the_simulator`, which asserts this order):**

| Event | Seconds after injection (event time) |
|---|---|
| Unusual behaviour raised | 173 |
| Predicted WARNING raised | 444 |
| Simulator WARNING | 501 |

So the predicted warning came **57 s before** the simulator's own WARNING, and the autoencoder's unusual-behaviour alert came 271 s before the predicted one. The strongest explained part at the alarm was `spindle_bearing` (vibration, attribution 0.74; the next parts were below 0.01).

**What the test enforces:** injection, then unusual behaviour at or before the predicted warning, then the predicted warning strictly before the simulator's WARNING, with the bearing as the top part. The test now dates the simulator's WARNING from the stored readings, to the second. If a later change to the models or the alarm rule breaks that order, the test fails and says which event came first.

**Say:** the lead time is a property of this simulator, this fault type and this progression (720 s, the shortest in training). The evaluated lead times and miss rates are in `experiments/results/robustness.md` and `e5_explain.md`. The stricter prediction gate (only fully operating windows) costs a few seconds of lead (`live_inference.md`).

## (b) An unseen fault: unusual behaviour, cause unknown

**Do:** inject `lubrication_loss` on CNC-01 (progression 720 s). This fault type was never in the training data.

**Show:** within about a minute a purple outline appears around the **whole** CNC and the Alerts tab shows "unusual behaviour · cause unknown". The supervised model keeps predicting NORMAL with confidence about 1.00, no alarm, no part highlighted. This is the Phase 9 finding made visible: the supervised models are blind to a fault they have not seen, and the autoencoder reacts without being able to say what or where.

**Measured (2026-11-20 run, `demo_b_unseen_fault.json`; test `test_demo_b_unseen_fault_is_unusual_behaviour_only`):** the alert came 49 s of event time after the injection. At that moment the supervised model said NORMAL with probability 0.9998, and it raised no alarm during the whole observation. The alert switched on and off twice in the following minutes.

### The unusual-behaviour alarm raises false alarms and flaps

Be explicit about this before anyone is impressed:

- **It is the least trustworthy signal of the three.** On the clean test runs of Phase 9 the autoencoder alarm (99.9th percentile threshold) fired about **0.57 times per hour of normal operation** (95 % interval 0.31 to 0.89, `experiments/results/e5_explain.md`). On a real line that means a false "unusual behaviour" every couple of hours per machine, with no cause to look for.
- **It flaps.** The alert has no hysteresis: it switches on and off as the score crosses its threshold (two switches in the minutes after the alert in (b)). Once it has switched several times within five minutes, the dashboard shows it as "unusual behaviour (intermittent)" and keeps the indicator for 30 s after it clears. The stored alert events are unchanged.
- **It names no cause and no part.** The supervised model's attributions highlight a part; the autoencoder never does.

Treat it as "look at this machine", not as a diagnosis.

## (c) Operating-condition drift, before any overload fault

**Do:** on CNC-01 send **Set load** 120 % (the simulator's limit; the models were trained on loads up to about 96 %).

**Show:** after about two minutes a banner appears: "Model drift on CNC-01. The operating conditions differ from the data the models were trained on. Predictions … are marked DEGRADED". The banner shows the backend's message (the channel, its PSI against the alert level, since when). The machine's reliability badge in the Prediction tab turns DEGRADED. The predictions themselves still appear: what degrades is the claim that their calibration and sets hold. Set the load back to 55 % and, after about two more minutes, the banner and the DEGRADED flag clear.

**The ordering check.** The drift flag has to come **before** any overload-related fault: overheating, a trip, or a simulator WARNING or FAULT caused by the overload. Otherwise the twin would only say "conditions differ" after the machine had already said so itself. The check has two parts:

- **Physics, without a model** (`test_demo_c_load_step_stays_below_the_simulator_limits`, part of `uv run poe test`). The dev-stack simulator (same line, physics and seeded scenario as the demo) is held at 120 % on CNC-01 for 1260 s. That is longer than the demo can keep the load high (up to 600 s to raise the drift plus 60 s to see DEGRADED). The test asserts that CNC-01 never reaches WARNING or FAULT. The step was chosen from the physics before any model was involved: 120 % is far outside the trained 30–95 % range, the healthy CNC has no friction heat, and its steady temperature and current at 120 % stay below CNC-01's warning thresholds. If a change to the simulator makes the test fail, pick a smaller step between about 100 % and 120 % from the physics, never from model output.
- **Live** (`test_demo_c_load_drift_comes_before_any_overload_fault`). After the load is back, the test reads CNC-01's stored readings for the whole high-load period. It asserts that no WARNING or FAULT reading comes before the `model_drift` alert. It records the first such event (or none), the highest temperature and current, and how long the load was held.

**Measured (2026-11-20 run, `demo_d_load_drift.json`, before the ordering check was added):**

- The `model_drift` alert was raised **119 s** after the load change (the window is 120 s of operating rows plus a 30 s sustain; ADWIN dated the change at 34 s after the command).
- The predictions turned DEGRADED and no predicted alarm was raised during the drift.
- It cleared **135 s** after the load went back to 55 %.
- Both channels exceeded their alert levels in that run: the message reads speed PSI 8.74 against 7.88 and load PSI 9.10 against 6.00.
- TODO: the live ordering facts (first overload-related event, peak temperature and current) come from the next `uv run poe demo-tests` run.

**Say:** drift is measured on speed and load only (never on vibration or temperature, which a developing fault changes). It says "the conditions differ from the training data", not "the prediction is wrong". The false-drift rate and the detection of synthetic shifts are in `experiments/results/drift.md`. On the clean simulated runs of that report there was no false drift episode at the live margin, and every shift that left the training support was detected after about 90 to 110 s.

## (d) A what-if during the bearing fault, then apply it

**Do:** run (a) until the predicted WARNING is on. Open the What-if tab, select CNC-01, change **Set load** to 40 %, project 300 s, **Run what-if**. Compare "no change" with "with change". Then use *Apply as command…* (operators only, after a confirmation).

**Show:** the projection table gives, for both variants, when the simulator would reach WARNING and FAULT and when the model would raise its alarm. The chart draws the measured values solid and both projections dashed. After *Apply as command*, the live load drops and the live line can be compared with what the projection said.

**What the test checks (`test_demo_d_what_if_lower_load_during_the_bearing_fault`):**

- The projection is scored by the deployed model, and its projected load is lower than in the unchanged projection.
- A lower load does not bring the simulator's WARNING or FAULT forward. Bearing wear progresses with load^1.5 and vibration rises with load, so a lower load can only delay them.
- Applying the change sends it through the normal command path. The command is EXECUTED and appears in the command audit, and the live load drops.
- The test then watches the live line until the end of the projection and records the live WARNING / FAULT times next to the projected ones. It asserts only the structure: whether the lower load **removes** the fault is recorded, not assumed.

**Expect:** the lower load may only slow a bearing fault that is already in progress, not stop it (load_exponent 1.5 slows the progress; nothing reverses it). Say whatever the projection shows.

**Measured:** TODO: not run yet; the next `uv run poe demo-tests` run writes `demo_d_whatif_during_fault.json`.

**Say:** a projection is the simulator's future under one change, scored by the same model. It is a simulation, not a promise, and it only exists for simulated sources.

## (e) Extra: a healthy line, and a stop what-if

Not in the plan. It shows the quiet baseline and the simplest what-if.

**Show:** no alert of any kind for two minutes. In the What-if tab select CNC-01, change **Stop**, project 60 s, **Run what-if**. The table shows, per machine, the real state now and the projected states "no change → with change" as translucent "ghost" badges. With the stop, CNC-01 goes IDLE, the buffers fill or drain, and the inspection station is starved. **Nothing is sent to the machine.**

**Measured (2026-11-20 run, `demo_a_healthy_line.json`; test now `test_demo_e_healthy_line_and_stop_what_if`):** 120 s of watching a healthy line: no alert of any kind on any machine. The stop projection took about 6 s (the first request, which starts the what-if scoring workers) and showed CNC-01 IDLE and INSP-01 STARVED at +30 s, with the live line untouched.

## What to say about limits

- Simulator data only; no early warning was shown on real machines (Phase 10).
- Confidence is calibrated on clean simulated data and is not validated under drift, noise or an unseen fault. The DEGRADED flag is the dashboard's way to say so.
- A what-if is the simulator's future, not a forecast of a real machine.
- The unusual-behaviour alarm has a false-alarm rate of about one every couple of hours per machine on clean data and no cause.
- Drift detection needs 1 to 2 minutes and sees only speed and load.

## Screenshots

`uv run poe screenshots` writes the dashboard shots to `experiments/results/frontend_screenshots/dashboard/`, including (live inference, real time, about 25 minutes): `prediction_normal`, `prediction_unusual`, `alerts_unusual`, `twin_unusual_closeup`, `prediction_alarm`, `twin_alarm_closeup`, `alerts_predicted`, `prediction_warning` (scenarios (a) and (b)), and `whatif_projection`, `whatif_chart`, `whatif_probability`, `whatif_apply_confirm`, `prediction_drift`, `alerts_drift` (scenarios (e) and (c)).

## Re-running the demos as tests

The demo scenarios are opt-in (pytest marker `demo`, option `--run-demo`): `uv run poe test` skips them with a message.

```text
uv run poe up && uv run poe deploy-models
uv run poe demo-tests                  # all five, about 40 min, real time; writes experiments/results/demo_scenarios/
uv run poe demo-tests -k demo_d        # one scenario (extra arguments pass through to pytest)
```

Without the deployed bundles the demo tests skip with a message.

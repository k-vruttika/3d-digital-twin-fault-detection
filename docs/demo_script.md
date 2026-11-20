# Demo script: the digital twin with live predictions

A guided tour of Phase 11 in about 30 minutes of real time. Everything here runs on **simulated data**: the models were trained, calibrated and validated on the LineTwin simulator, and Phase 10 did not show early warning on real machines (`experiments/results/real_data.md`). Say so at the start of any demo.

The four scenarios below are also end-to-end tests (`tests/test_phase11_demo.py`, real-time simulator, real inference workers, real database). The times quoted are **measured by those tests** and stored in [`experiments/results/demo_scenarios/`](../experiments/results/demo_scenarios); they are event times of one run on one laptop, not guarantees.

## Set up (once)

```text
uv run poe up                  # TimescaleDB + Mosquitto
uv run poe deploy-models       # the three bundles: GRU, XGBoost, autoencoder (about 15 min)
uv run poe dev --source mock --inference    # backend + simulator, live inference on
uv run poe frontend            # dashboard on http://127.0.0.1:5173
```

Enter an operator key in the header (`LINETWIN_API_KEYS_OPERATOR`, see `docs/setup.md`) to send commands and inject faults. The simulator runs in real time: a demo needs patience, which is the point.

Reset between scenarios by restarting `uv run poe dev ...` (an injected fault stays until the simulator restarts).

## (a) A healthy line, and a what-if

**Show:** the Prediction tab. After 30 s of running data every machine gets a prediction: NORMAL with confidence about 1.00, reliability OK, no alarm. Say what the confidence text means: "Most likely NORMAL (1.00) · cannot rule out …" lists the rest of a class-conditional set, which guarantees coverage per class, so a rare class can appear at low probability.

**What-if:** open the What-if tab, select CNC-01, change **Stop**, project 60 s, **Run what-if**. The table shows, per machine, the real state now and the projected states "no change → with change" as translucent "ghost" badges. The chart draws the measured values solid and the projections dashed. With the stop, CNC-01 goes IDLE, the buffers fill or drain, and the inspection station is starved. **Nothing is sent to the machine:** *Apply as command…* (operators only) asks for a confirmation first; cancel it.

**Measured (test `test_demo_a_healthy_line_and_what_if`):** 120 s of watching a healthy line: no alert of any kind on any machine. The stop projection took about 6 s (the first request, which starts the what-if scoring workers) and showed CNC-01 IDLE and INSP-01 STARVED at +30 s, with the live line untouched.

**Say:** a projection is the simulator's future under one change, scored by the same model. It is a simulation, not a promise, and it only exists for simulated sources.

## (b) An unseen fault: unusual behaviour, cause unknown

**Do:** in the Scenario panel inject `lubrication_loss` on CNC-01 (progression 720 s). This fault type was never in the training data.

**Show:** within about a minute a purple outline appears around the **whole** CNC and the Alerts tab shows "unusual behaviour · cause unknown". The supervised model keeps predicting NORMAL with confidence about 1.00, no alarm, no part highlighted. This is the Phase 9 finding made visible: the supervised models are blind to a fault they have not seen, and the autoencoder reacts without being able to say what or where.

**Measured (test `test_demo_b_unseen_fault_is_unusual_behaviour_only`):** the alert came 49 s of event time after the injection; the supervised model said NORMAL with probability 0.9998 at that moment and raised no alarm during the whole observation; the alert switched on and off twice in the following minutes.

### The unusual-behaviour alarm raises false alarms and flaps

Be explicit about this before anyone is impressed:

- **It is the least trustworthy signal of the three.** On the clean test runs of Phase 9 the autoencoder alarm (99.9th percentile threshold) fired about **0.57 times per hour of normal operation** (95 % interval 0.31 to 0.89, `experiments/results/e5_explain.md`). On a real line that means a false "unusual behaviour" every couple of hours per machine, with no cause to look for.
- **It flaps.** The alert has no hysteresis: it switches on and off as the score crosses its threshold (two switches in the minutes after the alert in (b)). The dashboard shows it as "unusual behaviour (intermittent)" once it has switched several times within five minutes, and keeps the indicator for 30 s after it clears. The stored alert events are unchanged.
- **It names no cause and no part.** The supervised model's attributions highlight a part; the autoencoder never does.

Treat it as "look at this machine", not as a diagnosis.

## (c) A known fault, and the order of the alerts

**Do:** inject `bearing_wear` on CNC-01 (progression 720 s).

**Show, in this order:**

1. **Unusual behaviour** (purple, whole machine) while the supervised model still says NORMAL and the twin shows RUNNING.
2. **Predicted WARNING** (the evaluated alarm rule: +30 s argmax WARNING or FAULT for 3 s in a row), still while the twin shows RUNNING. The Prediction tab shows the explanation: the spindle bearing is highlighted in the 3D view, labelled with the time of the window it explains. If a newer explanation is late the last one stays while the alarm is on.
3. **The simulator's own WARNING**: the bearing outlined in the warning colour.

**Measured (test `test_demo_c_known_fault_alert_order`, which asserts this order):**

| Event | Seconds after injection (event time) |
|---|---|
| Unusual behaviour raised | 173 |
| Predicted WARNING raised | 444 |
| Simulator WARNING | 501 |

So the predicted warning came **57 s before** the simulator's own WARNING, and the autoencoder's unusual-behaviour alert came 271 s before the predicted one. The strongest explained part at the alarm was `spindle_bearing` (vibration, attribution 0.74; the next parts were below 0.01).

**The ordering check** is what the test enforces: injection, then unusual behaviour at or before the predicted warning, then the predicted warning strictly before the simulator's WARNING, with the bearing as the top part. If a later change to the models or the alarm rule breaks that order, the test fails and says which event came first.

**Say:** the lead time is a property of this simulator, this fault type and this progression (720 s, the shortest in training). The evaluated lead times and miss rates are in `experiments/results/robustness.md` and `e5_explain.md`. The stricter prediction gate (only fully operating windows) costs a few seconds of lead (`live_inference.md`).

## (d) Operating-condition drift

**Do:** on CNC-01 send **Set load** 120 % (the limit; the models were trained on loads up to about 96 %).

**Show:** after about two minutes a banner appears: "Model drift on CNC-01. The operating conditions differ from the data the models were trained on. Predictions … are marked DEGRADED". The banner shows the backend's message (the channel, its PSI against the alert level, since when). The machine's reliability badge in the Prediction tab turns DEGRADED. The predictions themselves still appear: what degrades is the claim that their calibration and sets hold. Set the load back to 55 % and, after about two more minutes, the banner and the DEGRADED flag clear.

**Measured (test `test_demo_d_load_drift_degrades_and_clears`):**

- The `model_drift` alert was raised **119 s** after the load change (the window is 120 s of operating rows plus a 30 s sustain; ADWIN dated the change at 34 s after the command).
- The predictions turned DEGRADED and no predicted alarm was raised during the drift.
- It cleared **135 s** after the load went back to 55 %.
- Both channels exceeded their alert levels in that run: the message reads speed PSI 8.74 against 7.88 and load PSI 9.10 against 6.00.

**Say:** drift is measured on speed and load only (never on vibration or temperature, which a developing fault changes). It says "the conditions differ from the training data", not "the prediction is wrong". The false-drift rate and the detection of synthetic shifts are in `experiments/results/drift.md`: on the clean simulated runs of the report there was no false drift episode at the live margin, and every shift that left the training support was detected after about 90 to 110 s.

## What to say about limits

- Simulator data only; no early warning was shown on real machines (Phase 10).
- Confidence is calibrated on clean simulated data and is not validated under drift, noise or an unseen fault. The DEGRADED flag is the dashboard's way to say so.
- A what-if is the simulator's future, not a forecast of a real machine.
- The unusual-behaviour alarm has a false-alarm rate of about one every couple of hours per machine on clean data and no cause.
- Drift detection needs 1 to 2 minutes and sees only speed and load.

## Screenshots

`uv run poe screenshots` writes the dashboard shots to `experiments/results/frontend_screenshots/dashboard/`, including (live inference, real time, about 25 minutes): `prediction_normal`, `prediction_unusual`, `alerts_unusual`, `twin_unusual_closeup`, `prediction_alarm`, `twin_alarm_closeup`, `alerts_predicted`, `prediction_warning` (scenarios b and c), and `whatif_projection`, `whatif_chart`, `whatif_probability`, `whatif_apply_confirm`, `prediction_drift`, `alerts_drift` (scenarios a and d).

## Re-running the demos as tests

```text
uv run poe up && uv run poe deploy-models
uv run pytest tests/test_phase11_demo.py -s     # about 30 min, real time
LINETWIN_DEMO_RESULTS_DIR=experiments/results/demo_scenarios uv run pytest tests/test_phase11_demo.py   # also writes the measured JSON
```

Without the deployed bundles the demo tests are skipped with a message.

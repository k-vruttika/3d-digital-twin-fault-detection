# ADR 0031: What-if projections, drift detection and the demo scenarios

- **Status:** Accepted (what-if: Group 2, drift: Group 3 of Phase 11 Part C). The demo section is added with Group 4.
- **Date:** 2026-11-20
- **Phase:** 11 Part C

## Context

The dashboard can show predictions (ADR 0030). Operators also want to ask "what if I stop this machine, or change its speed?" before doing it. CLAUDE.md §3 puts this on the architecture: dashboard → `POST /whatif` → simulator snapshot → fast-forward copy → model → projected trajectory. Constraints:

- **Simulation only.** Real equipment cannot be forked, so the capability belongs to the source adapter like scenario injection (ADR 0015); other sources answer 409.
- **Minimal mode:** it must keep working with no ML. Without inference a what-if shows the simulator only.
- **The live line must not be disturbed:** no command is sent, nothing is stored, and the live inference (buffers, alarms, alerts) is untouched.
- **The evaluated alarm rule** must be the one applied to projections (ADR 0022 / 0030), not a new one.
- **Safety:** a change must pass the same rules as a command.

## Decision

### What-if

- **One change per request**, applied at the fork point (now): a command type, its parameters, a machine and a projection length (`config/whatif.yaml`: default 120 s, maximum 300 s). That is exactly what "apply as command" can send. Scheduling changes later in the run is not offered.
- **Validation in the backend first:** the command contract (a missing speed is a 422), the safety rules and limits of `commands.yaml` against the machine's current state (`SafetyRules`; STOP always passes), and the horizon limit. The simulator's own verdict is checked as well.
- **The fork lives in the simulator** (`twin_sim/whatif.py`, mock server `POST /whatif`, contracts in `twin_common/whatif.py`):
  - the state is copied and the recent history read in one tick-boundary call, so the history ends at the fork point;
  - an unchanged and a changed copy are built from the snapshot and fast-forwarded in a thread, so the live simulation keeps its pace;
  - the random-number streams are restored, so the unchanged copy equals what the live simulation does next (tested).
- **Scoring in the backend with its own worker pool** (`whatif/scoring.py`): the same bundles, gate and alarm rule (`MachineAlarms`), no attributions. The pool starts on the first request (a few seconds) and is reused; the live pool is not shared, because each projection needs empty buffers and a what-if must not delay or alter live predictions. Each variant is replayed from empty buffers over the recent readings plus its projection.
  - **Cost:** a second set of worker processes (memory) once a what-if has been asked.
  - **Failure:** if the workers cannot start or answer, the projection is still returned, with a note that it shows the simulator only.
- **Response:** per machine and variant, the projected readings per second, the projected simulator states at +10 s / +30 s / the end ("ghost states"), the first simulator WARNING / FAULT and, with a model, calibrated P(WARNING) / P(FAULT) at +30 s, top class, the evaluated alarm, time to fault, and the first alarm.
- **Status codes:** 404 unknown machine, 409 source without the capability (or no `whatif.yaml`), 422 refused (contract, safety, limits, simulator), 502 the simulator could not run it. Viewer role is enough: a what-if changes nothing.
- **Dashboard:** ghost states, dashed projection lines, outcome text, and "apply as command" for operators in live mode after a confirmation, through `POST /commands`.

### Drift detection

**Question.** Is the machine running under conditions the models were not trained on? If so, the evaluated calibration and conformal coverage are not validated for it (ADR 0022), so the predictions are marked `reliability = DEGRADED` and a `model_drift` alert is raised.

- **What is monitored: speed and load only** (`deploy.yaml` `drift.channels`). Vibration, temperature and current are health signals: a developing fault changes them by design, so drift on them would fire during every real fault and say nothing about the operating condition.
- **PSI against the training distribution** of the machine's kind: 10 quantile bins of the training rows inside a 0.1 %–99.9 % support, plus an overflow bin on each side, over a rolling window of 120 operating rows, with additively smoothed counts. The overflow bins matter: without them a load far above the trained range lands in the top bin and looks like "the top decile".
- **The alert level is calibrated, not 0.25.** A machine sits in one load band for long stretches, so even a clean window differs from the pooled training distribution and the textbook 0.25 would flag clean runs (a unit test and the first worker test showed this). The alert level per kind and channel is the 99.9 % quantile of the PSI of *clean validation windows* (`calibrate_thresholds`), times a live `threshold_margin`.
- **Decision rule** (one class, `Decider`, shared by the live monitor and the report): PSI at or above the alert level on any channel for 30 s raises drift; it clears below 0.6 × the alert level on every channel for 60 s.
- **ADWIN** (`river`, delta 0.002) runs on the standardised value of each channel and **does not decide**. It dates the change: the drift's `since` is ADWIN's last change time on a channel over its alert level. A decision by ADWIN alone would flag every legitimate change of band; PSI against the training range is the decision.
- **Where it runs:** in the monitor worker (the autoencoder bundle), because the reference is the autoencoder's training split (the E1 dataset, all load bands). The reference, the window and the alert levels are part of that bundle; a bundle built before this part has none and drift detection is simply off (`drift.enabled` in `config/inference.yaml`). Only operating rows (RUNNING / WARNING) count; the monitor keeps its window across short stops. Backend and dashboard never see ML code.
- **Consequence for the supervised models:** `gru_e4` (trained on low and medium load) is monitored against the E1 reference, not its own training range, so a high-load run is not drift for it.
- **Backend:** `InferenceService` takes each operating row's drift report; `reliability` is DEGRADED while the data quality is DEGRADED or the machine drifts; `model_drift` goes through the ISA-18.2 manager like the other inference kinds. The alarm rule and the numbers it uses are unchanged. Drift ends when the monitor reports it ended, or when the machine's link is lost or the workers restart (state unknown).
- **Dashboard:** a banner while any machine has an active `model_drift` alert, with the backend's message (channel, PSI against the alert level, since when), the DEGRADED reliability badge, and an explanation in "About these predictions".
- **Report** (`uv run poe drift-report` → `experiments/results/drift.md`): false drift on the clean validation and test runs over a margin grid, the live margin chosen by a rule on *validation* runs only (the smallest margin with at most 0.1 false episodes per machine-hour), and detection of synthetic input shifts of the clean test rows. The shifts are edits of recorded rows, not a physical simulation, and the report says so.

### Measured (from the report; simulator data only)

- Clean runs, margin 1.0 (the live setting, chosen by the rule): 0.064 false drift episodes per machine-hour on validation runs and 0.022 on test runs (2 of 120 runs). Margins from 1.25 gave none on either split, at the price of lower sensitivity.
- A shift that leaves the training support was detected in every case (load +5 to +45 points, speed +5 % to +40 %), with a median delay of about 1–2 minutes (the window is 120 s, plus the 30 s sustain), and ADWIN dated the change 20 s after it on median.
- A shift that stays inside the trained range is, by design, not drift. For the load that is most runs below +30 points: the trained loads cover 10–96 %.

## Consequences

- **A projection is the simulator's future, not a promise.** The model's numbers on it are as trustworthy as on live data from the same simulator, and no more. The dashboard says it is a simulation.
- **The what-if shares the simulator's world model:** with the mock source it is the same simulator that produces the data, so projected states are what the simulator would do (including a planned fault later in the run). With real equipment this feature would need a different mechanism and does not exist.
- **Worker memory:** a second pool after the first what-if with inference on.
- **Not covered (what-if):** several changes at once, delayed changes, and projections from replay time (the fork is always the running simulation).
- **Drift is not fault detection.** It says the conditions differ from the training data, not that a prediction is wrong, and it is blind to a drift of the health signals themselves.
- **Detection delay is minutes:** a 120 s window plus a 30 s sustain. A sudden condition change is flagged after about 1–2 minutes, not at once.
- **False drift is rare but not zero** (0.022–0.064 per machine-hour, a few independent runs); the margin trades it against sensitivity.

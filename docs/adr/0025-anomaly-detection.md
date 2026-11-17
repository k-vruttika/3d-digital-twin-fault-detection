# ADR 0025: Anomaly detection and the combined alarm policy

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** 9 (unseen faults, anomaly detection, explainability)

## Context

A supervised model can only warn of patterns it was trained on. ADR 0024 tests this with the held-out `lubrication_loss` fault (E5). Phase 8 also showed that the supervised models cannot tell sensor drift from degradation. Phase 9 adds detectors that model normal operation only, so they can, in principle, flag any departure from it: an unseen fault, but also a sensor problem or an operating condition not seen in training. The protocol, settings and primary outcomes are fixed in ADR 0024; this ADR records the design and its implementation.

## Decision

### Training windows: normal operation only

`twin_ml.models.anomaly.normal_windows(split, no_fault_code, horizon, states)` keeps a window when all of these hold:
- its last row is in a normal state (RUNNING, `e5.yaml` `anomaly.normal_states`);
- no row of the window has an active fault (`fault_type` = none on every row; computed with a cumulative sum over the rows, so it is exact per window);
- the health target at the primary horizon (+30 s) is NORMAL.

The training split gives the training windows. The validation split, with the same rule, gives the scores the thresholds are taken from. On the E1 dataset that is 78,759 training and 64,449 validation windows. The training split's far-normal subsample (Phase 7, keep 0.25) applies, so the detectors see the same windows the supervised models saw, without the weights.

### Detectors

| Detector | Input | Score | Module |
|---|---|---|---|
| Isolation Forest | the 50 window summary features | `-score_samples` | `models/anomaly.py` (scikit-learn, imported lazily) |
| Autoencoder | the 6 normalised value channels × 30 s, flattened, plus the kind one-hot as a condition on encoder and decoder | mean squared reconstruction error over observed values | `models/autoencoder.py` (torch) |

- **Autoencoder:**
  - MLP 183 → 64 → 16 (+3) → 64 → 180, ReLU.
  - Adam, a fixed 20 epochs, seeded batch order, torch deterministic algorithms; the epoch with the lowest validation normal-window loss is kept.
  - A value channel counts as unobserved where its sensor's missing flag is set (`vib_norm` follows vib).
  - `sensor_errors()` gives the squared error per sensor (vib_norm folded into vib). It is used only for the exploratory per-sensor analysis; a unit test shows that a shift in one input can raise the reconstruction error of another, so these shares are not attributions in a strict sense.
- **Separate config type:** the detector configs are a separate discriminated union (`AnomalyModelConfig`, `load_anomaly_config`), not part of `ModelConfig`. `train-all` and the E1 trainer therefore can never pick them up.

### Thresholds and alarms

- `AnomalyThresholds` stores one threshold per machine kind and percentile (99.0, 99.5, 99.9) of the validation normal-operation scores. 99.9 is primary (ADR 0024).
- Alarm = score above the window kind's threshold. It is sustained exactly like the supervised alarm (`sustain_s` = 3 s), and lead time, miss rate and false alarms per hour come from the same early-warning code.

### Combined policy

- **Rule:** per window, the supervised model's *sustained* alarm OR the detector's *sustained* alarm.
- **How it is computed:**
  - Every study run stores its sustained alarms (primary horizon for supervised models, every percentile for detectors) as `alarms.npz` (bit-packed, about 50 KB per condition), with the window count and a digest of the windows' series, run and end time.
  - `python -m twin_ml.experiments.e5 --combine` reads the two source runs' alarms, checks that the digests match, ORs them, and computes the per-run sums with `e5_eval.alarm_run_sums`. That function calls the normal early-warning code with a sustain of one window, so an already sustained alarm is not delayed a second time. A unit test checks that this path gives exactly the early-warning statistics of the direct path.
  - One MLflow run is logged per pair (study `e5_combined`), tagged with both source run ids.
- **Pairs:** E1 and augmentation-trained models with the E1-trained detectors; E4 models with the E4-trained detectors. Seed *i* is paired with seed *i*; the threshold rule (one run) is paired with every detector seed.

### Sensor-fault flags (`sf_drift`)

- `e5_eval.drift_flag_sums`: per run, the number of sensor-fault segments, the number flagged, and the time from segment start to the first flag.
- A segment is flagged by a sustained alarm on a *negative* window inside it (operating, no machine fault, NORMAL ahead). An alarm caused by a real fault therefore does not count as flagging the sensor problem.
- The same function is applied to supervised models, detectors and combined policies. For a detector, a flag is the desired outcome; for a fault-warning policy, the same alarm is a false fault warning (`sf_false_alarms_per_h`).

### Runner

- **Runner:** `twin_ml.experiments.e5` (parent; poe `e5`) runs every (study, model, seed) in its own process (`e5_worker`, ADR 0020). It resumes through a study config hash that includes the E5 data's content hash.
- **Smoke runs** (`--smoke`) go to `P9_smoke` and are never part of a result.
- **Reuse:** supervised models are only reused (`find_run` on their source experiment), never trained, and each must reproduce its stored clean test predictions (`check_predictions`).

## Consequences

- **Anomaly results across studies:** detector and combined results use the same per-run sums and bootstrap as every Phase 8 result, so they can be paired with the supervised models on the same runs.
- **Training cost is small:** a whole detector study run (training plus three test conditions) took 34–48 s per seed. The measured times are in `docs/PROGRESS.md`.
- **The autoencoder's validation loss was still falling at epoch 20.** The budget was fixed in advance (ADR 0024) and is not changed after results; a longer budget could lower the normal-score spread.
- **Window-level only:** no detector is retrained online, and drift in normal operation (for example, slow wear of a healthy machine) would raise its scores over time. That is out of scope here; online drift detection is planned with `river` (ADWIN).
- **Results:** in the generated [e5_explain.md](../../experiments/results/e5_explain.md) (Part C).

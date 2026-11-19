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
- **Results:** below, from the generated [e5_explain.md](../../experiments/results/e5_explain.md).

## Results (2026-11-17)

### Primary (ADR 0024 § Pre-registration; detectors at the 99.9th percentile)

| | E5 miss rate | E5 lead vs sim WARNING | Clean false alarms per hour |
|---|---|---|---|
| Autoencoder | 0.000 [0.000, 0.000] | 656 s [617, 693] | 0.57 [0.31, 0.89] |
| Isolation Forest | 0.992 [0.978, 1.000] | 324 s (from 2 / 1 / 0 detected episodes in seeds 0 / 1 / 2, per the runs' logged `e5.p99.9.miss_rate`) | 0.23 [0.12, 0.38] |
| Threshold rule (reference) | 0.042 [0.010, 0.075] | 23 s [2, 47] | 0.00 |

- **Autoencoder:** it detects every E5 episode, early.
- **Combined policy:** "supervised OR autoencoder" therefore has a 0 % E5 miss rate for every supervised model, E1 and augmentation-trained (paired vs the model alone, 3/3 seeds). The cost is +0.56 to +0.57 false alarms per hour on clean data.
- **Isolation Forest:** at the pre-registered threshold it is nearly useless. It misses 0.992 of E5 episodes and 0.974 of the clean seen-type episodes, and an OR with it changes no miss rate significantly.

### Exploratory

- **Percentile trade-off:**
  - The Isolation Forest at p99 detects nearly all E5 episodes (miss rate 0.003) at 0.91 false alarms per hour.
  - The autoencoder at p99 / p99.5 has 0.66 / 0.54 false alarms per hour and still misses nothing.
  - These percentiles were exploratory by design, so the primary result stands unchanged. The autoencoder's clean miss rate at p99.9 is 0.077 [0.011, 0.184]: it misses some seen-type episodes that the supervised models catch.
- **Score quality:** AUPRC on operating windows is 0.835 (autoencoder) and 0.272 (Isolation Forest) on clean data, 0.357 and 0.149 on E5. The Isolation Forest's summary features separate precursors poorly.
- **Load shift (E4, detectors trained on low + medium load):**
  - Autoencoder false alarms per hour: 0.43 on `e4_in` vs 1.10 on `e4_high` (+0.67 [0.20, 1.07], 3/3).
  - Isolation Forest: 0.23 vs 0.51 (2/3).
  - A load band outside training alone raises the anomaly false alarms; the supervised E4 models stay at 0.00–0.03.
- **Sensor drift (`sf_drift`):**
  - The autoencoder flags 0.239 of the drift segments (time to flag 241 s), but every such alarm also counts as a false fault warning: 2.63 per hour inside drift segments, against 0.22–0.97 for the supervised models.
  - The Isolation Forest flags 0.046.
- **Per-sensor reconstruction-error shares (ADR 0024 amendment):** each re-fitted autoencoder reproduced its stored alarms bit for bit. Unit = episode or flagged segment; counts from seed 0.
  - **Clean, seen types:** the largest share is the fault's leading sensor in every episode (26 of 26; unit agreement 1.000; window agreement 0.996).
  - **E5:** it never is. The largest share points at vibration, not current, in all 120 episodes: mean shares vib 0.91, current 0.03; unit agreement 0.000, window agreement 0.001. The autoencoder sees the stick-slip vibration bursts, not the current rise the fault trips on.
  - **`sf_drift`:** 23 of 95 drift segments are flagged (seed 0).
    - Unit agreement 0.779 [0.603, 0.936], all of it from vibration drifts (18 of 18 agree).
    - Temperature drifts (3) and current drifts (2) are attributed to vib / load instead (0 agree).
  - The shares are not attributions in a strict sense (see Decision). They tell an operator where the reconstruction failed, which for E5 and for non-vibration drift is not the sensor that is actually wrong.
- **Consequence for Phase 11:** keep autoencoder alarms as a separate "unusual behaviour" category without a part highlight (ADR 0024 § Outcome).

## Addendum: real data (Phase 10, 2026-11-19)

Same architectures and configs, input sizes from the channel schema (ADR 0028). The detectors were fitted on normal training windows of each dataset, with thresholds from normal validation windows. Protocol: ADR 0027. Numbers: the generated [real_data.md](../../experiments/results/real_data.md).

- **MetroPT-3** (one real compressor, 4 air-leak failures, lookback 6 h, p99.9):
  - **Autoencoder:**
    - warns of 3 / 1 / 3 failures (seeds 0 / 1 / 2), against 1.01 / 1.06 / 0.84 expected by chance, at 0.57–0.86 false alarms per day;
    - the only full-lead "warning" (#1) is an anomaly already present when the data resumed after a gap (ADR 0027 § Outcome);
    - it flags every failure *during* the reported interval (#1–#3 within 0–2 min of the start, #4 after 110–172 min).
  - **Isolation Forest:** warns of nothing and flags 1–2 failures during the interval.
- **Azure PdM** (+24 h, lookback 48 h, p99.9):
  - **Autoencoder:** detection rate 0.000 (0.090 at p99).
  - **Isolation Forest:** 0.139 at p99.9, 0.281 at p99.
  - **Combined policy:** "best supervised OR autoencoder" equals the supervised model's detection and adds 0.0007 false alarms per day.
- **Machine holdout (Azure):** the autoencoder retrained on 80 machines detects 0.000 on the 20 new ones.
- **Transfer from the simulator:** the simulator result does not transfer.
  - On the simulator the autoencoder caught every unseen-fault episode (miss rate 0.000).
  - On real data at the same threshold rule, it gave no early warning beyond chance (MetroPT-3) or none at all (Azure). The p99.9 threshold from normal validation windows is far above the precursor scores: on MetroPT-3 failure #4 the score rises for about 14 h before the start but crosses p99.9 only inside the failure.
  - Lower percentiles warn of all four MetroPT-3 failures at 2.5–6.3 false alarms per day, and on Azure they stay far below the supervised models.
- **False alarms by month (MetroPT-3, autoencoder):** they are highest in August (1.40 per day vs 0.34–0.71 in April–July). That is consistent with the expected rise as normal operation drifts away from the training period (§ Consequences, "Window-level only"), but it is one series, not a drift test.
- **Consequence for Phase 11:**
  - Keep the "unusual behaviour" category, but present it as a *during-event* detector, not an early warning, and do not carry the simulator's unseen-fault coverage into any claim about real machines.
  - The detector threshold needs per-deployment validation, and the drift monitoring (ADWIN) planned for the live twin is needed.

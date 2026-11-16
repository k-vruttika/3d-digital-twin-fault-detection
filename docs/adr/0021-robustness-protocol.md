# ADR 0021: Robustness protocol: perturbations, sensor-fault sets, augmentation and pre-registered outcomes

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 8 (robustness, calibration and conformal prediction)

## Context

The research question asks how reliably the twin warns of faults when its input is noisy or has gaps, when the operating conditions shift, and whether the reported confidence stays trustworthy. Phase 7 evaluated every model on clean data only (E1). Phase 8 stresses the same models and needs a protocol that is fixed before any result is seen: the perturbations, the test sets, the statistics, and which outcomes count as primary.

## Decision

### Perturbations: raw series, before windowing, test data only

- **Where they act:** `twin_ml.data.perturb` changes a machine series' raw sensor array (T × 5, NaN = missing) before forward fill, normalisation and windowing. The test split is then windowed again with the clean dataset's normaliser, vocabulary and config (`twin_ml.data.test_sets`). Missing masks and forward fill therefore behave as on live data.
- **Types and levels** (`config/models/robustness.yaml`):

| Condition | What it does | Levels |
|---|---|---|
| `noise_<p>` | additive Gaussian noise, σ = p % of the sensor's normal-operation std (per machine kind; RUNNING rows without an active fault in the clean E1 training split); unclipped | 10, 20, 30 |
| `mcar_<p>` | each sensor value removed independently with probability p % | 5, 10, 20, 30 |
| `dropout_<p>` | single-sensor gaps of U[10, 60] s placed until p % of the sensor-time is covered | 5, 10, 20, 30 |
| `stuck_<p>` | a single sensor repeats its last value for U[30, 300] s, until p % of the sensor-time is covered | 5, 10, 20, 30 |

- **Shared axis:** dropout and stuck use sensor-time coverage as their level, so they sit on the same axis as MCAR.
- **Seeding:** every draw comes from `SeedSequence(808, (crc32(condition), crc32(run), crc32(machine)))`. Every model sees the identical perturbed test set, which makes paired comparisons valid, and the result does not depend on processing order.

### Simulator sensor-fault test sets

- **Generation:** the 200 runs are regenerated with base seed 42 under three scenarios (`config/scenarios/sf_all.yaml`, `sf_seen.yaml`, `sf_drift.yaml`). Each is a copy of `default.yaml` with only the `sensor_faults` section changed, at 1 per machine-hour:

| Set | Fault types |
|---|---|
| `sf_all` | stuck, drift, dropout |
| `sf_seen` | stuck, dropout (seen in augmentation) |
| `sf_drift` | drift (never seen in training) |

- **Same physics:** the simulator plans sensor faults on their own RNG stream, so physics, machine faults and load are identical to `data/sim`. The build checks this. Every test series must have exactly the clean run's timestamps, states and fault types, or the evaluation stops (`check_same_labels`).
- **Which runs:** only the test-split runs are used.
- **Sensor-fault label:** the simulator's `sensor_fault` column is read as an evaluation-only label. It gives the false alarms per hour inside sensor-fault segments, and is never a model input.

### Studies

| Study | Models | Test conditions |
|---|---|---|
| `e1models` (E2 noise, E3 missing data, sensor faults) | the E1 models, no retraining | clean, the full grid, `sf_all`, `sf_seen`, `sf_drift` |
| `e4` (condition shift, ADR 0023) | retrained on low + medium load runs | high-load runs, low + medium test runs |
| `aug` (ablation) | retrained on augmented training data | as `e1models` |

- **Every study includes both baselines** (persistence, threshold). Persistence reads the true state at the window end and is therefore unaffected by sensor perturbations; it is kept as a reference.
- **Reusing E1 models:**
  - XGBoost and the sequence models are loaded from their E1 artifacts.
  - Random Forests are not saved (ADR 0020); they are retrained from the same seed.
  - Every reused model must reproduce its stored clean test predictions: identical argmax, probabilities within 1e-5. Otherwise the run fails.
- **Augmentation (ablation):**
  - Each training series is replaced, with probability 0.5, by a perturbed copy. The type is drawn uniformly from {noise, mcar, dropout, stuck}, and the level uniformly from that type's test-grid range.
  - The training size is unchanged; validation stays clean.
  - The normaliser is fitted on the clean training series.
  - Augmentation uses its own seed, never the test seed.
  - The report separates perturbation types seen in augmentation (noise, MCAR, dropout, stuck; `sf_seen`) from unseen ones (`sf_drift`).

### Statistics

- **Bootstrap:** all CIs come from the run-level bootstrap (1000 resamples, 95 %, percentile), as in E1. Every metric is computed from per-run sums, so ECE and conformal coverage are bootstrapped over runs too (`evaluate.run_sums`).
- **Paired differences** (same resampled runs on both sides):
  - clean vs perturbed, per model and seed;
  - each model vs the threshold baseline under the same condition;
  - clean-trained vs augmentation-trained under the same condition.
- **Averaging over seeds:** results are averaged over seeds as in E1 (mean of the per-seed point and CI bounds), with the number of seeds whose own CI excludes zero.
- **Episode counts:** the number of episodes behind each figure is reported next to it.

## Pre-registration (fixed 2026-11-15, before any Phase 8 study run)

**Primary outcomes.** Each one is reported as a paired difference with a 95 % CI, at horizon +30 s:

1. **Early warning, each learned model vs the threshold baseline,** paired:
   - miss rate;
   - lead time relative to the simulator's threshold WARNING (`rel_lead_mean_s`);
   - false alarms per hour.
2. **Calibration:** ECE after calibration (`ece_cal`, top-label, 15 bins, operating windows), per model.
3. **Conformal:** FAULT-class coverage of the LAC, APS and Mondrian LAC sets at α = 0.1 (`cov_lac_FAULT`, `cov_aps_FAULT`, `cov_mondrian_lac_FAULT`), per model.

**Primary conditions:** `noise_20`, `mcar_20`, `sf_all` and the E4 high-load test (`e4_high`).

**Everything else is exploratory.** This covers:
- other levels and types;
- the +10 s horizon;
- macro F1, fault recall and lead time;
- set sizes and time-to-fault intervals;
- the clean vs perturbed degradation curves;
- the augmentation comparison.

**Rules:**
- The report labels every section and table as primary or exploratory.
- The list is stored in `robustness.yaml` (`primary:`), and the report reads it from there. A test checks that the config matches this ADR.
- The list is not changed after results are seen. No setting in `robustness.yaml` is tuned on test results.

## Amendment (2026-11-15, after the Part A results): exploratory additions only

Requested after the Part A results were seen. **None of them changes the pre-registered primary outcomes or conditions above.** All are labelled EXPLORATORY in the report.

- **Report additions:**
  - FAULT-class ECE next to every top-label ECE;
  - a case study of the `sf_all` episode that every model missed;
  - perturbation strength relative to the fault signal;
  - a note that simulator drift raises false alarms for every model, as a Phase 9 motivation.
- **Signal-relative noise grid** (`signal_noise` in `robustness.yaml`; studies `e1models_snoise` and `aug_snoise`):
  - **Noise:** Gaussian, with sigma per machine kind and sensor equal to 10 / 25 / 50 / 100 % of that sensor's typical rise before FAULT.
  - **Typical rise:** computed on the clean *training* episodes (`perturb.signal_scale`). A sensor uses its median |rise| over the episodes of the fault types it leads (largest median |rise| / normal-operation std), otherwise over all of the kind's episodes.
  - **Scope:** evaluation only, for the E1 models and the augmentation-trained models.
  - **Purpose:** to find where early warning breaks down. The original noise grid turned out to be small against the fault signal (at most about 5 % of the precursor rise).
- **Config hashes:** extension studies (`extension: true`) and the `signal_noise` settings are kept out of the shared part of the study config hash. Adding them therefore leaves every earlier run's hash, and so its validity, unchanged.

## Second amendment (2026-11-16, after the Part C results): uniform noise in std units

Requested after the Part C results were seen. Exploratory and evaluation-only; **the pre-registered primary outcomes and conditions are unchanged.**

- **Grid** (`std_noise` in `robustness.yaml`; studies `e1models_stdnoise` and `aug_stdnoise`): Gaussian noise with sigma = 1, 2, 4, 8 × the sensor's normal-operation std, the same multiple for every sensor.
- **Why:** to locate where false alarms start rising between the original noise grid (at most 0.3 × std) and the signal-relative grid. The signal-relative grid is many std on some sensors and a fraction of one on others.
- **One axis with the original grid:** it uses the same scale (normal-operation std) as `noise_<p>`. The report therefore puts both grids on one axis, with the `noise_<p>` points taken from the `e1models` / `aug` studies (same models, seeds and test runs).
- **Onset rule, fixed before the onset table was generated:** the lowest level at which the paired Δ false alarms per hour is positive and its CI excludes zero, in at least 2 of 3 seeds (1 of 1 for the threshold rule).
- **Config hashes:**
  - The `std_noise` settings are kept out of the shared hash.
  - A study's extension hash adds them only if the study lists that grid.
  - So every earlier run keeps its hash, the signal-noise studies included. Checked with `--dry-run --skip-existing` on all five earlier studies (0 runs to do) and in `test_ml_signal_noise.py`.

## Consequences

- **Reproducible from config:** every perturbed test set can be regenerated from the config and seed, and nothing perturbed is stored. Building one takes a few seconds per condition.
- **Cost:** the RF retraining adds about 2 minutes per seed to the e1models study.
- **Why perturbations are synthetic:** the noise and gap types are synthetic and independent across sensors. Correlated or physically caused sensor faults are covered only by the simulator sets.
- **Persistence is unaffected by construction.** Its numbers under perturbation equal its clean numbers.

## Outcome (2026-11-16, Phase 8 complete)

All studies in this ADR ran as specified: `e1models`, `e4`, `aug`, and the extension studies `e1models_snoise` and `aug_snoise`. Every number is in the generated [robustness.md](../../experiments/results/robustness.md); `docs/PROGRESS.md` § Phase 8 has a short record. Findings that affect later phases:

- **Augmentation (exploratory):**
  - On clean data, misses and false alarms do not change. Macro F1 is 0.01–0.02 lower, but the CIs include zero. The FAULT-class ECE of the GRU and Transformer is slightly higher (under 0.001).
  - On the types it saw, it mainly cuts the false alarms under stuck sensors and fixes the Random Forest's calibration under noise. It does not remove the lead lost under stuck sensors, and it lowers Mondrian FAULT coverage of the sequence models under MCAR.
  - On the unseen type (simulator drift), nothing changes: false alarms inside drift segments stay above zero.
  - The `sf_all` episode hidden by an all-sensor dropout is still missed by every augmentation-trained run.
- **Signal-relative noise (exploratory):**
  - Early warning breaks down through false alarms, not misses. Already at 10 % of the precursor rise, the recurrent models raise tens of false alarms per hour; at 25 %, every learned model does.
  - The growing lead and zero miss rate at high levels are artefacts of an alarm that is on all the time. The report explains how to read them.
  - Augmentation with the original noise grid (at most 30 % of the normal-operation std) does not help at these levels.
- **Report wording rule:** a level that is relative to the fault signal can be many normal-operation std on a sensor whose precursor rise is large (vibration). The report prints that range next to the grid.

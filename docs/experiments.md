# Experiments (Milestone B)

The research question (CLAUDE.md §2): how reliably can a data-driven digital twin warn of machine faults when its input is noisy or incomplete, when the machine runs under conditions different from training, or when a fault type was never seen, and does its reported confidence stay trustworthy?

Every number in `experiments/results/` is written by a script from tracked runs. This page defines the data, the labels and the metrics those scripts use. Design decisions are in [ADR 0019](adr/0019-ml-dataset-labels-splits.md) (dataset) and [ADR 0020](adr/0020-ml-models-training-tracking.md) (models, tracking).

## Reproduce E1

```text
uv run poe dataset                     # data/sim (200 runs x 1 h, default scenario, seed 42) + data/ml/<key>/
uv run poe train --model rf            # one model, seeds from config/models/eval.yaml (or --seeds 0,1)
uv run poe train-all                   # baselines + rf, xgb, lstm, gru, transformer x 3 seeds
uv run poe train-all --skip-existing   # resume after an interruption
uv run poe report-e1                   # experiments/results/E1_clean.md + .json + lead-time plot
uv run poe mlflow-ui                   # browse the runs (http://127.0.0.1:5000)
```

- **Configs:** everything lives in `config/models/`:
  - `dataset.yaml`: windows, horizons, cap, strides, split, subsample;
  - `eval.yaml`: seeds, subsets, early warning, latency, bootstrap, report;
  - one file per model.
- **Tracking:** MLflow runs go to `mlruns/`, a local file store that is gitignored.
- **One process per run:** every (model, seed) trains in its own worker process, so torch and XGBoost never share a process (a crash on macOS, [ADR 0020](adr/0020-ml-models-training-tracking.md)). If a worker fails, `train-all` stops; the finished runs are kept, and `--skip-existing` resumes.
- **Dataset identity:** the report and every run record the simulator manifest's sha256 (file bytes) and its `content_sha256` (stable across identical regenerations).

## Dataset

| Item | Value (config) |
|---|---|
| Source | `data/sim`: 200 runs × 3600 s, 1 Hz, 4 machines, default scenario, held-out `lubrication_loss` excluded |
| Split | by run, 70 / 15 / 15 (140 / 30 / 30 runs), fixed split seed |
| Window | 30 s per machine series, ending at `t` |
| Stride | train 5 s, val 5 s, test 1 s |
| Horizons | +10 s and +30 s |
| Time-to-fault cap | 600 s |

### Targets

- **Health at t + h:** NORMAL / WARNING / FAULT from the simulator state at `t + h`, with the CLAUDE.md §6 mapping (RUNNING, IDLE, STARVED, BLOCKED and MAINTENANCE map to NORMAL). Windows whose `t + 30` lies beyond the run are dropped.
- **Time to fault (TTF):** seconds from `t` to the next FAULT onset, capped at 600 s.
  - **Known:** a fault lies ahead, or at least 600 s of the run are left without one (the target is then exactly 600).
  - **Right-censored:** otherwise. Only "at least the time left" is known.
  - **Training:** trees train on known rows only; sequence models use a hinge on censored rows (no cost at or above the bound).
  - **MAE:** reported on known rows with TTF < 600 s.
- **Fault type (auxiliary):** the active hidden degradation at `t` (`none`, `bearing_wear`, `overheating`, `spindle_imbalance`, `belt_jam`). The simulator sets it when a gradual fault starts, before any WARNING.

### Inputs

- **Allowed:** only speed, temp, vib, load and current, plus the machine kind. The state, every label column, `run_time`, `seq` and the other simulator columns never reach a model. A test scrambles them and checks that every input stays bit-identical.
- **Engineered:** `vib_norm = vib / max(speed / speed_ref_kind, 0.1)`. `speed_ref_kind` is the training median of nonzero speed per kind. It separates speed-proportional bearing vibration from speed-squared imbalance.
- **Missing values:** forward fill within the series, the training mean for a leading gap, and a missing mask per sensor.
- **Normalisation:** mean and std per channel from training rows only.
- **Sequence models** see (30 × 14) windows: 6 value channels, 5 masks and the kind one-hot.
- **Trees** see 50 summary features: per value channel the mean, std, min, max, least-squares slope, last value and last − first; plus the missing fractions and the kind.

### Imbalance

- **Subsample:** training windows that are NORMAL now and at both horizons, with no active fault and no onset within 600 s, are kept with probability 0.25.
- **Compensation:** each kept one gets sample weight 4, so the weighted class distribution equals the original one.
- **Class weights** are then computed from these weighted counts: `min((W / (K · W_c)) ^ 0.5, 50)`. The loss weight is sample weight × class weight.

## Models

| Name | Input | Notes |
|---|---|---|
| persistence | state at `t` | baseline: the current health class continues |
| threshold | last observed vib / temp / current | baseline: nominal (unperturbed) per-kind limits from `simulator.yaml`; the same class for every horizon |
| rf | summaries | Random Forest per head; TTF regressor on known rows |
| xgb | summaries | XGBoost (`hist`) per head, early stopping on validation loss |
| lstm, gru | raw windows | 2 layers, hidden 64, multi-head |
| transformer | raw windows | 2 pre-norm layers, d_model 64, 4 heads, mean pooling, multi-head |

- **Sequence-model training:**
  - fixed budget: every run trains `max_epochs` (22) epochs, with no early stopping on patience (`patience: null`); the EMA of the validation macro F1 (operating windows, mean of both horizons) smooths the epoch-to-epoch noise;
  - the checkpoint kept is the epoch with the best EMA (`checkpoint_by: ema`);
  - cosine learning-rate decay over the budget, identical for every run (reduce-on-plateau and early stopping on patience remain available as options, used in the previous configuration);
  - per-epoch learning curves (train / validation loss per head, validation macro F1 per horizon, its EMA, the learning rate) are logged to MLflow as step metrics and plotted in the report.
  
  Settings: `early_stopping` and `lr_schedule` in `config/models/<model>.yaml`.
- **Seeds:** 3 per trained model; the baselines are deterministic and run once.

## Metrics

### Classification, per horizon and subset

- **Metrics:** accuracy, per-class precision / recall / F1, macro F1, fault recall, PR-AUC.
  - **Macro F1** (the main metric) averages F1 over the classes present in that subset's ground truth.
  - **PR-AUC** is the one-vs-rest average precision for WARNING and FAULT, and their mean.
- **Subsets:**
  - (a) **operating:** state at `t` is RUNNING or WARNING. This is the main result.
  - (b) **stopped:** state at `t` is FAULT or MAINTENANCE.
  - (c) **gradual / jam:** operating windows whose next fault onset within 600 s is a gradual fault / a belt jam. Jams have no precursors, so (c) shows what can and cannot be anticipated.
  - **all** windows.

### Early warning (gradual episodes, test runs, 1 s stride)

- **Episode:** the stretch where the simulator's `fault_type` is set (the degradation start), ending at the FAULT onset. It also records the first simulator WARNING inside it (the threshold onset on noise-free signals with per-run perturbed limits).
- **Sustained alarm:** the model predicts WARNING or FAULT for 3 consecutive seconds of windows. It is raised at the window that completes the run.
- **Lead time** = FAULT onset − the first sustained alarm with degradation start ≤ alarm < onset. An episode with no such alarm is **missed**.
- **Relative lead** = simulator WARNING onset − alarm time. Positive means the model warned earlier than the simulator's own threshold WARNING.
- **False alarms per hour** = rising edges of the sustained alarm in negative windows, divided by the hours of negative windows. Negative windows are operating, with no active fault and a NORMAL target at that horizon.
- **Per kind:** the report gives the number of episodes per machine kind behind every early-warning figure.

### Uncertainty

- **Over seeds:** mean ± std over the three seeds.
- **Over test data:** a 95 % percentile bootstrap CI, from 1000 seeded resamples of whole test runs (windows and episodes of a run stay together). It covers macro F1 and fault recall (operating), lead, relative lead, miss rate and false alarms per hour. The report shows the mean of the per-seed bounds.

### Paired comparisons

- **Method:** to compare two models, both are evaluated on the *same* bootstrap resamples of the test runs (paired bootstrap), so variation between runs that affects both cancels out. The statistic is the difference a − b.
- **Metrics:** mean lead time, lead relative to the threshold WARNING, operating macro F1 and false alarms per hour.
- **Comparisons:**
  - every model against the static threshold baseline;
  - the best sequence model against the best tree model at each horizon ("best" = the highest mean lead time; operating macro F1 if no model of a family detected an episode).
- **Seeds:** models with several seeds are compared seed by seed (seed i vs seed i, or each seed vs the deterministic baseline).
- **Reporting:** the report shows the mean difference, the mean CI bounds, whether that CI excludes zero, and in how many seeds the per-seed CI excludes zero.
- **Lead differences** compare the mean leads of the detected episodes of each side.

### Interpretation

- **Episode-level early warning is the primary result.** The research question is whether the twin warns before a fault. Per-window WARNING precision penalises a model that warns earlier than the simulator's WARNING label, because those windows count as false positives. So lead time, relative lead, miss rate and false alarms per hour come first, and per-window macro F1 is secondary.
- **Low FAULT recall at +10 s** means a model warns early as WARNING rather than timing the exact FAULT transition. The early-warning metrics count a WARNING or FAULT prediction as an alarm, so such a model still warns.
- **Stopped machines (subset b):** a stopped machine reads zero on every sensor, so sensor-only models cannot tell IDLE, FAULT and MAINTENANCE apart. This follows from the input restriction (no state input); persistence wins there by construction. The live system (Phase 11) should not predict for stopped machines.
- **Belt jams** have no precursors in the signals, and no model can anticipate them (subset c).

### Cost

- **Inference latency:** p50 / p95 at batch size 1, one CPU thread, window features included. 500 timed calls on random test windows after 50 warm-up calls.
- **Training time:** wall clock, measured on the machine that ran it. The report and PROGRESS name that machine.

## Phase 8: robustness, calibration and conformal prediction

The protocol, including the pre-registered primary outcomes, is in [ADR 0021](adr/0021-robustness-protocol.md). Calibration and conformal prediction are in [ADR 0022](adr/0022-calibration-conformal.md). All settings live in `config/models/robustness.yaml`.

### Reproduce

```text
uv run poe sim-sensor-faults                           # data/sim_sf_{all,seen,drift}: the 200 runs with sensor faults (seed 42)
uv run poe robustness --study e1models --dry-run       # plan and ETA
uv run poe robustness --study e1models --skip-existing # E2 / E3 / sensor faults on the E1 models (no retraining)
uv run poe dataset-e4                                  # E4 dataset: trained on low + medium load (ADR 0023)
uv run poe robustness --study e4 --skip-existing       # E4: retrain all models, test on high load
uv run poe dataset-aug                                 # augmentation dataset: half the training series perturbed
uv run poe robustness --study aug --skip-existing      # ablation: retrain all models on augmented data, same 19 conditions
uv run poe robustness --study e1models_snoise --skip-existing  # exploratory: signal-relative noise, E1 models
uv run poe robustness --study aug_snoise --skip-existing       # exploratory: signal-relative noise, augmentation-trained models
uv run poe robustness --study e1models_stdnoise --skip-existing  # exploratory: uniform noise in std units, E1 models
uv run poe robustness --study aug_stdnoise --skip-existing       # exploratory: uniform noise in std units, augmentation-trained models
uv run poe report-robustness                           # experiments/results/robustness.md + .json + plots
```

- **Order:** `aug_snoise` and `aug_stdnoise` reload the models trained by `aug`, so they need that study first. The other studies are independent.
- **Measured times on the reference laptop (CPU only):** e1models about 73 min, e4 about 2 h 26 min, aug about 4 h 00 min, e1models_snoise about 25 min, aug_snoise about 30 min, e1models_stdnoise about 26 min, aug_stdnoise about 30 min (`docs/PROGRESS.md` § Phase 8).
- **How to read the signal-relative noise grid:** under strong noise the alarm stays on, so the lead grows towards the episode length and false alarms (rising edges) can fall again. The report section explains this. Read those columns together with macro F1.

- **Resuming:** each (study, model, seed) runs in its own worker process and logs one MLflow run in the experiment `P8_robustness`. `--skip-existing` skips a pair only if a finished run exists with the same study config hash. That hash covers the model config, the robustness and evaluation settings, the dataset key and the reused E1 run.
- **Progress:** progress lines (plan, model ready, each condition, finished, with a re-estimated remaining time) go to `experiments/progress/<study>.jsonl`, which is gitignored.

### Test conditions

- **Synthetic perturbations** act on the raw sensor series before windowing, so forward fill and the missing masks behave as on live data. They apply to test data only, with the same seed for every model.

| Condition | Definition |
|---|---|
| `noise_p` | Gaussian noise, σ = p % of the sensor's normal-operation std (per machine kind: RUNNING rows of the clean training split without an active fault) |
| `mcar_p` | each value missing with probability p % |
| `dropout_p` | single-sensor gaps of 10–60 s covering p % of the sensor-time |
| `stuck_p` | a single sensor repeats its last value for 30–300 s, covering p % of the sensor-time |
| `sf_all` / `sf_seen` / `sf_drift` | the test runs regenerated by the simulator with sensor faults (stuck + drift + dropout / stuck + dropout / drift only), identical physics and labels |
| `e4_high` / `e4_in` | E4 only (ADR 0023): every high-load run (never trained or validated on) / the low + medium test runs |
| `snoise_p` | exploratory extension (ADR 0021 amendment): Gaussian noise, σ = p % (10, 25, 50, 100) of the sensor's typical rise before FAULT, from the clean training episodes |
| `stdnoise_m` | exploratory follow-up (ADR 0021 second amendment): Gaussian noise, σ = m × (1, 2, 4, 8) the sensor's normal-operation std, the same multiple for every sensor. This is the scale of `noise_p` (`noise_30` = 0.3 ×), so the report puts both grids on one axis and gives the level at which false alarms first rise |

- **Augmentation ablation (`aug`):** each training series is replaced with probability 0.5 by a perturbed copy (type uniform over noise / MCAR / dropout / stuck, level uniform over that type's test range, own seed). Validation and test stay as in E1, and the normaliser is the clean one. The report splits the conditions into types seen in augmentation (the synthetic grid and `sf_seen`) and unseen ones (`sf_drift`; `sf_all` mixes both).

### Added metrics

- **Calibration:** top-label ECE (15 bins) and FAULT-class ECE, before and after calibration, on operating windows; reliability diagrams.
- **Conformal:** for LAC, APS (randomised) and Mondrian LAC at α = 0.1:
  - marginal coverage and average set size;
  - coverage and set size per true class;
  - singleton and empty-set rates.
- **Time to fault:**
  - interval coverage and mean width on known targets below the cap;
  - on censored rows, the certain-miss rate (the interval ends below the known lower bound).
- **Sensor faults:** false alarms per hour inside simulator sensor-fault segments while the machine is healthy.
- **Degradation:** perturbed minus clean per model, as a paired run bootstrap. The number of seeds whose CI excludes zero is shown.
- **Alarms** keep the E1 rule, the uncalibrated argmax, so the robustness results compare directly with E1.

### Primary vs exploratory

The report labels every section. **Primary** (pre-registered, +30 s):
- the paired miss rate, lead relative to the threshold WARNING and false alarms per hour of each model vs the threshold baseline;
- ECE after calibration;
- FAULT-class coverage of LAC, APS and Mondrian LAC;

all under `noise_20`, `mcar_20`, `sf_all` and the E4 high-load test. **Everything else is exploratory.**

## Phase 9: unseen fault (E5), anomaly detection, explainability

The protocol and the pre-registered primary outcomes are in [ADR 0024](adr/0024-e5-unseen-fault-protocol.md). The detectors and the combined policy are in [ADR 0025](adr/0025-anomaly-detection.md), and the explanations and their sanity checks in [ADR 0026](adr/0026-explainability.md). All settings live in `config/models/e5.yaml` (plus `iforest.yaml` and `autoencoder.yaml`).

### Reproduce

```text
uv run poe sim-e5                                   # data/sim_e5: run indices 0-269, lubrication_loss only, seed 42 (~11 s)
uv run poe e5 --study e5_anomaly --skip-existing    # Isolation Forest + autoencoder x 3 seeds, trained on E1 normal windows
uv run poe e5 --study e5_e1models --skip-existing   # E1 models reused: clean, e5, sf_drift
uv run poe e5 --study e5_aug --skip-existing        # augmentation-trained models reused (needs Phase 8 `aug`)
uv run poe e5 --study e5_e4 --skip-existing         # E4 models + detectors trained on low + medium load (needs Phase 8 `e4`)
uv run poe e5 --combine --skip-existing             # combined policy (supervised OR detector) from the stored alarms
uv run poe e5 --explain --skip-existing             # explanation sanity checks (E1 XGBoost, RF, LSTM, GRU, Transformer)
uv run poe report-e5                                # experiments/results/e5_explain.md + .json + plots
```

- **Run selection:** the evaluation uses the 30 E1 test-split run indices plus 200–269 (100 runs), fixed in ADR 0024. Train and validation run indices are never used.
- **Reuse only:** supervised models are never trained by this runner. Each must reproduce its stored clean test predictions. `--combine` and the report need the source study runs first.
- **Resuming:** `--skip-existing` skips a run only if a finished run exists with the same config hash (model config, E5 / robustness / evaluation settings, dataset key, source run and the E5 data's content hash). Pipeline checks use `--smoke` (experiment `P9_smoke`).
- **Measured times:** in `docs/PROGRESS.md` § Phase 9.

### Primary vs exploratory

Primary (ADR 0024), at +30 s:
- E5 miss rate, E5 lead relative to the simulator WARNING, and clean false alarms per hour for every supervised model (E1 and augmentation-trained), each detector at the 99.9th percentile, and each combined policy, paired against the threshold rule (and each combined policy against its supervised model);
- Mondrian LAC mean set size on E5 vs clean;
- the explanation agreement rate for XGBoost (TreeSHAP) and the sequence models (Integrated Gradients).

Everything else is exploratory. Read every threshold-rule comparison on E5 with the interpretation note in ADR 0024: the rule checks the very signal `lubrication_loss` trips on.

## Results

- [E1: clean data](../experiments/results/E1_clean.md): generated by `uv run poe report-e1`.
- [Phase 8: robustness, calibration and conformal prediction](../experiments/results/robustness.md): generated by `uv run poe report-robustness`. It covers:
  - the pre-registered primary outcomes (`noise_20`, `mcar_20`, `sf_all`, `e4_high`);
  - E2 / E3 / sensor faults on the E1 models;
  - E4 condition shift;
  - the augmentation ablation (seen vs unseen perturbation types);
  - the signal-relative noise grid;
  - the uniform noise grid in std units, with the false-alarm onset per model;
  - perturbation strength relative to the fault signal;
  - the missed-episode case study and the drift note.

  Plots: `robustness_degradation_{noise,mcar,dropout,stuck}.png`, `robustness_reliability.png`, `robustness_coverage.png`, `robustness_e4.png`, `robustness_ablation.png`, `robustness_signal_noise.png`, `robustness_std_noise.png`.

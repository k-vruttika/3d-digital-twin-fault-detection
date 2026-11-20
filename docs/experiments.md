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
uv run poe e5 --ae-shares --skip-existing           # exploratory: autoencoder per-sensor error shares (re-fit, must reproduce alarms)
uv run poe e5 --ig-sensitivity --skip-existing      # exploratory: LSTM Integrated Gradients at 256 steps
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

Everything else is exploratory, including the two follow-ups added after the results were seen (ADR 0024 § Amendment; settings in `config/models/e5_followup.yaml`, never in `e5.yaml`): the autoencoder's per-sensor error shares and the IG step-count check. Read every threshold-rule comparison on E5 with the interpretation note in ADR 0024: the rule checks the very signal `lubrication_loss` trips on.

## Phase 10: validation on real public data

The protocol and the pre-registered primary outcomes are in [ADR 0027](adr/0027-real-data-protocol.md). How the code handles other sensor sets, kinds and time steps is in [ADR 0028](adr/0028-channel-schema.md). Settings:
- `config/datasets/<name>.yaml`: facts about each dataset (files, signals, failure list with its source);
- `config/models/real_<name>.yaml`: grid, window, labels and split, in rows of the dataset's own grid;
- `config/models/real.yaml`: studies, baselines, selection rule, `primary:`.

### Reproduce

```text
# files in data/external/metropt3/ and data/external/azure_pdm/ (never downloaded by the code)
uv run poe real-audit     # data audit of both datasets -> MLflow P10_real (about 10 s)
uv run poe real-data      # windowed datasets -> data/ml_real/<key>/ (about 10 s for all four)
uv run poe report-real    # experiments/results/real_data.md + .json + plots
```

If a file is missing, the loaders stop with an error that names the file, the dataset page and the folder to put it in.

### The datasets as they are (audit)

| | MetroPT-3 | Azure PdM |
|---|---|---|
| What | one metro train compressor (air production unit), real operational data | 100 machines; public benchmark, origin widely understood to be synthetic (Microsoft tutorial) |
| Signals | 7 analog (pressures, oil temperature, motor current) + 8 digital (valves, switches) | volt, rotate, pressure, vibration (units not documented) |
| Sampling (measured) | about 10 s (the PDF says 1 Hz; its row count is 10× the file's), 331 gaps over 60 s | exactly hourly, no gaps, no missing values |
| Labels | none; 4 air-leak failures from company reports (start and end time) | 761 component-failure rows = 719 events (machine, hour), no duration |

Every number is generated in [real_data.md](../experiments/results/real_data.md) § Data audit.

### Mapping to the common window format

**Grid and inputs**
- **Grid:**
  - MetroPT-3 is resampled to 1-minute means. A digital 0/1 signal becomes the fraction of the minute it was active.
  - Empty minutes stay as missing rows (forward fill + missing mask, as in Phase 7). A run of more than 10 empty minutes starts a new series, so no window spans a long gap.
  - Azure keeps its native hourly rows.
  - *Limitation:* 1-minute means smooth the compressor's load/unload cycles (a few minutes long), and short transients are lost.
- **Inputs:**
  - every signal of each dataset, normalised with training means and stds, plus missing masks and the kind one-hot;
  - no engineered `vib_norm` (no speed and vibration pair);
  - the kind is `APU` (MetroPT-3) or the machine model (Azure).
  - Azure error counts are inputs only in one exploratory XGBoost variant.
- **Variable number of sensors:** windows have the dataset's own channels: MetroPT-3 15 values + 15 masks + 1 kind; Azure 4 + 4 + 4. Tree summaries have 7 statistics per value channel. The model architectures are unchanged; only their input width follows the schema.

**Windows and labels**
- **Lengths, in the dataset's own time scale:**
  - MetroPT-3: window 60 min, sustain 3 min, lookbacks 2 h / **6 h** / 24 h.
  - Azure: window 24 h, horizons +12 h / **+24 h**, sustain 3 h, lookback 48 h, time-to-fault cap 7 days.
- **Labels (pseudo-states from the failure events):**
  - WARNING = within the warning band before a failure start (MetroPT-3 6 h, Azure 24 h);
  - FAULT = inside the reported failure interval (MetroPT-3), or the failure hour (Azure);
  - MAINTENANCE = 24 h after a MetroPT-3 failure end (excluded from negatives and detector training);
  - RUNNING otherwise.

  The fault type is set from 24 h (MetroPT-3) or 48 h (Azure) before the start. A supervised health target at +24 h (Azure) is therefore WARNING when a failure lies 24–48 h ahead.
  - *Limitation:* the band is a label choice, not a machine signal. Leads cannot exceed the lookback, and there is no machine WARNING to measure relative lead against.
- **Negatives** (false-alarm time): RUNNING windows with no fault type, i.e. more than the fault context before and more than the post-failure exclusion after any failure.

**Split, evaluation and limits**
- **Time-ordered split, no shuffling across time:**
  - MetroPT-3 trains on 2020-02-01 to 03-22, validates on 03-22 to 04-11 and tests on 04-11 to 09-02. All four failures are in the test period.
  - Azure trains on January–August 2015, validates on September–October and tests on November–December (493 / 112 / 114 events).
  - Rows are labelled on each machine's full timeline before the cut, so windows and horizons never cross a split boundary.
  - An event counts only if its whole lookback lies inside the split and at least 50 % of it has data. Otherwise it is "not evaluable", not a miss.
  - *Limitation (Azure):* train and test contain the same machines, so the results show forecasting for known machines, not new ones (exploratory machine-holdout variant: ADR 0027).
- **MetroPT-3 is a case study:**
  - Four failures are too few for supervised training, and with a time-ordered split none of them is in the training period. **No supervised model is trained on MetroPT-3.**
  - The anomaly detectors (Isolation Forest, autoencoder, trained on normal February–March operation) are the main method, with the threshold rule and the rolling z-score as baselines.
  - The questions: was each failure warned about, how early, and how many false alarms per day over months of normal operation? Each detection count comes with the number expected by chance at the measured false-alarm rate.
  - *Limitations:* four events allow no statistical claim; unreported failures may exist in the training period; the report times are approximate; the time zone is not stated.
- **Azure:** enough failures for the supervised models (RF, XGBoost, LSTM, GRU, Transformer) plus the detectors, with machine-level bootstrap CIs. The best supervised model is chosen on validation PR-AUC only.
- **Training cost:** Azure training windows are subsampled as in Phase 7. Far-normal windows are kept with p = 0.1 at weight 10, so the weighted class prior is restored. Training times are measured and reported.

### Primary vs exploratory

The primary outcomes are pre-registered in ADR 0027 § Pre-registration, before any model was trained on real data:
- **Azure (+24 h, lookback 48 h):** detection rate, mean lead and false alarms per day, for the best supervised model, the autoencoder (p99.9), the combined policy and the threshold / z-score / persistence baselines, paired against the threshold rule.
- **MetroPT-3 (lookback 6 h):** per failure, warned and lead; the warned count over evaluable failures; false alarms per day; the chance expectation. For the autoencoder, Isolation Forest, threshold rule and z-score.

Everything else is exploratory: calibration and conformal, robustness, other lookbacks and percentiles, the error-log variants, the machine holdout, false alarms by month, explanations and the comparison with the simulator.

### Studies and commands (Parts B and C)

```text
uv run poe real --study all --skip-existing   # metropt3, azure, azure_err, azure_holdout
uv run poe real --combine --skip-existing     # combined policies from the stored alarms
uv run poe real --explain --skip-existing     # exploratory explanations (Part C)
uv run poe replay-metropt                     # replay demo CSV (failure #4) -> data/replay/
uv run poe minimal-metropt                    # the twin on that CSV, minimal mode
uv run poe report-real
```

| Study | Dataset build | Methods (3 seeds; rules once) |
|---|---|---|
| `metropt3` | `real_metropt3.yaml` | Isolation Forest, autoencoder, threshold rule, rolling z-score |
| `azure` | `real_azure.yaml` | RF, XGBoost, LSTM, GRU, Transformer, Isolation Forest, autoencoder, persistence, threshold rule, z-score; plus the 7 robustness conditions for every supervised model, both detectors, threshold and z-score (exploratory) |
| `azure_err` (exploratory) | `real_azure_err.yaml` | XGBoost with error counts; the error-log rule (any code and each code) |
| `azure_holdout` (exploratory) | `real_azure_holdout.yaml` | the five supervised models and the autoencoder on 80 machines; tested on the 20 held-out machines |

Code: `twin_ml/experiments/real.py` (runner), `real_worker.py` (one process per study run; torch and XGBoost never share a process), `real_eval.py` (pure numpy event evaluation), `baselines/rules.py`, `real_explain.py`, `report_real_results.py`, `datasets/replay.py`. Every run is in MLflow `P10_real`; `--skip-existing` resumes by a hash of every setting that changes a result.

### Implementation details fixed in Part B, before the study runs

These complete definitions that ADR 0027 leaves open. They were written before any real-data study run. The only earlier contact with results was a pipeline check in the separate smoke experiment `P10_smoke` (MetroPT-3 rules and Isolation Forest, Azure rules / XGBoost / autoencoder / LSTM, seed 0). Nothing was changed after it.

- **Warned:** a window's sustained alarm counts at its *end* time. An alarm that is already on when the lookback starts and is still on inside it counts as a warning, with lead = the full lookback. The chance check accounts for such long-lasting alarms through the alarm-on fraction f.
- **Clipping:** if an earlier failure of the same machine ends inside the lookback, the lookback starts after it and the event is counted as `clipped`. Coverage is still measured against the full lookback, so a lookback clipped to less than half fails the evaluability rule.
- **Window coverage** (reported, not a rule): windows need a full window of history inside a series, and Azure windows also need the +24 h target row. So an Azure event in the last day of the test period, or a MetroPT-3 failure shortly after a gap, has windows for only part of its lookback, even when the data coverage is complete.
- **False alarms:** rising edges per series. An alarm carried over a gap or a split cut starts a new onset.
- **Rules:** window means use observed raw values only (missing and forward-filled values are skipped). Threshold bounds come from the training normal windows (Azure: the subsampled training windows). The z-score's trailing stretch never leaves the window's series. A sensor needs at least one window's worth (60 rows / 24 rows) of observed trailing rows, otherwise it gives no z.
- **Detectors (Azure):** fitted on the normal windows of the subsampled training split (far-normal windows kept with p = 0.1), with thresholds from all validation normal windows.
- **Combined policy:** per window, the OR of the two stored *sustained* alarms (not sustained a second time), seed i with seed i. Primary pair: the validation-chosen model OR the autoencoder at p99.9. Every supervised model × both detectors × every percentile is logged too (exploratory).
- **Robustness (exploratory):** the Phase 8 perturbation functions are applied to each test series' raw rows (NaN stays NaN), which are then re-normalised and forward filled. Each series draws from `SeedSequence(808, (condition, machine, machine#series))`, so every method sees the same perturbed data. Noise σ = level × the training normal-operation std per machine kind and sensor.
- **Calibration (exploratory):** the Phase 8 code (temperature scaling, or isotonic / Platt for trees), fitted on validation operating windows and evaluated on test operating windows. Reported: top-label ECE, FAULT-class ECE, positive-class ECE and Mondrian LAC coverage and set size per class at α = 0.1.

### Explanations and replay demo (Part C, exploratory)

- **Azure:** the validation-chosen model and XGBoost (TreeSHAP) are restored from their study runs, and must reproduce their stored test predictions.
  - Attributions are taken on the alarmed windows in the 48 h lookback of every warned, evaluable test event, at most 30 per event.
  - **Expected sensor per component, from training data only:** the sensor whose mean normalised value over the 24 h before that component's training failures deviates most from the same machines' normal rows. Two-component events are left out of this derivation.
  - Agreement = the event's top sensor is the expected one. Chance = 1/4 (1/2 for two-component events, which agree with either expected sensor).
- **MetroPT-3:** the autoencoder is re-fitted from each seed (its weights are not stored) and must reproduce its stored alarms bit for bit.
  - Per failure, it gives the per-sensor error shares on alarmed windows in the 6 h before the reported start and inside the reported interval, one window every 10 min, at most 30.
  - These are compared with the PDF-derived air-leak-plausible signals (10 of 15), and with the shares on 2,000 evenly spaced normal test windows as the reference. Error shares are not attributions in a strict sense.
- **Replay demo** (`config/datasets/metropt3_replay.yaml`):
  - the raw ~10 s samples from 6 h before to 2 h after failure #4, written in the simulator's CSV columns, so the unchanged file source replays them in minimal mode;
  - `temp` = Oil_temperature, `current` = Motor_current; speed, vibration and load stay empty, so every reading is SUSPECT (ADR 0007);
  - state = RUNNING when the motor current is ≥ 1 A, else IDLE (the measured current is bimodal: ~0.04 A and ≥ 3.5 A). The failure report is never injected as a state;
  - machine `CNV-01` of `line_single.yaml`; `MachineKind` has no compressor, so CONVEYOR is a placeholder;
  - timestamps are the naive MetroPT-3 clock written as UTC.
- **Simulator comparison** (report § Simulator vs real data): a generated table that puts the simulator's E1 clean and E5 results (read from the committed `E1_clean.json` and `e5_explain.json`) next to the real-data primary outcomes, for the Azure-chosen supervised model, both detectors and the threshold rule. Simulator false alarms per hour are scaled to per day; nothing else is converted, so it compares rankings, not values.

## Phase 11: drift detection report

`uv run poe drift-report` (needs the E1 dataset in `data/ml`, not the deployed bundles) writes [drift.md](../experiments/results/drift.md) + `.json`: the PSI alert levels per machine kind and channel, false drift on the clean validation and test runs over a margin grid, the live margin chosen on the validation runs by a stated rule, detection of synthetic input shifts, and ADWIN's dating of the change. See [ADR 0031](adr/0031-whatif-drift-demo.md). It is a detector evaluation on simulator data; the shifts are edits of recorded rows.

## Phase 11: demo scenarios

`tests/test_phase11_demo.py` runs four scenarios in real time against the real stack ([docs/demo_script.md](demo_script.md), [ADR 0031](adr/0031-whatif-drift-demo.md)) and, with `LINETWIN_DEMO_RESULTS_DIR=experiments/results/demo_scenarios`, writes what it measured as JSON there. These are single runs (event times), not evaluations.

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
- [Phase 9: unseen fault, anomaly detection, explanations](../experiments/results/e5_explain.md): generated by `uv run poe report-e5`. It covers the pre-registered E5 outcomes (early warning of supervised models, detectors and combined policies; conformal set size; explanation agreement), and, exploratory, the detector percentile trade-off, confidence of wrong predictions, the fault-type head, seed-to-seed variation, sensor drift, the E4 load shift, detector AUPRC, the deletion test and IG completeness, the autoencoder error shares and the IG step-count check. Conclusions: ADR 0024 § Outcome, ADR 0025 / 0026 § Results.

  Plots: `e5_explain_early_warning.png`, `e5_explain_tradeoff.png`, `e5_explain_set_size.png`, `e5_explain_agreement.png`, `e5_explain_deletion.png`.
- [Phase 10: validation on real public data](../experiments/results/real_data.md): generated by `uv run poe report-real` from MLflow `P10_real`. Conclusions: ADR 0027 § Outcome, ADR 0025 / 0026 § Phase 10 addenda.
  - **It covers:**
    - the data audit and mapping;
    - the pre-registered primary outcomes for MetroPT-3 (per-failure warnings, false alarms per day, chance check) and Azure PdM (detection rate, lead, false alarms per day with machine-bootstrap CIs, paired against the threshold rule);
    - exploratory: other lookbacks and percentiles, detection during the failure, false alarms by month, every model and combined policy, results by failure component, the error-log variants, the machine holdout, robustness, calibration and conformal sets, explanations, and the comparison with the simulator.
  - **Primary, MetroPT-3** (6 h, p99.9):
    - No method warned of the four air-leak failures beyond chance. Autoencoder 3 / 1 / 3 warned over seeds (chance 0.84–1.06), Isolation Forest 0, threshold rule 2 (chance 0.61), z-score 2 (chance 0.98).
    - Failure #1's "warning" is an anomaly already present when the data resumed after a gap.
    - The autoencoder and the threshold rule detect all four failures *during* the reported interval.
  - **Primary, Azure** (+24 h, 48 h):
    - The validation-chosen `lstm` detects 0.954 [0.913, 0.987] of 108 events, 41.2 h ahead, at 0.059 false alarms per day; the threshold rule detects 0.278.
    - The autoencoder detects 0.000, so the combined policy adds nothing.
    - Exploratory: an "any error in 24 h" rule from the error log detects 1.000, but 14 h later and with more false alarms than the telemetry models.

  Plots: `real_data_metropt3_availability.png`, `real_data_metropt3_timelines.png`, `real_data_metropt3_fa_month.png`, `real_data_azure_primary.png`, `real_data_azure_robustness.png`, `real_data_azure_reliability.png`, `real_data_explain_azure.png`, `real_data_explain_metropt3.png`.

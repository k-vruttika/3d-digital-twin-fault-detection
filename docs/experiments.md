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

## Results

- [E1: clean data](../experiments/results/E1_clean.md): generated by `uv run poe report-e1`.

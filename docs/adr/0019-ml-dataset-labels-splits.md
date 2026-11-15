# ADR 0019: ML dataset: windows, labels, censoring, splits, inputs and sample weights

- **Status:** Accepted
- **Date:** 2026-11-14
- **Phase:** 7 (ML dataset and models, Milestone B)

## Context

The research question asks how reliably a data-driven twin warns of faults, and whether its confidence holds up under noise, missing data, shifted operating conditions and unseen faults. Every later experiment (E1–E5, calibration, conformal prediction) reuses one dataset built from the simulator runs. Its labels, splits and inputs have to be leak-free and documented once. The simulator's Parquet files contain columns a model must never see (the state, the label columns, `run_time`), and its time-to-fault label caps at 1800 s while every run is 3600 s long.

## Decision

### Source data and identity

- **Source:** `uv run poe dataset` regenerates `data/sim`: 200 runs × 3600 s, the default scenario, seed 42, held-out `lubrication_loss` excluded. It then builds `data/ml/<dataset_key>/`.
- **Manifest hashes:** the simulator manifest now records a sha256 per data file and a `content_sha256`, the manifest without its volatile fields (`created_at`, `generation_time_s`, `workers`, `git_commit`, `cli_args`). `content_sha256` stays the same when identical data is regenerated.
- **Where the hashes go:** the build verifies every Parquet file against its hash. Both the raw `manifest_sha256` (the file's bytes) and `content_sha256` go into `meta.json`, every MLflow run's tags and the E1 report.
- **`dataset_key`:** `sha256(config fingerprint : content_sha256)[:16]`. Training finds the dataset that matches the current data and `config/models/dataset.yaml`, or stops with "run uv run poe dataset".

### Windows and targets

- **Windows:** 30 s per machine series, identified by the row of the last sample `t`. The stride comes from config: train 5 s, val 5 s, test 1 s (dense, for lead times). Windows never cross a series boundary.
- **Health targets:** `health@+h` = `to_health_class(state[t + h])` for h ∈ {10, 30} s, with the CLAUDE.md §6 mapping. A window whose `t + max(h)` lies past the end of the run is dropped, so every head sees the same windows.
- **Time to fault:** computed from `state` with the simulator's `time_to_fault`, without its cap, then capped at `ttf_cap_s = 600`. The target is `y = min(ttf, 600)`.
  - **Known:** a FAULT onset lies ahead in the run, or at least 600 s of the run are left without one (then the true value is ≥ 600, so `y = 600` is exact).
  - **Right-censored:** otherwise (no onset seen and less than 600 s left). Only `true ≥ y` is known.
  - **Handling:** tree regressors train on known rows only. The sequence models use Huber on known rows plus a hinge `max(0, y − ŷ)` on censored rows, all in units of the cap. MAE is reported on known rows with `y < cap` only.
  - **Why 600 s:** with the simulator's 1800 s cap, about half of every run would be ambiguous. 600 s still covers the ≥ 300 s minimum lead time.
- **Fault type (auxiliary target):** `fault_type[t]`, the active hidden degradation. The simulator sets it from the start of a gradual fault, before WARNING. Classes are the values seen in the data (`none`, the four faults, `overload` if present).
- **Episodes** (for early warning): a gradual fault episode is the contiguous `fault_type` stretch that ends in a FAULT onset. It records the degradation start, the first simulator WARNING row (threshold onset) and the FAULT onset.
- **Episode type** per row: gradual, jam, other or none. It is the type of the current fault (rows in FAULT) or of the next fault onset within the cap. Evaluation subset (c) uses it on operating windows only, so "gradual vs belt jam" measures warning before the fault.

### Inputs: one door

- **Allowed:** `twin_ml.features.inputs.ALLOWED_INPUT_COLUMNS` = speed, temp, vib, load, current, plus the machine kind (from the line config). The Parquet loader passes only these columns to `select_inputs`. Label columns are read separately and never reach the feature code.
- **Engineered:** `vib_norm = vib / max(speed / speed_ref_kind, 0.1)`, where `speed_ref_kind` is the training-set median of nonzero speed per kind. Bearing vibration scales with speed and imbalance with speed², so this separates them across CNC speed changes.
- **Missing values:** forward fill within the series. Leading gaps take the training mean. A per-sensor missing mask becomes extra channels (sequence models) or missing fractions (trees). The default scenario has no missing values; E2 will.
- **Normalisation:** mean and std per value channel over the training rows only, stored in `meta.json`.
- **Model views:**
  - sequence models: (30 × 14) windows of 6 value channels, 5 masks and the kind one-hot;
  - trees: 50 summary features (mean, std, min, max, slope, last, delta per value channel; missing fractions; kind).
- **Test:** scrambling every forbidden column (state, run_time, seq, quality, fault_type, time_to_fault_s, ttf_censored, root_cause_machine, load_range, sensor_fault, buffers) leaves every input array bit-identical.

### Splits

- **By run:** 70 / 15 / 15, i.e. 140 / 30 / 30 runs. The run ids are sorted, then permuted with a fixed split seed (2026), independent of the model seeds. No run contributes to two splits (tested).

### Training subsample and sample weights (no double correction)

- **Subsample:** training windows that are NORMAL now and at every horizon, with no active fault and no FAULT onset within the cap ("far-normal"), are kept with probability `normal_keep_frac = 0.25`. Every other window is kept. Val and test are never subsampled.
- **Weights:** a kept far-normal window gets sample weight `1 / normal_keep_frac`. The weighted class distribution of the training set then equals the unsubsampled one in expectation. Class weights are computed from these weighted counts and applied on top: `weight = sample_weight × class_weight[y]`, with `class_weight = min((W / (K · W_c)) ** power, max)`.
  - Without the compensation, subsampling and class weights would both push the prior towards the rare classes. That would inflate the fault prior and raise test-time false alarms.
  - Tests: the weighted training distribution matches the unsubsampled build, and class weights from the subsample match those of the full set.

## Consequences

- **Inference without the simulator:** `twin_ml.features`, `data.dataset`, `models` and `evaluate` import neither `twin_sim` nor torch (except `models.nn`), so Phase 11 can run inference in the backend. Only `data.build` / `data.load` and the threshold baseline import `twin_sim` (tested).
- **Cost:** the test split at stride 1 s has 424,920 windows (`meta.json` of the current build). Windows are gathered on demand from row arrays. Stored as (30 × 14) float32 windows they would take about 0.7 GB; the whole built dataset is 192 MB on disk.
- **Changing labels:** any change to the labels, cap, strides or features changes the fingerprint and therefore the `dataset_key`. Old MLflow runs stay tied to their own dataset.

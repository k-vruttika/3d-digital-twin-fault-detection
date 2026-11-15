# ADR 0020: ML models, training, evaluation and tracking (CPU PyTorch, MLflow file store)

- **Status:** Accepted
- **Date:** 2026-11-14
- **Phase:** 7 (ML dataset and models, Milestone B)

## Context

E1 compares baselines and five model families on clean simulator data. Three constraints apply:

- it has to run on a laptop CPU (the development machine has no NVIDIA GPU) and on CI on three OSes;
- every result must come from code and be reproducible from seeds;
- the backend and minimal mode must never load the ML stack.

## Decision

### Dependencies

- **PyTorch 2.15.0, CPU build.** On Linux and Windows it comes from PyTorch's CPU index (`[[tool.uv.index]] pytorch-cpu`, `explicit = true`, used only for `torch`); on macOS it comes from PyPI, which is CPU/MPS only. No CUDA wheels anywhere.
- **scikit-learn 1.9.1.**
- **XGBoost 3.2.0.** It is the last release that supports Python 3.11; 3.3 and later need 3.12. Linux and Windows use `xgboost-cpu`, which has no CUDA/NCCL dependency; macOS uses `xgboost`, which is CPU-only there. Both provide the `xgboost` module.
- **`mlflow-skinny` 3.17.2.** It is the MLflow tracking client with the file store, but without the UI server, pandas or sqlalchemy. Full `mlflow` 3.17 pins `pyarrow < 26`, which conflicts with the simulator's `pyarrow ≥ 26.0.1`; 3.18 lifts the pin but was under two weeks old. The UI runs isolated: `uv run poe mlflow-ui` (`uv tool run --from mlflow==3.17.2`).
- **Versions:** each is the newest release that was at least two weeks old.

### Tracking

- **Store:** a local MLflow file store at `<repo>/mlruns` (gitignored), experiment `E1_clean`. MLflow 3.x keeps the file store in maintenance mode and refuses it unless `MLFLOW_ALLOW_FILE_STORE=true`. `twin_ml.train.tracking` sets that, since CLAUDE.md §4 asks for a local file store and the version is pinned. If a later MLflow drops the file store, `mlflow migrate-filestore` moves the runs to SQLite.
- **Per run:**
  - tags: phase, model, family, seed, `dataset_key`, both manifest hashes;
  - params: the flattened model and dataset configs, the seed;
  - metrics: every metric × subset × horizon, bootstrap CI bounds, train time, latency;
  - artifacts: `metrics.json` (the full nested result, including per-episode lead times), `splits.json`, `test_predictions.npz`, and the model (XGBoost JSON or the torch state dict).
- **Random Forests are not saved:** hundreds of MB per seed; they can be retrained from the seed.

### Models

- **Baselines:**
  - persistence: the current health class, read from `state` by design and never a model input;
  - static threshold: the last observed vib, temp and current vs the nominal per-kind limits from `config/simulator.yaml`.
- **Random Forest and XGBoost** on the 50 window summary features:
  - per model, one classifier per horizon, a time-to-fault regressor (known rows only) and a fault-type classifier;
  - XGBoost uses `hist` and early-stops each head on its validation loss (logloss, or absolute error for time to fault);
  - the Random Forest has no early stopping.
- **LSTM, GRU and Transformer** (`twin_ml.models.nn`):
  - one shared encoder (2 layers, width 64; the Transformer has 4 heads, feed-forward width 128, learned positions, pre-norm and mean pooling);
  - four heads: health at +10 s, health at +30 s, time to fault (sigmoid × cap, censoring-aware loss) and fault type (auxiliary, loss weight 0.3);
  - AdamW, gradient clipping;
  - training budget and checkpoint rule on the validation macro F1 of operating windows, averaged over both horizons: see "Amendment 2026-11-15" below (initially patience 3, at most 15 epochs);
  - focal loss is available (`focal_gamma`) and off by default.
- **Weights:** sample weight × class weight as in ADR 0019 (power 0.5, cap 50) for every classifier.
- **Seeds:** three model seeds (0, 1, 2) per trained model; the data split is fixed. Torch runs with `use_deterministic_algorithms(True)` and a seeded batch order; same seed, same predictions (tested).

### Evaluation (E1)

- **Classification:** accuracy, per-class precision / recall / F1, macro F1 (the main metric; the mean over the classes present in the subset's ground truth), fault recall, PR-AUC (one-vs-rest average precision for WARNING and FAULT).
- **Regression and auxiliary:** time-to-fault MAE on known rows below the cap; fault-type macro F1.
- **Subsets:** (a) operating now (RUNNING or WARNING), the main result; (b) FAULT or MAINTENANCE now; (c) operating windows before a gradual fault vs before a belt jam (next onset within the ttf cap); all windows.
- **Early warning** on gradual episodes of the test runs (stride 1 s):
  - a sustained alarm is ≥ 3 s of WARNING/FAULT predictions;
  - lead time = onset − first sustained alarm after the degradation started;
  - relative lead = simulator WARNING onset − alarm;
  - false alarms per hour of operating, fault-free, NORMAL-target time.
- **Uncertainty:** 1000 seeded bootstrap resamples of whole test runs give 95 % percentile CIs for macro F1, fault recall, lead, relative lead, miss rate and false alarms per hour. The report shows mean ± std over seeds and the mean of the per-seed CIs, plus the number of episodes per machine kind.
- **Latency:** p50 / p95 at batch size 1 on one CPU thread, features included, 500 calls after 50 warm-up calls.
- **Report:** `python -m twin_ml.experiments.report_e1` reads MLflow only and writes `experiments/results/E1_clean.md`, `.json` and a lead-time plot.

## Consequences

- **Isolation from the backend:** the backend and minimal mode stay free of torch, sklearn, xgboost, mlflow and `twin_ml`. Every import guard (`tests/test_phase*_minimal.py`, `/health?modules=true`) now lists `mlflow` and `twin_ml`.
- **CI cost:** CI installs the CPU wheels on all three OSes (the uv cache keeps repeat runs short). The tiny end-to-end training tests run everywhere.
- **No GPU path:** CUDA wheels would need a separate index and an approval.

## Amendment 2026-11-15: fixed training budget for the sequence models

- **Decision:** LSTM, GRU and Transformer train for a fixed 22 epochs with a cosine learning-rate decay (`lr` to `min_lr` 1e-5 over the 22 epochs). There is no early stopping on patience. The checkpoint kept is the epoch with the best exponential moving average (alpha 0.4) of the validation macro F1.
- **Why:** the validation learning curves of the previous configuration (EMA early stopping, reduce-on-plateau, cap 16) showed two problems. Several runs were still improving at the 16-epoch cap. Others stopped at 10 to 11 epochs on patience before the first learning-rate reduction, so seeds got different schedules and the seed-to-seed spread mixed model variance with schedule differences. A fixed budget and a fixed schedule give every run the same annealing phase. Reduce-on-plateau cannot do that, because its drops depend on each run's validation loss.
- **Evidence used:** validation curves only (per-epoch validation loss, validation macro F1 and its EMA, learning rate). The budget of 22 epochs and the schedule were not chosen from test-set results. The test-set metrics of the previous runs had been seen (the previous E1 report), but they were not used for this choice.
- **Consequences:** training takes longer for the runs that used to stop early (22 epochs each). Old runs keep their earlier config hash; `train-all --skip-existing` reruns every sequence model. The code keeps `patience`, `min_epochs`, `checkpoint_by: raw` and the plateau schedule as options.

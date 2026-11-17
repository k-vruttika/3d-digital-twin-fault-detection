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

## Amendment 2026-11-15: one worker process per training run (macOS OpenMP)

- **Problem:** CI on `macos-latest` crashed with a segmentation fault (exit code 139) in `ml/tests/test_ml_e2e_train.py` once `brew install libomp` let XGBoost load. The crash report showed torch and XGBoost loaded in the same process. Both use an OpenMP runtime, and on macOS both are LLVM `libomp.dylib`: the PyPI torch wheel ships its own copy in `torch/lib/`, and XGBoost loads Homebrew's through `@rpath/libomp.dylib`. Two copies of the same runtime in one process is a known crash cause. The training code loads torch, scikit-learn and XGBoost lazily, but the e2e tests and `train-all` trained every model in one process, so they all piled up there.
- **How far the cause is confirmed:** the crash report shows both libraries in the crashed process. On Windows (the development machine) the same process holds two runtimes too: torch's Intel `libiomp5md.dll` and XGBoost's MSVC `vcomp140.dll`. They have different symbol names and live side by side, which fits the crash showing up only on macOS. The macOS paths themselves have not been checked on a Mac: there is none here. Every worker now logs `train.openmp` with the paths of the OpenMP runtimes it loaded (through `threadpoolctl`, a scikit-learn dependency), so the macOS CI log shows which `libomp.dylib` each model uses. The workers run with faulthandler on, so if a crash ever happens again, the log shows the crashing frame.
- **Decision:** each (model, seed) trains in its own Python process. `twin_ml.train.isolated.run_isolated` writes the arguments of `run_one` to a JSON request, starts `python -X faulthandler -m twin_ml.train.isolated <request> <result>`, and reads back the `RunResult`. `train-all`, `poe train` and the e2e tests all use it. The worker calls `run_one` unchanged, so metrics, MLflow runs, tags, artifacts and `--skip-existing` resumability are the same as before. A worker that exits without a result raises `TrainingProcessError`, which names the exit code or signal. `train-all` then stops; the runs that finished are kept, so `--skip-existing` resumes.
- **Rejected:** `KMP_DUPLICATE_LIB_OK=TRUE`. It only turns off Intel's check for a second runtime, and both runtimes still compete for the same threads. Using Homebrew's libomp for torch as well would have meant patching wheels per OS.
- **Tests:** `test_training_worker_loads_only_its_own_library` trains rf, xgb and gru through the worker entry point. It checks that rf loads neither torch nor XGBoost, that xgb never loads torch, and that gru never loads XGBoost. `require_xgboost` checks the XGBoost import in a subprocess, because the pytest process loads torch for the loss and determinism tests.
- **Consequences:** each run pays about 2 to 4 s of interpreter, MLflow and library start-up. That is small next to training on the full dataset, but it makes the e2e test module slower. Workers inherit stdout and stderr, so their logs show up live in the CLI and in pytest's captured output.

## Amendment 2026-11-17: no torch next to XGBoost in tests, explanations or live inference

- **Problem:** CI on `macos-latest` crashed again with exit code 139, in `ml/tests/test_ml_explain.py`, inside XGBoost (`data.py`, `_meta_from_numpy`). It was the same clash as above. A module-scoped fixture fitted an XGBoost model in the pytest process, and earlier tests had already loaded torch there. The first amendment had isolated training, but tests that use a model directly were not covered, and nothing would have caught the next one before macOS CI.
- **Decision, tests:**
  - The XGBoost explanation tests (TreeSHAP additivity, `explain()` for XGBoost) moved to `ml/tests/xgb_isolated_explain.py`. pytest does not collect that file on its own (no `test_` prefix).
  - `test_ml_explain.py::test_xgboost_explanations_in_own_process` runs it in a fresh pytest process (`twin_ml.testing.run_pytest_isolated`, with faulthandler). It hands over the quick dataset by path, so the child does not rebuild it.
  - The child asserts that it never loads torch. The Random Forest and GRU explanation tests stay in-process.
- **Decision, guard:**
  - Mechanism: the root `conftest.py` checks `sys.modules` after collection and after every test's teardown. If torch and XGBoost are both imported in the pytest process, it fails the session, either as an error at the teardown of the first test after which both are loaded, or as a usage error after collection. The message points here.
  - The guard runs on every OS, so Linux and Windows CI catch a new clash before macOS crashes on it.
  - `tests/test_openmp_guard.py` checks the guard in child pytest processes with stand-in modules, so the real libraries are never loaded together.
  - On Windows (2026-11-17) the whole suite passes with the guard in place. Before the fix, the guard stopped at `test_treeshap_sums_to_margin`.
- **Decision, Phase 9 code:**
  - Every Phase 9 worker (`e5_worker`: study runs, explanations, autoencoder shares) handles one (model, seed) per process.
  - It records the ML libraries and OpenMP runtimes it loaded in `metrics.json` (`process`), and raises if torch and XGBoost are both loaded. The E5 e2e test asserts that XGBoost runs never load torch and that the sequence-model and autoencoder runs never load XGBoost.
  - `twin_ml.explain` (`explain()`, `attribute()`, `probabilities()`) raises `OpenMPClashError` before using XGBoost if torch is loaded, and before using torch if XGBoost is loaded.
  - The combined policy and the reports are pure numpy and MLflow.
- **Consequence for Phase 11 (live inference):**
  - Tree models (XGBoost) and sequence models (LSTM, GRU, Transformer; also the torch autoencoder) must run in **separate processes**: for example one inference worker process per model family, behind the backend's executor boundary.
  - One process that loads both would crash on macOS, and `explain()` refuses to do it.
  - scikit-learn (Random Forest, Isolation Forest) next to torch has run in the same pytest process on all three CI OSes without a crash. It is not covered by the guard; if it ever crashes, extend `OPENMP_CLASH` in the root conftest.

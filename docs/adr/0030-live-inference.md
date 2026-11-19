# ADR 0030: Live inference in the twin

- **Status:** Accepted (Part A; Part C adds what-if and drift)
- **Date:** 2026-11-19
- **Phase:** 11 (live intelligence in the twin)

## Context

The twin has to publish a `Prediction` (CLAUDE.md §6) for every running machine, every second, from the deployed bundles (ADR 0029). Constraints carried over from earlier phases:

- **Minimal mode never imports torch, scikit-learn, XGBoost or MLflow.** ML inference is optional and runs outside the API process.
- **torch and XGBoost never share a process** (ADR 0020).
- **The live alarm rule must be the evaluated one:** uncalibrated argmax at +30 s, sustained 3 s (ADR 0022). Calibrated probabilities and Mondrian sets are confidence only.
- **No prediction for stopped machines.** Sensor-only models cannot tell IDLE, FAULT, MAINTENANCE, STALE and OFFLINE apart.
- **Autoencoder alarms** are their own category ("unusual behaviour, cause unknown") and never highlight a part (ADR 0025 / 0026). Part highlights come only from `explain()` on supervised models.
- **Latency budget:** under 100 ms per machine per tick on CPU, measured and reported.

## Decision

### Processes and transport

- **The backend never imports `twin_ml`.** It starts `python -m twin_ml.serve worker` child processes with `subprocess.Popen` (no asyncio subprocess on the Windows selector loop, ADR 0008).
  - The transport is JSON lines on stdin / stdout (`twin_common/jsonl_process.py`). A reader thread collects replies; the event loop waits on a worker thread.
  - Worker stderr goes to the backend's stderr.
- **Worker plans** (`InferenceConfig.worker_plans`):
  - a sequence model and the autoencoder share one torch process (roles `supervised` + `monitor`);
  - an XGBoost model gets its own process next to a torch `monitor` process.
  - Each worker reports the libraries it loaded at hello and refuses to run with both. The pool refuses a worker that reports both, and a test asserts it for both plans.
- **Lifecycle:**
  - The pool sends every batch of readings to every worker and merges their answers per reading. Both workers keep their own buffers and compute the same gate; a disagreement is an error.
  - A crashed or silent worker is restarted with backoff (`restart_backoff_s`); meanwhile the gate is `unavailable` and inference alarms clear.
- **Switches:** `serve --inference` (also `poe dev --inference`, `--model`, `--model-version`) turns it on. `config/inference.yaml` has `enabled: false`, so minimal mode builds no inference service unless asked.
  - With it on, the API process still loads no ML module. Tested through `/health?modules=true` on a running minimal-mode server with workers.

### Windows: same inputs as training

- **Buffers.** Each worker keeps a 1 Hz buffer per machine (`serve/buffers.py`).
  - Rows are keyed by `seq`. A gap of up to `max_gap_rows` missing readings is filled with all-missing rows (masked). A longer gap, or a sequence restart, starts the warm-up again.
  - The buffer carries the last observed value of every sensor before its first row. Transforming `[carry] + rows` with `Normalizer.transform` is therefore bit-identical to the dataset build's series-wide forward fill. A test checks this row by row, with leading, scattered and buffer-length gaps.
- **Model inputs** come from the training functions only: the normaliser and `SplitData.sequence` / `summary` on a `SplitData` packed from live windows, passed to the model's own `predict` and the autoencoder's `score`.
- **Parity.** On the quick test split, live outputs equal offline `predict` for the same windows:
  - GRU in this process: atol 1e-5;
  - XGBoost in its own process: atol 1e-6;
  - autoencoder score: relative 1e-5.
  - Through the backend, the bus and JSON, an integration test compares every published prediction with an independent worker on the same readings. Float32 sums depend slightly on batch composition, hence the tolerances.

### Gate (decision with the user)

- **Rule:** a window is predicted only when all of its 30 rows are RUNNING or WARNING (the evaluated operating states).
- **Shown otherwise:**
  - `stopped` (IDLE, FAULT, MAINTENANCE, STALE, OFFLINE, or nothing received for `prediction_stale_s`): "no prediction (machine stopped)";
  - `waiting` (STARVED, BLOCKED): "no prediction (machine waiting)";
  - `warming_up` (operating, fewer than 30 operating rows in a row).
- **Stricter than the evaluation.** The evaluated "operating" subset of Phases 7–10 checked only the window's last row (`evaluator._state_mask` at `ends`). The gate predicts a subset of those windows and never mixes stopped and running rows.
- **Its effect on the evaluated test split** (exploratory, generated in `experiments/results/live_inference.md`, stored clean test predictions of the deployed seeds):
  - No episode changes detection, and false alarms stay 0.00 per hour.
  - The mean lead before the simulator's WARNING drops, because the sustain count restarts after every short stop: GRU at +30 s from 61.8 to 53.4 s, XGBoost from 60.5 to 55.1 s.
  - Of 342,391 evaluated operating windows, 2,996 are not predicted live.

### Alarm rule, alerts

- **The rule** (`backend/twin_backend/inference/alarms.py`):
  - the uncalibrated +30 s argmax is WARNING or FAULT for `k = ceil(sustain_s / 1 s)` consecutive gated windows;
  - `sustain_s` comes from the bundle, i.e. from `eval.yaml`.
  - A test compares it with `twin_ml.evaluate.early_warning.sustained` on random sequences and probabilities.
  - Counters restart when the gate closes.
- **Alert kinds** (`engine.yaml`, `source: inference`, `raise_after_s: 0`, so the 3 s sustain is the only debounce):
  - `predicted_fault` (CRITICAL): the alarm, with argmax FAULT sustained;
  - `predicted_warning` (WARNING): the alarm otherwise. The union of the two is exactly the evaluated alarm.
  - `unusual_behaviour` (WARNING): the autoencoder flag (score above its per-kind p99.9 threshold) sustained the same way, while the supervised alarm is off. This is the Phase 9 combined policy. The message says "cause unknown" and names no part.
  - `model_drift`: reserved for Part C.
- **Lifecycle.** The service sends only changed conditions to `TwinEngine.update_conditions`. That runs them through the engine's ISA-18.2 manager (ADR 0011): raise, deduplicate, clear, ack, re-raise, restore.
  - `AlertKindConfig.source` makes these alerts `source: inference`, and `AlertManager.load` restores them.
  - Messages carry the predicted class, the horizon, the time to fault with its interval, the calibrated probability, and the top signal when an explanation exists.

### Prediction events

- **One `Prediction` per horizon** (+10 s, +30 s) per gated reading, published on `Topic.PREDICTIONS`. The existing writer stores it and the hub sends it, so replay shows predictions as they were at time t.
- **Contents:**
  - `probs` are the calibrated probabilities (`calibrated: true`);
  - `conformalSet` is the Mondrian LAC set;
  - `ttfS` / `ttfInterval` come from the conformal time-to-fault interval;
  - `anomalyScore` / `anomalyFlag` / `anomalyThreshold` come from the autoencoder;
  - `alarm` is the evaluated supervised alarm, on the primary horizon;
  - `reliability` is DEGRADED while the machine's data quality is DEGRADED (drift is added in Part C);
  - `modelVersion` is the bundle labels.
- **New optional contract fields** (stored predictions without them stay valid): `alarm`, `anomalyFlag`, `anomalyThreshold`, `attributionsTs`.
- **WS snapshot.** `WsSnapshotData.predictions` was added, so a client connecting mid-run gets the newest predictions.
- **REST.** `GET /predict` describes the service, the bundles and model cards, and per machine the gate and newest predictions. `GET /predict/history` reads stored predictions.

### Explanations beside the prediction path

- **What is explained.** The newest window of a machine whose uncalibrated +30 s argmax is WARNING or FAULT, by the supervised model only: `SupervisedLive.explain`, the batched form of `explain()`, Integrated Gradients with 64 steps (the evaluated setting) or TreeSHAP.
- **Why it runs separately.** Measured first inside the prediction path, IG on the GRU made ticks with an explanation take 469 ms p50 on one CPU thread, against the 100 ms budget.
  - Lowering the step count would change the evaluated explanation; Phase 9 checked 64 and 256 steps, nothing fewer.
  - So a background thread in the worker explains while predictions continue. It keeps only the newest request per machine. Results come back with a later step reply, tagged with their window.
- **How they are attached.** The service attaches the newest explanation to the machine's +30 s predictions while the argmax stays WARNING / FAULT, with `attributionsTs` = the explained window. It drops it when the argmax is NORMAL, the gate closes, or it is older than `prediction_stale_s`.
- **Measured.** GRU explanations take 269 ms p50 to compute and arrive 0.56 s after the request (p95 0.87 s), before the next reading at 1 reading per second. XGBoost TreeSHAP takes 4 ms. Both are reported separately in `live_inference.md`.
- **Tests:** the asynchronous attributions equal the synchronous ones, and the batched ones equal `explain()`.

### Latency (measured, `uv run poe live-latency`, this laptop, CPU)

One test run replayed through the real worker processes, four machines, one reading per machine per tick, IPC included:

| Model | Per machine per tick p50 / p95 / max | Round trip per tick (4 machines) p50 / p95 |
|---|---|---|
| GRU (+ autoencoder) | 1.20 / 1.91 / 9.3 ms | 4.65 / 5.93 ms |
| XGBoost + monitor | 1.53 / 3.13 / 22.2 ms | 5.98 / 8.75 ms |

Worker start (bundle download + import + model load) takes 3.5 s.

## Consequences

- The twin warns with exactly the evaluated rule and shows calibrated, conformal confidence next to it. The UI can tell "stopped", "waiting" and "warming up" apart from "no risk".
- **The stricter gate costs lead time:** on average 8 s for the GRU and 5 s for XGBoost on the evaluated test split, with no missed episode. The demo docs say so.
- **Attributions can be up to about a second older than the prediction they are shown with** (`attributionsTs`). A highlight therefore appears shortly after an alarm is raised, not with it.
- **Worker restarts lose the buffers,** so machines warm up again for 30 s.
- **Not measured here:** Linux and macOS latency. The transport and process split are the same; CI runs the integration tests on all three OSes.

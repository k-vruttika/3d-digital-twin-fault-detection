# ADR 0029: Deployed models, deploy bundles and the model registry

- **Status:** Accepted
- **Date:** 2026-11-19
- **Phase:** 11 (live intelligence in the twin), Part A

## Context

Phases 7–10 trained and evaluated models offline, but the twin never ran one. Phase 11 runs one of them live. That raises three questions:

1. Which trained model is deployed, and why?
2. How do the model, its calibration and conformal maps, and the autoencoder get from the experiment runs to the live process?
3. How does the UI say exactly which model it shows?

Facts that constrain the answer:

- **Calibration and conformal maps were never stored.** Phase 8 fitted them per run (ADR 0022) and logged only their numbers (temperatures, conformal quantiles) in `metrics.json`. The scikit-learn Platt / isotonic objects were never stored.
- **The autoencoder weights were never stored** (Phase 9). The Part C follow-up showed that a refit from the seed reproduces the stored alarms bit for bit.
- **XGBoost needs its training data to load.** `TreeModel.load` recomputed each head's classes from `train.npz`.
- **Process isolation.** torch and XGBoost must never share a process (ADR 0020).

## Decision

### Deployed models

The decision with the user was: the **E1 GRU** is the default and the **E1 XGBoost** is the fast option. `config/inference.yaml` selects one (`model: gru | xgb | gru_e4`). The evidence comes from the committed, generated reports:

- **E1, clean** (`E1_clean.md`, +30 s):
  - GRU has the earliest warning relative to the simulator's own WARNING: 61 ± 1 s, against XGBoost 58, LSTM 53, Transformer 56 and RF 52. Every learned model misses 0 of 26 episodes, with about 0 false alarms per hour.
  - Latency at batch 1: XGBoost 1.41 ms p50, GRU 1.23 ms.
- **E4 load shift** (`robustness.md` § e4_high; models trained on low and medium load):
  - The GRU loses 9 s of relative lead (not significant); XGBoost loses 43 s (3/3 seeds).
  - Mondrian FAULT coverage drops for every model (GRU 0.815, XGBoost 0.777).
- **E5 unseen fault:** no supervised model is reliable there; this is why the autoencoder's "unusual behaviour" alarm runs next to the supervised model (ADR 0025).

**`gru_e4`, trained on low and medium load only, stays selectable but is not built by default.** The E1 models were trained on all load bands. For them, the high band is in-distribution, so a drift demonstration has to go above the trained range (Part C).

**Seed.** The deployed seed is the one with the highest validation macro F1 (operating windows, +30 s), computed by the bundle builder for every candidate seed. Test data is never used for the choice. Measured values:

| Model | Seed 0 | Seed 1 | Seed 2 | Chosen |
|---|---|---|---|---|
| GRU | 0.835 | 0.825 | 0.813 | seed 0 |
| XGBoost | 0.819 | 0.813 | 0.816 | seed 0 |

### Deploy bundles (`twin_ml/serve/bundle.py`, `config/models/deploy.yaml`)

`uv run poe deploy-models` builds every bundle in its own process, so torch and XGBoost never meet.

- **Supervised bundle** (`gru`, `xgb`, `gru_e4`):
  - **Model:** the trained model of the chosen seed, reused from its training run, never retrained. It must reproduce its stored clean test predictions (`check_predictions`, atol 1e-5, identical labels).
  - **Calibration and Mondrian conformal maps:** refitted on the clean validation split with the Phase 8 code (`robust_eval.fit_calibration`).
    - A strict build requires the Phase 8 study run that evaluated that exact source run (matched on `source_run_id`), and checks that the refitted temperature, the Mondrian per-class quantiles and the time-to-fault quantile equal the logged numbers (relative tolerance 1e-9). Otherwise it fails.
    - The maps are exported as plain numbers (`serve/calib.py`: T, Platt coefficients, isotonic thresholds, quantiles), so live inference needs no scikit-learn. A test shows the exported maps equal the scikit-learn ones to 1e-12.
  - **XGBoost class lists:** the classes each head was trained on are exported, so loading needs no training data (`TreeModel.present_classes` / `load_heads`).
  - **Also in the bundle:** the normaliser, the window, horizons and time-to-fault cap, the sustain time from `eval.yaml`, the evaluated operating subset, and a model card.
- **Autoencoder bundle** (`autoencoder`):
  - The Phase 9 detector, refitted on the E1 training split with seed 0. The seed was fixed in `deploy.yaml` before any build, not chosen on results.
  - A strict build checks that its per-kind validation thresholds (p99, p99.5, p99.9) equal the logged `e5_anomaly` thresholds.
  - A separate bundle, because an XGBoost bundle cannot refit a torch model in its process, and the monitor worker then needs only this bundle.
- **Model card:** in every bundle, shown in the UI. It says that training and calibration used simulated data only, that Phase 10 did not demonstrate early warning on real machines, and lists the main limitations.

### Registry

- Each build is logged as an MLflow run in the experiment `P11_deploy` and registered as a new version of `linetwin-<name>`, through the MLflow model registry of the local file store.
- `mlflow.register_model` in MLflow 3 needs a "logged model", so the bundle uses `MlflowClient.create_registered_model` / `create_model_version`. Tested here with `mlflow-skinny`: versions, tags and `latest` work.
- Live code resolves `<name>` + version (a number or `latest` = highest), then downloads `runs:/<run_id>/bundle`. That path still works if the repository moves; the registered source URI is absolute.
- **The UI shows the version label,** for example `linetwin-gru v1 (gru seed 0) + linetwin-autoencoder v1 (autoencoder seed 0)`, in every `Prediction.modelVersion` and in `GET /predict`.

### Built here (2026-11-19, strict, measured)

| Bundle | Version | Seed | Checks | Build time |
|---|---|---|---|---|
| `linetwin-autoencoder` | v1 | 0 | thresholds reproduced (run `e26979ea…`) | 176 s |
| `linetwin-gru` | v1 | 0 | test predictions identical (max diff 0.0); calibration reproduced (run `4e05d309…`) | 559 s |
| `linetwin-xgb` | v1 | 0 | test predictions identical; calibration reproduced (run `cb78b359…`) | 23 s |

### ONNX: not adopted

- Live latency through the worker processes is far below the budget without ONNX. Per machine per tick, from `live_inference.md`: GRU 1.20 ms p50 and 9.3 ms max; XGBoost 1.53 ms p50 and 22 ms max; budget 100 ms.
- The model forward pass is 1.7–2.3 ms p50 per batch of four machines. Feature building (buffer transform) costs about as much.
- ONNX would add a conversion step, a second runtime and a parity check for no needed gain. The one slow part, Integrated Gradients, runs through autograd, which ONNX Runtime does not provide.

### Retraining: not built

`POST /retrain` was optional; the decision with the user is to skip it. Retraining on recent stored data would register an unevaluated version: it needs its own split, leakage and pre-registration protocol, as Phases 7–10 had. A new version comes from `deploy-models` on evaluated runs.

## Consequences

- The live twin shows exactly the evaluated model and confidence maps, and a bundle that does not reproduce its evaluation cannot be built in strict mode.
- **Tests and CI** use non-strict bundles built from tiny models on quick data (`twin_ml.testing.serve_setup`). The bundle records `"no reference run (non-strict build)"` and the UI shows that check status.
- Rebuilding the same configuration registers a new version with identical content. `latest` picks it; pinning a number is possible in `inference.yaml`.
- `TreeModel.load` keeps its signature. `load_heads` / `present_classes` are additions, and the Phase 8–10 workers are unchanged.

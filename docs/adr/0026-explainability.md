# ADR 0026: Explanations: TreeSHAP, path attributions, Integrated Gradients, sanity checks

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** 9 (unseen faults, anomaly detection, explainability)

## Context

The twin highlights the parts that drive a warning on the 3D model (the Prediction contract's `attributions [{sensor, part, weight}]`). An explanation is only useful if it reflects what the model actually uses. A plausible-looking heat map can be unrelated to the model's computation. Phase 9 therefore builds explanations and tests them with two sanity checks fixed in ADR 0024:
- agreement with the physics of each fault type;
- a deletion test.

No new dependency was added. The `shap` and `captum` packages were not installed, by decision in the plan review.

## Decision

### Methods

| Model | Method | Explained output | Additivity |
|---|---|---|---|
| XGBoost | native TreeSHAP, `Booster.predict(pred_contribs=True)` up to the early-stopping best iteration | the margin of the predicted alarm class at +30 s | contributions + bias = margin (unit test) |
| Random Forest | Saabas-style decision-path contributions from the fitted trees' arrays | the predicted probability of the alarm class | contributions + bias = forest probability (unit test) |
| LSTM, GRU, Transformer | Integrated Gradients, implemented in torch | the logit of the predicted alarm class at +30 s | sum = F(x) − F(baseline) up to the trapezoid error (reported) |

- **Explained class:** whichever of WARNING and FAULT the model gives the higher probability (`twin_ml.explain._alarm_class`). Positive weights are evidence for the alarm.
- **Integrated Gradients:** settings fixed in ADR 0024.
  - Baseline: zeros after normalisation (the training mean) for every value channel; missing-mask channels 0 ("not missing"); kind channels equal to the input's own.
  - Steps: 64 equal intervals, trapezoid rule.
  - The completeness error per window is returned and reported. A unit test checks that IG on a linear model equals (x − baseline) · w exactly, and that the kind channels get exactly zero.
- **Random Forest path attributions are exploratory.**
  - They are not Shapley values.
  - They are known to over-credit features that are split near the root of the trees, because every sample's path passes those splits.
  - RF agreement is reported separately and not compared with the other models.

### Aggregation to sensors and parts (`twin_ml.explain.aggregate`)

- **Summary features:** `<channel>_<stat>` goes to the channel's sensor, with `vib_norm` going to vib. `<sensor>_missing_frac` goes to that sensor. The three kind features are dropped.
- **Sequence channels:** a value channel goes to its sensor, with `vib_norm` going to vib. `<sensor>_missing` goes to that sensor. Kind channels are dropped. IG attributions are summed over time first.
- **Weights:** attributions are summed per sensor (signed) and normalised by the sum of absolute values.
- **Parts:** each sensor goes to the 3D part given by `sensor_parts` in `config/line.yaml` (for example, CNC vib → `spindle_bearing`).
- **Test:** a unit test checks that every model input maps to exactly one sensor or is a kind feature.

### Phase 11 API

`twin_ml.explain.explain(model, values, mask, *, machine_kind, sensor_parts, horizon=30, ig_steps=64) -> [{sensor, part, weight}]`:
- `values` (W × 6) and `mask` (W × 5) are exactly what `Normalizer.transform` produces;
- the output is sorted by |weight|.

The module imports neither the simulator nor, at import time, any ML library (import test). Torch is loaded only when a sequence model is explained, and xgboost / scikit-learn only through the fitted model objects.

### Sanity checks (`twin_ml.experiments.explain_eval`; `python -m twin_ml.experiments.e5 --explain`)

- **Explained windows:**
  - For every gradual episode of the clean E1 test split (seen fault types) and of the E5 set: the windows ending between the episode's start and its fault, in which the model's sustained +30 s alarm is on.
  - Thinned to one per 10 s and capped at 30 per episode (evenly spaced).
  - Episodes the model never alarms on are not explained, and the report gives their count.
- **(a) Agreement (primary for XGBoost and the sequence models):**
  - The episode's top sensor is the largest mean signed weight over its explained windows.
  - It agrees if it equals the fault type's leading physical signal: bearing_wear → vib, overheating → temp, spindle_imbalance → vib, lubrication_loss → current.
  - Per-run sums of (episodes, agreeing) per fault type give the rate and its run-level bootstrap CI.
- **(b) Deletion test (exploratory):**
  - For every explained window, the top-attributed sensor's value channels (plus `vib_norm` for vib) are set to the baseline (0), and so are one random other sensor's.
  - The random sensor is seeded by (`deletion_seed` 17, window index).
  - Reported: the mean |Δ p(WARNING) + p(FAULT)| for each, their difference with a run-level bootstrap CI, and the share of windows where the top sensor moves the prediction more.
  - For the trees, the summary features are recomputed from the modified window.
- **Completeness:** the IG completeness error (mean, and relative to |F(x) − F(baseline)|).
- **Exploratory:** the autoencoder's per-sensor reconstruction-error shares on alarmed precursor windows and on sensor-fault windows (ADR 0025).

### Runs

- **Study `explain`:** one MLflow run per (model, seed) of the E1 models (XGBoost, LSTM, GRU, Transformer; RF exploratory), each in its own process.
- **Reuse:** the models are loaded as in the E5 studies, and must reproduce their stored predictions.
- **Run contents:** the per-run sums (`agree`, `deletion`, `ig_complete`), per-episode records and per-fault-type mean weights.

## Consequences

- **What agreement measures:** a high agreement rate means the model leans on the signal the physics says leads the fault. It does not prove that the explanation is faithful; the deletion test addresses that, within its limits.
- **Limits of the deletion test:** replacing a sensor with its training mean creates inputs that may lie outside the training distribution, so a large change can partly reflect that rather than reliance on the sensor.
- **Limit for lubrication_loss:** for an unseen fault, the supervised models were never taught the current-led signature. Agreement there tests whether their alarms happen to rest on current, and the result has to be read together with whether they alarm at all (E5 miss rate).
- **Results:** in the generated [e5_explain.md](../../experiments/results/e5_explain.md) (Part C).

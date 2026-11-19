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
- **Results:** below, from the generated [e5_explain.md](../../experiments/results/e5_explain.md).

## Results (2026-11-17)

### Primary: agreement rate (64 IG steps; E5 episodes explained in seed 0 in brackets)

| Model (method) | Clean, seen types | E5 (`lubrication_loss`) | Pooled (as pre-registered) |
|---|---|---|---|
| XGBoost (TreeSHAP) | 1.000 | 0.178 [0.107, 0.257] (90 of 120) | 0.363 |
| LSTM (IG) | 1.000 | 0.251 [0.166, 0.339] (45 of 120) | 0.650 |
| GRU (IG) | 0.974 [0.931, 1.000] | 0.228 [0.155, 0.328] (22 of 120) | 0.512 |
| Transformer (IG) | 1.000 | 0.000 [0.000, 0.000] (7 of 120) | 0.719 |

- **Seen faults:** explanations agree with the physics on every seen fault type (bearing wear and spindle imbalance → vib, overheating → temp), for every model.
- **Unseen fault:** they do not agree on E5. The models' alarms rest on current in at most a quarter of the explained episodes.
- **Read clean and E5 separately, not pooled.** The pooled rate weights the conditions by how many episodes a model explains, and the models explain very different numbers of E5 episodes. The Transformer's pooled 0.719 comes almost only from its 26 clean episodes. This is a reading clarification of the pre-registered "overall" rate, not a change of it.
- An explanation exists only for episodes the model alarms on. On E5 that is a minority of episodes for the sequence models, and it varies by seed (LSTM 45 / 53 / 8, GRU 22 / 53 / 95, Transformer 7 / 33 / 2 of 120).

### Exploratory

- **Deletion test** (Δ p(WARNING) + p(FAULT), top-attributed sensor minus a random other sensor):
  - **Clean:** +0.836 to +0.907 for every model, and the top sensor moves the prediction more in 0.98–1.00 of the windows. The explanations pass.
  - **E5, sequence models:** +0.358 to +0.383, so their attributions still identify inputs the prediction depends on, just not the physically expected one.
  - **E5, XGBoost: the deletion test fails.** −0.046 [−0.062, −0.028]: removing a random other sensor moves the prediction more than removing the top-attributed one, and top > random holds in only 0.530 of the windows. On the unseen fault, XGBoost's TreeSHAP ranking does not identify what the prediction depends on.
  - Limit: replacing a sensor with its training mean can leave the training distribution (see Consequences).
- **IG completeness error at the pre-registered 64 steps:** relative error 0.061 (LSTM), 0.048 (GRU) and 0.012 (Transformer) on clean windows, 0.002–0.012 on E5. That is larger than the unit-test networks. It is reported as is; the 64-step results stay primary.
- **IG step-count sensitivity (ADR 0024 amendment; LSTM, 256 steps, same windows):**
  - the completeness error falls to 0.0086 (clean) and 0.0002 (E5);
  - no episode's top sensor changes (0 of 78 clean, 0 of 106 E5 episode explanations over the three seeds), so the agreement rates are identical (Δ 0.000, 0/3);
  - the step count does not affect the conclusions.
- **Random Forest (Saabas, not comparable):** 1.000 on clean, 0.000 on E5; deletion test +0.878 / +0.500.
- **Autoencoder error shares:** they agree with the physics on clean episodes (26 of 26) and point at vibration, not current, in all 120 E5 episodes (ADR 0025 § Results).

### Consequence for the live twin (Phase 11)

- **Seen faults:** the part highlight (`attributions`) can be trusted for warnings of seen fault types; agreement and the deletion test both support it.
- **Unseen faults:** the same highlight on an unseen fault points at the wrong part (vibration parts instead of the current-led drive), and for XGBoost it is not even faithful to the model.
- Explanations therefore do not reveal novelty. Phase 11 should show them only with supervised warnings, and show autoencoder-only alarms as "unusual behaviour" without a part highlight (ADR 0024 § Outcome).

## Addendum: real data (Phase 10, 2026-11-19, exploratory)

Method: `docs/experiments.md` § Phase 10, "Explanations and replay demo". Numbers: the generated [real_data.md](../../experiments/results/real_data.md) § Explanations. All 9 explanation runs passed their reproduction checks: restored models reproduced their stored test predictions, and re-fitted autoencoders their stored alarms.

- **Azure PdM:**
  - **Expected sensor per component**, derived from the training split only: comp1 → volt (Δ +1.29), comp2 → rotate (−1.48), comp3 → pressure (+2.24), comp4 → vibration (+1.86). The other sensors' Δ are at most 0.03 in absolute value, so each component has one clearly dominant signal.
  - **Agreement** of the event's top sensor with the expected one, on every warned evaluable test event (chance 0.26):
    - LSTM (Integrated Gradients): 0.962 / 0.939 / 0.943 over seeds;
    - XGBoost (TreeSHAP): 0.925 / 0.924 / 0.934.
    - Misses are mostly comp1 and comp4 events.
  - So, as on the simulator's seen faults, the part highlight points at the expected sensor for the failure kinds the models were trained on. The benchmark's one-signal-per-component structure (likely synthetic) makes this an easy case.
- **MetroPT-3, autoencoder error shares** (not attributions in a strict sense):
  - **During** each reported failure, the largest share is an air-leak-plausible signal in every seed: `dv_pressure` for #1–#3 and `lps` for #4, with a plausible mass of 0.79–0.90. The normal-window reference is 0.564, and the top sensor is plausible in 0.218 of normal windows.
  - **Before #1**, the top sensor is `pressure_switch`, which is not plausible (mass 0.05–0.07). This supports reading that alarm as an anomaly unrelated to the leak (ADR 0027 § Outcome).
  - **Before #2 and #3**, the single alarmed window per seed (when there is one) points at `dv_pressure`.
- **Consequence for Phase 11:** unchanged.
  - Show attributions with supervised warnings of known failure kinds.
  - On MetroPT-3 the error shares do name plausible signals during a failure, which could label an "unusual behaviour" alarm while it is active, but four failures are no basis for trusting it before one.

## Process isolation for explanations (2026-11-17)

- **Separate processes:** XGBoost (TreeSHAP) and the sequence models (Integrated Gradients, torch) are never explained in the same process. Their OpenMP runtimes crash on macOS when both are loaded (ADR 0020, amendment of 2026-11-17).
- **Fail-fast check:** `twin_ml.explain.require_alone` runs inside `explain()`, `attribute()` and `probabilities()`, and raises `OpenMPClashError` if the other library is already loaded.
- **Phase 11:** the live twin's explanation service must explain tree models and sequence models in separate worker processes, the same ones that run their inference.
- **Tests:** the XGBoost explanation tests run in their own pytest process (`ml/tests/xgb_isolated_explain.py`).

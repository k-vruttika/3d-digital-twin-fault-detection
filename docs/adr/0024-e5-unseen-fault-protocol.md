# ADR 0024: E5 protocol: unseen fault data, anomaly detection, combined policy, explanations, and pre-registered outcomes

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** 9 (unseen faults, anomaly detection, explainability)

## Context

The last part of the research question asks whether a data-driven twin still warns when a fault type was never seen in training, and whether its confidence stays trustworthy then. Phases 7 and 8 trained and stressed the supervised models only on the four training fault types. Phase 8 also showed that every model raises false alarms under unseen sensor drift, so the models cannot tell sensor problems from degradation (ADR 0021 § Outcome).

The simulator already holds out one fault type, `lubrication_loss` (`config/simulator.yaml`, `held_out: true`). It is current-led, with stick-slip vibration bursts and speed jitter, and it trips on overcurrent (ADR 0003, item 7). No training scenario samples it. The E1 dataset records it as held out, and a test asserts it is absent from the training vocabulary.

Phase 9 needs a protocol fixed before any E5 data exists and before any study runs: the test data, the anomaly detectors and their alarm thresholds, the combined policy, the explanation checks, and which outcomes are primary. All of it is in this ADR and in `config/models/e5.yaml`. **This ADR and the config are committed before E5 data is generated.**

## Decision

### E5 test data (fixed in advance)

- **Scenario:** `config/scenarios/e5_lubrication_loss.yaml` (unchanged). Only `lubrication_loss` is sampled, at 0.5 per machine-hour (the default fault rate), earliest onset 120 s, all three load bands.
- **Seed:** base seed 42, the seed of `data/sim`, given on the command line (`--seed 42`), which overrides the file's own seed. Run *i* uses `SeedSequence(42, spawn_key=(i,))`.
- **Runs: 100, fixed now. No runs are added after the data is seen.**
  - **The 30 E1 test-split run indices.** The simulator draws load band, ambient, per-machine load level and thresholds first on its plan stream, and the process, measurement and speed-change noise on separate streams. These runs therefore share all of that with the clean test runs; only the fault plan differs.
  - **70 fresh run indices, 200–269.** They are outside the 200 E1 runs, so no split has ever used them.
  - **Excluded: the training and validation run indices** (the rest of 0–199). Their normal segments would repeat noise the models were trained on.
- **Expected size:** a plan-only estimate (planned onsets plus progression times, before simulation) gave about 37 episodes for the 30 test-run indices and about 110 for the 100 runs. The realised count is reported, whatever it is.
- **Generation:** `uv run poe sim-e5` generates run indices 0–269 into `data/sim_e5/` (about 20 s), and the evaluation selects the 100. The manifest hashes are checked as for every simulator set.
- **Windowing:** the E1 dataset's normaliser, vocabulary and config are reused, as for the Phase 8 test sets.
  - `lubrication_loss` is appended to the label vocabulary of the E5 split only, because the training vocabulary does not contain it.
  - Its episodes are gradual episodes (`GRADUAL_FAULTS`), so early warning is measured exactly as for the seen types.
  - The models' fault-type heads keep their trained classes. Fault-type metrics are not computed on E5; the distribution of predicted fault types is exploratory.
- **Clean false alarms** are measured on the clean E1 test split (30 runs, 26 gradual episodes), the same data as E1 and Phase 8.

### Anomaly detectors

- **Training windows:** normal operation only, from the dataset's training split.
  - The window's end state is RUNNING.
  - No fault is active anywhere in the window (`fault_type` = none on every row).
  - The health target at +30 s is NORMAL.
  - The same rule on the validation split gives the scores that set the thresholds.
- **Isolation Forest:** on the 50 window summary features (`twin_ml.features.summary`). Settings in `config/models/iforest.yaml`; `random_state` = seed. Score = `-score_samples`.
- **Autoencoder:**
  - Input: the 6 normalised value channels × 30 s, conditioned on the machine kind (one-hot into encoder and decoder).
  - Network: a small MLP, 180 → 64 → 16 → 64 → 180.
  - Training: a fixed budget of 20 epochs with Adam, seeded. The epoch with the lowest validation normal-window reconstruction loss is kept. Settings in `config/models/autoencoder.yaml`.
  - Score: the mean squared reconstruction error over observed values.
- **Seeds:** 3 per detector (0, 1, 2), as for the supervised models.
- **Process isolation:** each detector runs in its own process, like every model since ADR 0020. The autoencoder uses torch; the Isolation Forest uses scikit-learn.
- **Alarm threshold:**
  - **Primary:** the 99.9th percentile of the validation normal-operation scores, one threshold per machine kind.
  - **Exploratory:** the 99.0th and 99.5th percentiles, to show the trade-off between misses and false alarms. Conclusions use 99.9 only.
  - The percentiles were chosen before any test result and are never tuned on test data.
- **Alarm:** score above the threshold, sustained for `sustain_s` (3 s, `eval.yaml`) exactly like the supervised alarm. Lead time, miss rate and false alarms per hour therefore use the same code (`twin_ml.evaluate.early_warning`).

### Supervised models and studies

| Study | Models | Conditions |
|---|---|---|
| `e5_e1models` | persistence, threshold, and the E1 models (RF, XGBoost, LSTM, GRU, Transformer × 3 seeds), reused; RF retrained from its seed, as in Phase 8 | clean, `e5`, `sf_drift` |
| `e5_aug` | the augmentation-trained models of Phase 8, reused | clean, `e5`, `sf_drift` |
| `e5_anomaly` | Isolation Forest and autoencoder × 3 seeds, trained on the E1 training split | clean, `e5`, `sf_drift` |
| `e5_e4` | the E4 models (low + medium load) and both detectors trained on the E4 training split | `e4_high`, `e4_in` |

- **No supervised model is retrained for E5,** except the RF retrain from its seed. Every reused model must reproduce its stored clean test predictions, as in Phase 8.
- **Calibration and conformal sets** are fitted on clean validation data with the Phase 8 code and settings (ADR 0022, `robustness.yaml`), and applied unchanged to E5.
- **E4 detectors:** they are trained on low + medium load only. That is what makes "does a load change alone cause anomaly false alarms" testable; the E1 training split already contains high load.

### Combined policy

- **Alarm:** the supervised model's sustained alarm OR the detector's sustained alarm, per window.
- **Pairs:** {threshold, RF, XGBoost, LSTM, GRU, Transformer} × {Isolation Forest, autoencoder}, for both the E1 and the augmentation-trained models. Seed *i* is paired with detector seed *i*; the threshold rule (one run) is paired with each detector seed.
- **Thresholds:** primary at the 99.9th percentile; 99.0 and 99.5 exploratory.
- **How it runs:** a pure numpy step reads the stored per-window alarms of the two source runs from MLflow and logs a combined run.

### Explanations

- **XGBoost:** native TreeSHAP (`pred_contribs=True`) on the +30 s head, for the predicted non-NORMAL class.
- **LSTM, GRU, Transformer:** Integrated Gradients, implemented in torch (no new dependency).
  - **Baseline:** all zeros after normalisation (the training mean), missing-mask channels 0 ("not missing"), kind channels equal to the input's own (so the kind gets no attribution).
  - **Steps:** 64, trapezoid rule.
  - **Target:** the logit of the predicted non-NORMAL class at +30 s.
  - The mean completeness error on the explained windows is reported as a quality check.
- **Random Forest:** Saabas-style decision-path contributions, implemented from scikit-learn's tree arrays.
  - These are exploratory and not Shapley values.
  - Path attributions are known to be biased: they over-credit features that are split near the root.
  - RF agreement is reported separately and not compared with the other models.
- **Aggregation:**
  - Feature or channel attributions are summed per sensor (`vib_norm` and every mask or missing-fraction feature go to their sensor; kind features are dropped).
  - They are then mapped to machine parts through `sensor_parts` in `config/line.yaml`.
- **Explained windows:** alarmed precursor windows (from episode start to the fault) of the clean test gradual episodes and the E5 episodes, one every 10 s, at most 30 per episode.
- **Sanity check (a), agreement:**
  - Per episode, the top sensor of the mean attribution over its explained windows is compared with the fault type's leading physical signal: bearing_wear → vib, overheating → temp, spindle_imbalance → vib, lubrication_loss → current (`explain.expected_sensor`).
  - The agreement rate is reported per fault type and overall, with a run-level bootstrap CI.
- **Sanity check (b), deletion test (exploratory):**
  - The top-attributed sensor's value channels are replaced by the baseline, and so are one random other sensor's (seeded per window).
  - The change in p(non-NORMAL) must be larger for the top sensor. This is reported as a paired difference with a bootstrap CI.
- **Phase 11 API:** `twin_ml.explain.explain()` returns `[{sensor, part, weight}]` and is importable without `twin_sim`.

### Statistics

- **CIs:** all from the run-level bootstrap (1000 resamples, 95 %, percentile), from per-run sums, as in Phase 8.
- **Paired, same runs:** each model, detector and combined policy vs the threshold baseline on E5 and on the clean test runs; each combined policy vs its supervised model alone.
- **Unpaired, different runs:** E5 vs clean conformal set size; `e4_high` vs `e4_in`.
- **Averaging over seeds:** mean of the per-seed point estimates and CI bounds, with the number of seeds whose CI excludes zero.
- **Episode counts** are reported next to every figure.

## Pre-registration (fixed 2026-11-17, before any E5 data was generated or any Phase 9 study run)

**Primary outcomes,** each at horizon +30 s, with a 95 % run-level bootstrap CI and paired against the threshold baseline on the same runs where the runs match:

1. **Early warning,** for each supervised model (`e5_e1models` and `e5_aug`), each anomaly detector at the 99.9th percentile, and each combined policy (model OR detector, at the 99.9th percentile):
   - E5 miss rate (`miss_rate`, on `e5`);
   - E5 lead time relative to the simulator's threshold WARNING (`rel_lead_mean_s`, on `e5`);
   - clean false alarms per hour (`false_alarms_per_h`, on the clean test runs, `clean`).

   Each combined policy is also paired against its supervised model alone.
2. **Uncertainty under the unseen fault:** the Mondrian LAC mean conformal set size (`size_mondrian_lac`) on E5 operating windows vs clean operating windows, per supervised model (unpaired).
3. **Explanation agreement rate** (`agreement_rate`) for XGBoost (TreeSHAP) and the LSTM, GRU and Transformer (Integrated Gradients).

**Everything else is exploratory.** This covers:
- the 99.0 and 99.5 percentiles;
- `sf_drift` (drift-segment flag rate, false alarms inside drift segments);
- the E4 load shift (`e4_high` vs `e4_in`);
- the confidence of wrong predictions on E5 (calibrated top-label confidence, share above 0.9);
- the fault-type head on E5;
- the +10 s horizon;
- macro F1, ECE, coverage and AUPRC;
- the deletion test and the IG completeness error;
- the RF path attributions;
- the autoencoder's per-sensor reconstruction errors.

**Rules:**
- The list is stored in `config/models/e5.yaml` (`primary:`), and the report reads it from there. A test checks that the config matches this ADR.
- No setting in `e5.yaml`, `iforest.yaml` or `autoencoder.yaml` is changed after results are seen.

**Interpretation note, fixed in advance.** The threshold baseline checks the observed current against the nominal overcurrent limit (`current_warning_x` / `current_fault_x` × rated current). That is exactly the channel `lubrication_loss` trips on by design. No supervised model was trained on a current-led fault. Any advantage of the threshold rule on E5 must be read with this in mind: it reflects a hand-set limit on the very signal of this fault, not a general ability to detect unseen faults.

## Consequences

- **Reproducible from config:** the E5 set is regenerated from the scenario, seed and run selection, in about 20 s.
- **Small held-out signature space:** E5 tests one unseen fault type in one simulator, so it cannot show how the models generalise to unseen faults in general.
- **Threshold reference:** the threshold reference favours E5 by construction (see the interpretation note). The anomaly detectors and the combined policy are therefore also compared with the supervised models alone.
- **Exchangeability:** split-conformal guarantees assume exchangeable data, so they do not hold on E5. Set size is reported as a measure of uncertainty, not as coverage.
- **New config only:** Phase 8 configs and study hashes are unchanged.

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

## Amendment (2026-11-17, after the Part A and B results): exploratory follow-ups only

Two exploratory checks were added after the E5 results had been seen. They change no primary outcome, no setting in `e5.yaml`, `iforest.yaml` or `autoencoder.yaml`, and no earlier study hash; `--dry-run --skip-existing` of the `explain` study still shows 0 runs to do. Their settings are in a separate file, `config/models/e5_followup.yaml`.

1. **Autoencoder per-sensor error shares.** This analysis was already listed as exploratory above. The original runs stored the shares only pooled over all runs, which gives no run-level CI.
   - Study `ae_shares` (`uv run poe e5 --ae-shares`) re-fits the autoencoder from each seed, because its weights were not stored. Each re-fit must reproduce the source `e5_anomaly` run's stored sustained alarms at the 99.9th percentile bit for bit, on every condition, or the run fails.
   - Windows are selected as for the explanations: alarmed, one per 10 s, at most 30 per unit.
   - Units: the E5 and clean precursor episodes, with the expected sensor per fault type from `explain.expected_sensor`; and the flagged drift segments of `sf_drift`, where the expected sensor is the drifting one. A segment is flagged by a sustained alarm on a negative window inside it, as in ADR 0025.
   - Unit agreement: the top sensor of the unit's mean share is the expected one. Window agreement: each window's own top share is the expected one. Per-run sums give run-level bootstrap CIs.
2. **Integrated Gradients step count.** Study `explain_ig_steps` (`uv run poe e5 --ig-sensitivity`) reruns the LSTM explanation study at 256 steps. Models, windows, baseline and target are unchanged; only the step count differs. The 64-step results stay primary.
3. **Report additions.** The report also gains a per-seed table of the E5 miss rate, the lead and the explained E5 episodes. It shows the seed-to-seed variation, which the seed-averaged cells hide.

## Consequences

- **Reproducible from config:** the E5 set is regenerated from the scenario, seed and run selection, in about 20 s.
- **Small held-out signature space:** E5 tests one unseen fault type in one simulator, so it cannot show how the models generalise to unseen faults in general.
- **Threshold reference:** the threshold reference favours E5 by construction (see the interpretation note). The anomaly detectors and the combined policy are therefore also compared with the supervised models alone.
- **Exchangeability:** split-conformal guarantees assume exchangeable data, so they do not hold on E5. Set size is reported as a measure of uncertainty, not as coverage.
- **New config only:** Phase 8 configs and study hashes are unchanged.

## Outcome (2026-11-17, Phase 9 complete)

Every number below is from the generated [e5_explain.md](../../experiments/results/e5_explain.md): the mean over seeds of the per-seed estimate [mean per-seed 95 % run-level CI], at +30 s, with detectors at the 99.9th percentile. Data: 100 runs and 120 realised `lubrication_loss` episodes (CNC 33, conveyor 58, inspection 29; the plan estimate was about 110). Clean reference: 30 runs, 26 episodes.

### Primary outcomes

1. **Early warning.**
   - **Supervised models miss 25–88 % of the unseen-fault episodes.**
     - E1 models: XGBoost 0.250, GRU 0.528, RF 0.547, LSTM 0.706, Transformer 0.883.
     - Augmentation-trained models: 0.278–0.864.
     - Every one misses more than the threshold rule (0.042; CI excludes zero in 3/3 seeds), with no clean false alarms (Transformer 0.01 per hour).
   - **Lead when a supervised model does warn:** the trees warn early (RF 394 s, XGBoost 620 s before the simulator WARNING); the sequence models warn −4 to +117 s.
   - **Autoencoder:** detects every E5 episode (miss rate 0.000) with a lead of 656 s, at a cost of **0.57 [0.31, 0.89] false alarms per hour on clean data** (3/3 vs threshold).
   - **Isolation Forest:** misses 0.992 of E5 episodes and still raises 0.23 false alarms per hour.
   - **Combined policy:**
     - **"supervised OR autoencoder"** reaches a 0 % E5 miss rate for every supervised model. Δ vs the model alone is −0.25 (XGBoost) to −0.88 (Transformer), 3/3 seeds. The lead becomes 656 s, and the combined policy keeps every supervised warning by construction (OR only adds alarms).
     - The price is the autoencoder's false alarms: +0.56 to +0.57 per hour on clean data, 3/3 seeds.
     - "Supervised OR Isolation Forest" changes the miss rate by at most −0.008 (0/3) and adds 0.23 false alarms per hour.
2. **Uncertainty under the unseen fault.** The Mondrian LAC mean set size on E5 differs from clean by −0.046 to +0.002, and no seed's CI excludes zero for any of the 10 supervised models. **Calibration and conformal prediction do not reveal the novelty:** the sets do not grow when the fault is new.
3. **Explanation agreement:** see ADR 0026 § Results.
   - Clean, seen types: 0.974–1.000.
   - E5: 0.000–0.251.
   - **Read clean and E5 separately, not pooled.** The pre-registered pooled rate (0.363–0.719) weights the conditions by how many episodes each model explains. A model that alarms on few E5 episodes is dominated by its clean episodes: the Transformer explains 7 of 120 E5 episodes (seed 0), so its pooled 0.719 says little about E5.

### Exploratory results

- **Confidence when wrong:** on E5 the sequence models are wrong with calibrated confidence ≥ 0.9 in most missed windows (a WARNING or FAULT target predicted NORMAL):
  - LSTM 0.947, GRU 0.881, Transformer 0.974 (augmentation-trained 0.797–0.949);
  - the trees less often: RF 0.057, XGBoost 0.585 (augmentation-trained 0.107 / 0.304).
- **Fault-type head:** on E5 precursor windows it mostly predicts `bearing_wear` (0.42–0.83 of windows). That is the most common prediction for 9 of 10 models; LSTM (aug) predicts `none` most often (0.50).
- **Detector percentiles:**
  - The Isolation Forest at the pre-registered 99.9th percentile is nearly useless (0.992 E5 miss rate, 0.974 clean miss rate).
  - At p99 it detects nearly all E5 episodes (miss rate 0.003) at 0.91 false alarms per hour, with a clean miss rate of 0.179.
  - The autoencoder misses no E5 episode at any percentile (0.54–0.66 false alarms per hour).
  - **The primary result, at p99.9, stands unchanged;** the lower percentiles were not chosen in advance.
- **Load shift (E4) and drift:**
  - Under the E4 load shift, the autoencoder's false alarms rise from 0.43 (`e4_in`) to 1.10 per hour (`e4_high`), +0.67 [0.20, 1.07] in 3/3 seeds. The supervised models stay at 0.00–0.03.
  - Inside `sf_drift` segments, the autoencoder raises 2.63 false fault warnings per hour, against 0.22–0.97 for the supervised models.
- **Threshold baseline:** its overcurrent advantage on E5 (miss rate 0.042, lead 23 s) is the one foreseen in the interpretation note above. It checks the very signal `lubrication_loss` trips on, so it does not show a general ability to detect unseen faults.
- **Seed-to-seed variation:** the E1 sequence models vary strongly between seeds on E5, so their seed means hide very different models.
  - E5 miss rate by seed: GRU 0.817 / 0.558 / 0.208; LSTM 0.625 / 0.558 / 0.933; Transformer 0.942 / 0.725 / 0.983. Explained E5 episodes vary with them (LSTM 45 / 53 / 8).
  - XGBoost is stable (0.242–0.258). The augmentation-trained sequence models vary less (GRU 0.458–0.583).
- **Amendment follow-ups:** the autoencoder error shares and the IG step-count check are in ADR 0025 and ADR 0026, § Results.

### Answer to the research question (unseen fault type)

- **The supervised twin is not reliable for an unseen fault type,** and its reported confidence does not warn of that:
  - it misses a quarter to nine tenths of the episodes;
  - the sequence models are confidently wrong on most missed windows;
  - the conformal sets do not grow;
  - the fault-type head names a known fault.
- **A normal-behaviour autoencoder closes the detection gap** (0 % misses, early). It pays with false alarms that grow under a load shift and inside sensor drift.
- **Recommendation for Phase 11:** run "supervised OR autoencoder", but present autoencoder-only alarms in the live twin as a separate **"unusual behaviour"** category, not as a fault warning with a fault type or a highlighted part.
  - Neither the supervised explanations nor the autoencoder's error shares point at the right sensor on E5 (ADR 0026, ADR 0025).
  - The autoencoder's false-alarm rate is too high to treat its alarms as fault warnings.

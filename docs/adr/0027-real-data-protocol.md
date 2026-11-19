# ADR 0027: Real-data validation protocol (MetroPT-3, Azure PdM) and pre-registered outcomes

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** 10 (validation on real public data; completes Milestone B)

## Context

Phases 7–9 answered the research question on simulator data only: noise and missing values (E2/E3), condition shift (E4) and an unseen fault type (E5). Phase 10 asks what holds on public data that the simulator did not produce, with the same models and the same evaluation code (ADR 0028 makes that code dataset-agnostic):

- **MetroPT-3** (UCI, CC BY 4.0): real operational data from the air production unit of a metro train compressor. 15 signals, February–September 2020. The dataset is unlabeled; the company's reports list **four air-leak failures** (`config/datasets/metropt3.yaml`, with source and a manual check against `Data Description_Metro.pdf`).
- **Azure PdM** (Kaggle, from a Microsoft tutorial): 100 machines, hourly telemetry for 2015, 719 component-failure events, error and maintenance logs. **Its origin is widely understood to be synthetic.** It is used as a public benchmark with enough failures for supervised models, never described as real plant data.

The data audit (`uv run poe real-audit`, logged to MLflow `P10_real`, rendered in `experiments/results/real_data.md`) was run before this ADR. It measured the facts the design depends on:

- MetroPT-3 is sampled at about 10 s (88 % of intervals exactly 10 s), not at the 1 Hz the PDF states; the PDF's row count is exactly 10× the file's.
- MetroPT-3 has 331 gaps over 60 s (910 h in total, the longest 48 h).
- All four failures lie after 11 April 2020, so a time-ordered split puts every failure in the test period.
- Azure is exactly hourly with no gaps or missing values. Every failure is stamped at 03:00 or 06:00, and 98 % of failure events have an error-log entry in the 24 h before.

No model had been trained on real data when this ADR and its configs were written. No outcome had been seen.

## Decision

### Data and labels

- **Loaders** (`twin_ml/datasets/`, no simulator, no ML library) read the files from `data/external/<name>/`. A missing file raises an error that says what to download and where. Nothing is downloaded automatically.
- **Grid:**
  - MetroPT-3: 1-minute means of the ~10 s samples. Digital 0/1 signals become duty fractions. Empty minutes stay as missing rows (forward fill + missing mask). A run of more than 10 empty minutes starts a new series.
  - Azure: the native hourly rows.
- **Inputs:** all 15 MetroPT-3 signals; the four Azure telemetry channels. The machine kind is the one-hot input: `APU` for MetroPT-3, the machine `model` (model1–model4) for Azure. **Error counts are not inputs of any primary model.**
- **Pseudo-states from the failure events** (so the Phase 7–9 label code applies unchanged; later rules win):
  - MAINTENANCE: `post_failure` rows after a failure end (MetroPT-3: 24 h; Azure: none);
  - WARNING: `warning_band` rows before a failure start (MetroPT-3: 6 h; Azure: 24 h);
  - FAULT: from the failure start to its end (MetroPT-3: the reported interval; Azure: the failure hour);
  - RUNNING otherwise.
- **Fault type:** the failure's type from `fault_context` rows before its start to its end (MetroPT-3: 24 h, the largest lookback; Azure: 48 h). `air_leak`; for Azure the component, or `multi` for the 42 events with two components.
- **Split by time, never shuffled.** Rows are labelled on each machine's full timeline first, then cut, so no window or horizon crosses a split boundary.
  - MetroPT-3: train 2020-02-01 to 03-22, validation 03-22 to 04-11, test 04-11 to 09-02.
  - Azure: train 2015-01-01 to 09-01, validation 09-01 to 11-01, test 11-01 to 2016-01-02, all machines.

All settings are in `config/models/real_metropt3.yaml`, `real_azure.yaml` and `real.yaml`, in rows of each dataset's own grid.

### Windows, horizons, alarms

| | MetroPT-3 | Azure PdM |
|---|---|---|
| Window | 60 rows (60 min) | 24 rows (24 h) |
| Supervised horizons | none | +12 h, **+24 h (primary)** |
| Lookbacks | 2 h, **6 h (primary)**, 24 h | **48 h** (warning band + horizon) |
| Sustain | 3 rows (3 min) | 3 rows (3 h) |
| Train subsample | none | far-normal windows kept with p = 0.1, weight 10 (Phase 7 rule) |

- **Alarm:** a model's WARNING-or-FAULT prediction, or a detector's score above its threshold, sustained over 3 consecutive windows (as in Phases 7–9).
- **Warned:** a sustained alarm in `[start − L, start)` for lookback L. Lead time = start − first such alarm. **Detected during the failure** (MetroPT-3): an alarm in `[start, end]`, with its delay after the start; reported separately, never counted as a warning.
- **False alarms per day:** rising edges of the sustained alarm in negative test windows (state RUNNING and no fault type, i.e. more than the fault context before and more than the post-failure exclusion after any failure), per day of negative windows.
- **Evaluable failures:** a failure counts for a lookback only if the whole lookback lies inside the split and **at least 50 % of that lookback has data** (`evaluable_min_coverage: 0.5`). Otherwise it is reported as **not evaluable**, listed by number, and never counted as a miss. Counts are given out of the evaluable failures. The audit found all four MetroPT-3 failures evaluable at every lookback (data coverage 71–100 %).
- **Chance check, for every lookback:** from each method's measured onset rate λ (false alarms per day) and alarm-on fraction f on the test negatives, P(chance hit) = 1 − (1 − f)·e^(−λL). The expected number of failures warned by chance alone (the sum over the evaluable failures) is reported next to the actual count.

### Methods

- **MetroPT-3: no supervised models.** With a time-ordered split there is no failure in the training period, so a supervised model would have nothing to learn from. MetroPT-3 is a **case study**: for each of the four failures, was it warned about, how early, and how many false alarms happened over months of normal operation? The anomaly detectors are the main method. "Best supervised model" and "combined policy" do not exist for MetroPT-3.
- **Detectors (both datasets):** Isolation Forest and the autoencoder (ADR 0025 architectures and configs, input sizes from the schema) × 3 seeds.
  - Fitted on normal training windows (RUNNING, no fault in the window, NORMAL at the horizon).
  - Thresholds per kind from normal validation windows. **The 99.9th percentile is primary**; 99.0 and 99.5 are exploratory.
- **Supervised models (Azure):** RF, XGBoost, LSTM, GRU, Transformer with the existing configs × 3 seeds. Each (model, seed) runs in its own process; torch and XGBoost never share a process (ADR 0020).
- **Baselines** (no training; parameters fixed in `real.yaml`):
  - *threshold rule:* any sensor's window mean outside [p0.1, p99.9] of the training normal window means, per kind and sensor;
  - *rolling z-score:* z = (window mean − trailing mean) / trailing std over the 24 h (MetroPT-3) or 7 days (Azure) before the window; alarm if max |z| over sensors > 3;
  - *persistence* (Azure): observable state only, FAULT now → FAULT. The WARNING band is a label, not something a machine reports.
- **Best supervised model (Azure):** chosen on **validation only**, as the highest seed-mean validation PR-AUC (mean of WARNING and FAULT, operating windows, +24 h). **No setting is tuned on test results.**
- **Combined policy (Azure):** best supervised model OR autoencoder at the 99.9th percentile, per window, seed i with seed i.
- **Statistics:**
  - Azure: 95 % percentile bootstrap over machines (1000 resamples). Paired comparisons use the same resamples for both sides. Seed means of the per-seed estimates and bounds, with the number of seeds whose CI excludes zero.
  - MetroPT-3: per-failure results with honest caveats instead of CIs (four events), and the spread over seeds.

### Exploratory (definitions fixed now; never conclusions on their own)

- Calibration on Azure validation data with the Phase 8 code: top-label ECE, FAULT-class ECE and a positive-class ECE (WARNING + FAULT vs NORMAL, since FAULT is a single hour per event). Mondrian LAC coverage and set size per class at α = 0.1, with counts.
- Robustness on Azure test data with the Phase 8 perturbation functions: noise 10/20/30 %, MCAR 5/10/20/30 %, for the best supervised model, the autoencoder, the threshold rule and the z-score; paired against clean.
- The other lookbacks (2 h, 24 h) and percentiles (99.0, 99.5); every supervised model × both detectors as combined policies.
- MetroPT-3 false alarms per day by test month, to show whether false alarms grow as the compressor moves away from the February–March training period.
- **Error log (Azure):** an error-log rule (alarm if any error, or a given code, occurred in the window) and one XGBoost variant with per-hour error counts as extra inputs (`real_azure_err.yaml`, 3 seeds). Both are compared with telemetry-only models to show what the error log alone, and errors + telemetry, add.
- **Machine holdout (Azure):** 20 % of the machines (seed 2027, `real_azure_holdout.yaml`) held out entirely; the best supervised model (same validation rule, within the variant) and the autoencoder retrained on the other machines and tested on the held-out machines' test period, side by side with the temporal-split result on the same machines.
- Explanations with `explain()`: Azure by component (expected sensor derived from training data only); MetroPT-3 autoencoder per-sensor error shares before each failure, checked against the air-leak-plausible signals from the PDF semantics (pressures, motor current, COMP / MPG / DV electric, LPS).
- The comparison with the simulator results.

## Pre-registration (fixed 2026-11-17, before any model was trained on real data)

**Primary outcomes, Azure PdM** (horizon +24 h, lookback 48 h, test period, 95 % machine-level bootstrap CI, paired against the threshold baseline on the same machines):

- `detection_rate` (warned evaluable events / evaluable events), `lead_mean_h` (mean lead of the warned events, hours) and `false_alarms_per_day`, for:
  - the best supervised model (`best_supervised`, chosen on validation PR-AUC);
  - the `autoencoder` at the 99.9th percentile;
  - the `combined` policy (best supervised OR autoencoder at the 99.9th percentile), also paired against the best supervised model alone;
  - the baselines: `threshold` rule, rolling `zscore`, `persistence`.

**Primary outcomes, MetroPT-3** (lookback 6 h, test period, no CIs):

- per failure: `warned` (yes / no / not evaluable) and `lead_min` (minutes);
- `warned_count` out of the evaluable failures (evaluable: at least 50 % of the lookback has data);
- `false_alarms_per_day` over the normal test periods;
- `chance_expected`: the failures expected to be warned by chance alone at that false-alarm rate;

for the `autoencoder` and the `iforest` at the 99.9th percentile, the `threshold` rule and the rolling `zscore`. There is no supervised model and no combined policy for MetroPT-3.

**Everything else is exploratory.** The settings are mirrored in `config/models/real.yaml` (`primary:`) and guarded by `ml/tests/test_ml_real_prereg.py`.

## Consequences

- **Known limitations, stated in the report:**
  - MetroPT-3 has four failures: no statistical claim is possible. Unreported failures may exist in the training period. The failure times come from maintenance reports and are approximate. The time zone is not stated.
  - Azure is likely synthetic, so "real data" holds for MetroPT-3 only. Its failures are instantaneous labels without duration.
  - **Azure train and test contain the same machines (temporal split): the results show forecasting for known machines, not new ones.** The machine-holdout variant is exploratory.
  - The WARNING band is a label choice, not a machine signal, so leads are bounded by the lookback, and relative lead (vs a machine WARNING) does not exist.
- Results go to MLflow (`P10_real`); `experiments/results/real_data.md` is generated from those runs only.

## Outcome (2026-11-19)

All numbers come from the generated [real_data.md](../../experiments/results/real_data.md) (MLflow `P10_real`: 8 MetroPT-3, 24 Azure, 4 error-log and 18 holdout study runs, 45 combined-policy runs, 9 explanation runs). No setting was changed after any result was seen. Primary and exploratory outcomes are kept apart below.

### Primary, MetroPT-3 (lookback 6 h, detectors at p99.9; four failures, no CIs)

| Method | Warned / 4 (seeds) | Chance expected | False alarms / day |
|---|---|---|---|
| Autoencoder | 3 / 1 / 3 | 1.01 / 1.06 / 0.84 | 0.81 / 0.86 / 0.57 |
| Isolation Forest | 0 / 0 / 0 | 0.10 / 0.06 / 0.06 | 0.10 / 0.06 / 0.06 |
| Threshold rule | 2 | 0.61 | 0.30 |
| Rolling z-score | 2 | 0.98 | 0.95 |

- All four failures were evaluable at 6 h.
- **No method gives evidence of early warning beyond chance.** No warned count exceeds its chance expectation by more than 2.2 failures, and four events allow no statistical claim (as pre-registered).
- **The "warnings" are not precursors:**
  - *Failure #1* is "warned" with lead 360 min, the full lookback, by the autoencoder (3 seeds) and the threshold rule. The alarm was already on when the lookback began: the data resume after a gap about 15 h before the reported (whole-day) start with the score already above threshold, and the error shares in that period point at `pressure_switch`, not at an air-leak signal.
  - *Failures #2 and #3* are warned 2–8 min before the reported start. That is within the precision of report-based failure times, so it is detection at onset, not early warning.
  - *Failure #4* is warned only by the threshold rule (49 min).
- **The Isolation Forest warned of nothing** (0 / 4 in every seed), as on the simulator.

### Primary, Azure PdM (+24 h, lookback 48 h, 108 evaluable test events, 100 machines; machine bootstrap)

| Group | Detection rate | Lead (h) | False alarms / day | Warned (chance) |
|---|---|---|---|---|
| Best supervised (`lstm`, chosen on validation PR-AUC 0.160) | 0.954 [0.913, 0.987] | 41.2 [40.3, 41.9] | 0.059 [0.051, 0.066] | 103 (14.3) |
| Autoencoder (p99.9) | 0.000 [0.000, 0.000] | – | 0.001 [0.000, 0.002] | 0 (0.2) |
| Combined (`lstm` OR autoencoder) | 0.954 [0.913, 0.987] | 41.2 [40.3, 41.9] | 0.059 [0.052, 0.066] | 103 (14.4) |
| Threshold rule | 0.278 [0.200, 0.363] | 20.3 [15.6, 24.9] | 0.021 [0.017, 0.025] | 30 (5.0) |
| Rolling z-score | 0.009 [0.000, 0.031] | 33.0 | 0.001 [0.000, 0.001] | 1 (0.2) |
| Persistence | 0.000 | – | 0.000 | 0 (0.0) |

- **Best supervised vs threshold:** +0.676 [+0.584, +0.760] detection rate and +20.9 h [+16.3, +25.5] lead (3/3 seeds), at +0.038 [+0.031, +0.045] false alarms per day (3/3).
- **The autoencoder warned of no event.** On the simulator it caught every unseen-fault episode. Combined − best supervised: detection +0.000, false alarms +0.0007 per day (CI excludes 0 in 2/3 seeds). On Azure the OR adds false alarms only.
- **Threshold rule:** it beats the z-score (detection −0.269 for the z-score) and persistence (−0.278), CIs excluding 0.
- **Persistence warns of nothing,** because a machine never reports the WARNING band.

### Exploratory (never conclusions on their own)

- **Error log (Azure):**
  - The rule "any error in the 24 h window" reaches detection 1.000. Against telemetry-only XGBoost (0.978) that is +0.022 [0.000, +0.053], 0/3 seeds; the lead is −14.3 h shorter [−16.1, −12.5] and false alarms per day are +0.019 [+0.009, +0.028] higher (3/3).
  - Adding error counts to XGBoost raises validation PR-AUC from 0.151 to 0.644. Test detection does not improve (−0.028 [−0.066, +0.004]).
  - The telemetry models forecast earlier; the error log alone already flags almost every failure, a day ahead. This fits the audit (98 % of events have an error in the 24 h before) and the benchmark's likely synthetic origin.
- **Machine holdout (20 machines, Azure):** `lstm` again chosen on validation. On new machines it detects 0.942 against 0.971 for the temporal split on the same 23 events (lead 41.7 vs 42.2 h, false alarms per day 0.067 vs 0.062). The autoencoder detects 0.000 in both.
- **Other percentiles and lookbacks:**
  - MetroPT-3 autoencoder at p99 / p99.5 warns of 4 / 4 failures in every seed, but at 6.3 / 2.5 false alarms per day; its chance expectation at 6 h is 3.14 / 2.04.
  - Azure autoencoder at p99: detection 0.090.
  - Azure Isolation Forest: 0.139–0.281 across percentiles.
- **Detected during the failure (MetroPT-3):**
  - The autoencoder (every seed) and the threshold rule flag all four failures inside the reported interval, mostly within 0–12 min of the start. Autoencoder delays for #4: 110–172 min.
  - The z-score also flags all four (0–28 min).
  - The Isolation Forest flags 1–2.
  - The detectors recognise an air leak while it is happening, not before.
- **False alarms by month (MetroPT-3):** autoencoder 0.67 / 0.49 / 0.34 / 0.71 / 1.40 per day from April to August. Highest in August, the month farthest from the February–March training period. One trend over five months, not a drift test.
- **Robustness (Azure, `lstm`):**
  - Detection is unchanged under noise and MCAR (|Δ| ≤ 0.015).
  - False alarms per day rise with missing data: +0.010 (MCAR 5 %) to +0.108 (MCAR 30 %), 3/3 seeds, i.e. 0.059 → 0.166.
  - Noise 30 %: +0.009 (3/3).
- **Calibration (Azure, `lstm`, +24 h):**
  - Calibration on validation lowers top-label ECE from 0.013–0.022 to 0.003–0.016 and positive-class ECE from 0.047–0.060 to 0.037–0.050.
  - Mondrian LAC coverage at α = 0.1 is 0.898–0.900 (NORMAL), 0.910–0.916 (WARNING) and 0.889–0.935 (FAULT, n = 108). Mean FAULT set size is 1.96–2.00.
- **Explanations:** ADR 0026 § Phase 10 addendum. On Azure, top-sensor agreement with the training-derived expected sensor is 0.92–0.96 (chance 0.26). On MetroPT-3, the error shares during failures point at air-leak-plausible signals.
- **Simulator vs real:** the method ranking does not carry over (report § Simulator vs real data).
  - The autoencoder, best on the simulator's unseen fault, detects nothing on Azure and gives no warning beyond chance on MetroPT-3.
  - The threshold rule, near-perfect on the simulator, is weak on Azure (0.278).
  - The supervised models carry over where failures are plentiful and labelled (Azure).

### Answer to the research question, real-data part

- **Where many labelled failures of the same kinds exist (Azure, likely synthetic):**
  - A supervised model warns of most failures well beyond chance, about 41 h ahead, at about one false alarm per 17 machine-days.
  - Its confidence stays calibrated, and the conformal sets keep their coverage per class.
  - Missing data costs false alarms, not detections.
- **Where failures are rare and unlabelled (MetroPT-3, real):** the unsupervised detectors recognise failures *during* the event but give no early warning that can be told apart from chance. The one early "warning" (#1) is an anomaly already present when the data resumed after a gap.
- **On the twin:** simulator results on detector behaviour, especially the autoencoder's unseen-fault coverage, must not be assumed to hold on real machines.

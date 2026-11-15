# ADR 0022: Calibration, conformal prediction sets and time-to-fault intervals

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 8 (robustness, calibration and conformal prediction)

## Context

The research question asks whether the reported confidence stays trustworthy under noise, gaps, condition shift and unseen faults. The E1 models output probabilities trained with class weights, which push probability mass towards the rare classes on purpose. Before Phase 8 nothing was calibrated, and `Prediction.calibrated` was always false. The `Prediction` contract (CLAUDE.md §6) also needs a conformal set and a time-to-fault interval. Every choice below is fixed in advance or comes from validation data only. Test data never feeds back into any choice.

## Decision

### Where calibration is fitted, and on which windows

- **Data:** the clean validation split. It is never subsampled, so it carries the natural class prior.
- **Windows:** the operating windows only (machine RUNNING or WARNING now; `eval_subset: operating`). This is the population where an early warning matters, and the same population as the E1 main result. Windows already in FAULT or MAINTENANCE are trivially predicted, and including them would make every model look calibrated.
- **One map per horizon head.**

### Methods

| Family | Method | Notes |
|---|---|---|
| Sequence models (LSTM, GRU, Transformer) | Temperature scaling of the head's logits | One scalar T minimises the validation NLL (golden-section search over log T within [0.05, 20]). The argmax is unchanged. |
| Trees (RF, XGBoost) | One-vs-rest isotonic or one-vs-rest Platt (logistic on logit p_k, effectively unregularised), then rows renormalised | Both are fitted. The method per model and horizon is the one with the lower mean held-out NLL over 5 run-grouped folds of the validation split (folds never split a run). The choice is logged per run. |
| Baselines | None | Their one-hot outputs are reported as they are. |

### Decisions use the uncalibrated argmax

- Isotonic and Platt maps can change the argmax; temperature scaling cannot. So that the early-warning results stay comparable with E1, every alarm uses the **uncalibrated argmax** with the E1 sustain rule.
- Calibrated probabilities feed only:
  - the ECE;
  - the reliability diagrams;
  - the conformal sets.

### Calibration metrics

- **Top-label ECE**, 15 equal-width bins on [0, 1] over the operating windows; this is pre-registered as `ece_cal`. The FAULT-class ECE (`p(FAULT)` against the event "FAULT") is exploratory.
- **Per-run bin sums.** Each test run's bin sums (count, confidence sum, outcome sum) are stored, so the ECE is bootstrapped over runs like every other metric.
- **The ECE is small in absolute terms,** because most operating windows are confidently NORMAL. The FAULT-class ECE and the reliability diagrams show the rare-class behaviour.
- **Reliability diagrams** pool all seeds. They hide bins with fewer than 50 windows (`reliability_min_count`); the ECE uses every bin.

### Conformal prediction sets (split conformal, α = 0.1)

- **Calibration data:** the clean validation operating windows, using the calibrated probabilities.
- **Quantile:** the `ceil((n + 1)(1 − α))`-th smallest calibration score. If too few scores exist, the quantile is infinite and the set is every class.

| Method | Score | Set |
|---|---|---|
| LAC | `1 − p_y` | every class with `1 − p_k ≤ q` |
| APS, randomised (Romano et al. 2020) | probability mass ranked strictly above the true class + `u · p_y`, with `u ~ U(0, 1)` | every class whose (mass above + `u · p_k`) ≤ q |
| Mondrian LAC (class-conditional) | `1 − p_y` | one threshold per true class, from the validation windows of that class |

- **Why Mondrian:** LAC and APS guarantee coverage only on average over the classes, which means mostly over NORMAL windows. Mondrian LAC guarantees coverage per class, which matters most for WARNING and FAULT.
- **APS uniforms:** seeded per (condition, horizon), so every model sees the same draws. Randomised APS can return an empty set; it counts as size 0 and not covered.
- **Reported per condition:**
  - marginal coverage;
  - average set size;
  - coverage and set size per true class;
  - singleton and empty-set rates.
- **Pre-registered:** FAULT-class coverage for all three methods.

### Time-to-fault intervals

- **Method:** absolute-residual split conformal. The interval is `[ŷ − q, ŷ + q]`, clipped to `[0, cap]` with cap = 600 s.
- **Censoring** (ADR 0019): a row's target is right-censored when no fault was seen and less than the cap of the run remains. Only `true ≥ y` is known there, so coverage cannot be measured.
  - Censored rows never enter the calibration. Calibration uses rows with a known target below the cap, i.e. a FAULT onset within 600 s.
  - On test data, coverage and mean width are reported on the known rows below the cap.
  - On censored rows, the **certain-miss rate** is reported: the interval ends below the known lower bound (`hi < y`). Their mean width is reported too.
  - Rows with a known target equal to the cap (no fault within 600 s) are in neither group, because the regression target is capped there.

### Exchangeability

Split conformal assumes the calibration and test windows are exchangeable. Windows of one run are strongly dependent, so the guarantee holds at best approximately. The run is the exchangeable unit. Under the Phase 8 perturbations and the E4 shift, exchangeability is broken on purpose. Measuring how coverage degrades there is the point of the experiment.

## Consequences

- **Prediction contract:** `Prediction.calibrated`, `conformalSet` and `ttfInterval` can now be filled. Phase 11 (online inference) will use these maps. They are fitted per model run and are not yet exported as artifacts.
- **Cost of the uncalibrated alarm rule:** a well calibrated tree model could warn later or earlier if its calibrated argmax were used. That variant is not evaluated.
- **NN logits:** `SequenceModel.predict` now returns the logits in `Predictions.extra["logits"]`.

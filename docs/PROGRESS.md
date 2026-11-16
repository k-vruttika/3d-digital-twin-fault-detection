# LineTwin progress

## Phase 8: Robustness, calibration and conformal prediction (done, 2026-11-16)

- **Pre-registration recorded before any Phase 8 study run:** [ADR 0021](adr/0021-robustness-protocol.md) § Pre-registration, mirrored in `config/models/robustness.yaml` (`primary:`).
  - **Timestamp correction:** this line first said "11:30 UTC", which was a mistake. The file times are `robustness.yaml` 11:10 UTC and ADR 0021 11:19 UTC (local 16:40 / 16:49 +05:30).
  - **Smoke runs:** the first worker runs, from 11:21 UTC, went into a separate MLflow experiment (`P8_smoke`: persistence, then xgb and lstm seed 0, to check the pipeline). They are not part of any result.

### Tooling fix (2026-11-15): `poe lint` and pre-commit now check the same thing

- **Cause:** `[tool.ruff] extend-exclude` listed a bare `"data"`. Ruff treats a bare name as "every directory of that name", so `ruff check .` and `ruff format --check .` (in `poe lint`) silently skipped all of `ml/twin_ml/data/`. The pre-commit hooks pass file names explicitly, and ruff checks explicitly named files unless `force-exclude` is set. So the hook found the E501s and the unformatted code that `poe lint` never saw.
- **Fix:**
  - The excludes are anchored to the repo root (`./frontend`, `./data`, `./experiments/results`), and `force-exclude = true` is set.
  - `.pre-commit-config.yaml` is now one hook that runs `uv run --frozen poe lint`. The old hooks auto-fixed with `ruff --fix` and `ruff format`, skipped `tsc`, and ran ESLint only on frontend changes; the new hook does none of that. To fix findings, run `uv run poe fmt`.
  - Guarded by `tests/test_lint_config.py`.
- **Line endings:** the report generators (`report_robustness`, `report_e1`) write their `.md` / `.json` with LF on every OS. Before this, Windows `write_text` produced CRLF. The robustness e2e test asserts LF.
- **Checked:** `uv run poe lint` passes and `uv run pre-commit run --all-files` passes.

### Phase 8 summary

- **Built across Parts A–C:**
  - perturbations, the sensor-fault test sets, calibration (temperature scaling, isotonic / Platt) and conformal prediction (LAC, APS, Mondrian LAC, time-to-fault intervals);
  - run-level statistics and a resumable study runner;
  - seven studies: `e1models`, `e4`, `aug`, `e1models_snoise`, `aug_snoise`, and the follow-up `e1models_stdnoise`, `aug_stdnoise`;
  - one generated report: [robustness.md](../experiments/results/robustness.md) + `robustness.json` + 11 plots.
- **Decisions:** [ADR 0021](adr/0021-robustness-protocol.md) (protocol, pre-registration, amendment, outcome), [ADR 0022](adr/0022-calibration-conformal.md) (calibration and conformal, results), [ADR 0023](adr/0023-condition-shift-e4.md) (E4, results).
- **Answer to the research question, short version (details in the report):**
  - **Noise and missing values at the pre-registered levels:** early warning holds, and every learned model keeps its lead over the threshold rule.
  - **Calibration and coverage there:** calibration stays good, except the clean-trained RF under noise. Only Mondrian LAC gives a usable FAULT coverage, and it is below target under perturbation.
  - **Condition shift (E4):** warnings hold, but Mondrian FAULT coverage drops below target for every model.
  - **Stuck sensors and unseen drift:** these cost lead time and raise false alarms. Augmentation helps against stuck-sensor false alarms but not against drift.
  - **Noise relative to the fault signal:** at 10–25 % of the precursor rise, false alarms make the warnings useless, and calibration fitted on clean data no longer holds.
  - **Noise in normal-operation std (follow-up):** for most models, false alarms start rising between 0.3 × and 1 × the normal-operation std.

### Follow-up (2026-11-16): uniform noise in std units (exploratory)

- **What:** Gaussian noise with sigma = 1, 2, 4, 8 × each sensor's normal-operation std, the same multiple for every sensor. It is evaluation-only, for the E1 models and the augmentation-trained models ([ADR 0021](adr/0021-robustness-protocol.md) § Second amendment).
  - It uses the scale of the original `noise_<p>` grid (`noise_30` = 0.3 ×). The report puts both grids on one axis, so the onset can be located between them.
  - Report: § EXPLORATORY uniform noise in std units (onset table, value and Δ tables); plot `robustness_std_noise.png`, with false alarms on a symlog axis.
- **Code:**
  - `StdNoiseConfig` / `std_noise` in `robustness.yaml`;
  - `perturb.std_noise_conditions` (conditions `stdnoise_<m>`, kind noise, level m with the clean normal-operation std);
  - worker expansion and dispatch;
  - the report's signal-noise code generalised to a `NoiseGrid` used by both grids. The signal-noise values and section text were checked to be identical before and after the change.
- **Hashes:** the `std_noise` settings stay out of the shared hash, and enter an extension study's hash only if the study lists the grid. A `--dry-run --skip-existing` of all five earlier studies showed 0 runs to do.
- **Runs:**
  - `uv run poe robustness --study e1models_stdnoise --skip-existing`: 17 runs, 1,588 s (26 min).
  - `uv run poe robustness --study aug_stdnoise --skip-existing`: 17 runs, 1,809 s (30 min).
  - Run one after the other, no interruptions.

**Results** (+30 s, mean over seeds; onset = lowest level where the paired Δ false alarms per hour is positive with a CI excluding zero in at least 2 of 3 seeds, 1 of 1 for the threshold rule; the original 0.1–0.3 × points included)

| Model | E1 models: last level without a rise → first with a rise (FA/h there) | Augmentation-trained |
|---|---|---|
| threshold | 0.3 × → 1 × (0.27) | same rule, same values |
| RF | 2 × → 4 × (0.36) | 2 × → 4 × (0.66) |
| XGBoost | 0.3 × → 1 × (0.15) | 1 × → 2 × (2.76) |
| LSTM | 1 × → 2 × (5.45) | 1 × → 2 × (4.67) |
| GRU | 0.3 × → 1 × (0.80) | 1 × → 2 × (9.35) |
| Transformer | 0.3 × → 1 × (0.40) | 0.3 × → 1 × (0.42) |

- **Where false alarms start:**
  - For the threshold rule, XGBoost, GRU and Transformer (E1 models), false alarms first rise between 0.3 × and 1 × std; at 1 × they are still below 1 per hour.
  - The LSTM's rise at 1 × is significant in only 1 of 3 seeds.
  - The RF has no false alarms up to 2 ×. Its probabilities move, though: top-label ECE after calibration is 0.066 at 1 × and 0.161 at 2 ×.
  - From 2 × the false alarms grow quickly (E1 models at 2 ×: XGBoost 20.2, GRU 26.6, threshold rule 12.2, LSTM 5.5, Transformer 5.9 per hour). By 8 × they reach 63–244 per hour for every model except the RF (23).
- **Augmentation helps on this axis:** unlike the signal-relative grid, augmentation (noise up to 0.3 × in training) moves the onset one level later for XGBoost and GRU, from 1 × to 2 ×. It also lowers false alarms at 2 × for most models (augmentation vs E1, per hour):

  | Model | Augmentation-trained | E1 |
  |---|---|---|
  | XGBoost | 2.8 | 20.2 |
  | GRU | 9.4 | 26.6 |
  | Transformer | 2.2 | 5.9 |
  | LSTM | 4.7 | 5.5 |

  The augmentation-trained RF has slightly more false alarms at 4–8 × (0.66 / 36 vs 0.36 / 23).
- **Where this sits against the signal-relative grid:** 1 × std is about 1–3 % of the precursor rise on vibration, the leading sensor of most faults, but several times the rise on sensors that barely move before a fault (load). The signal-relative grid at 10 % puts up to 9 × std on vibration. That explains why it broke the recurrent models at once, while uniform noise of 1 × barely does.
- **Calibration and coverage:**
  - The FAULT-class ECE after calibration stays at or below about 0.03 up to 1 × for every model, then grows (E1 models: 0.10–0.63 at 4–8 ×).
  - Mondrian FAULT coverage of the E1 sequence models sits at 0.79–0.86 across 1–2 ×, below the 0.90 target. The trees' coverage goes to 1.00 because their sets grow.
- **Limitation:** the grid has no point between 0.3 × and 1 ×, so the onset for most models is located only to within that factor of 3.3. A 0.5 × level would narrow it, but it was not run because the requested grid was 1, 2, 4, 8.
- **Tests:**
  - `test_ml_signal_noise.py`: std-noise conditions and expansion; the grid keeps every earlier hash, the signal-noise studies included.
  - `test_ml_report_robustness.py`: both grid plots with partial studies; the onset rule across the base and new grids.
  - Totals: `uv run poe lint` clean; `uv run poe test` pytest 766 passed, 50 skipped (services not running), Vitest 163 passed; `uv run poe minimal-data-check` 14 passed.

### Part C (done, 2026-11-16): augmentation ablation, signal-relative noise, final report

**Runs**

| Study | Command | Runs | Wall time | Per run |
|---|---|---|---|---|
| `aug` | `uv run poe robustness --study aug --skip-existing` | 17 (5 models × 3 seeds + 2 baselines) | 14,403 s (4 h 00 min); the plan ETA said 143 min | RF 505–532 s, XGBoost 208–227 s, LSTM 825–1032 s, GRU 1083–1166 s, Transformer 1975–2105 s |
| `e1models_snoise` | `uv run poe robustness --study e1models_snoise --skip-existing` | 17 | 1,507 s (25 min) | RF 144–169 s (retrain), others 35–130 s |
| `aug_snoise` | `uv run poe robustness --study aug_snoise --skip-existing` | 17 | 1,805 s (30 min) | RF 243–251 s (retrain), others 34–131 s |

- **Study order:** the studies ran one after another, never in parallel, because every model uses all cores (`n_jobs: -1`, torch default threads). Running them in parallel would have slowed both and could change torch's floating-point summation order.
- **Clean runs:** no study was interrupted, and nothing needed `--skip-existing` to resume.
- **Datasets:** augmentation used `data/ml/fe3ac4753945ec7d` (280 of 560 training series perturbed: noise 81, MCAR 54, dropout 83, stuck 62; clean normaliser; val and test unchanged).

**Results: ablation, augmentation-trained minus clean-trained** (exploratory; paired, seed i vs seed i, +30 s; "significant" = the seed-averaged CI excludes zero and at least 2 of 3 seeds' CIs exclude zero; numbers in [robustness.md](../experiments/results/robustness.md) § ablation)

- **Clean data:**
  - No change in miss rate or false alarms.
  - Macro F1 is 0.007–0.018 lower (CIs include zero).
  - The RF's relative lead is +7 s.
  - The GRU and Transformer FAULT-class ECE is under 0.001 higher.
- **Seen types** (noise / MCAR / dropout / stuck grid and `sf_seen`, 16 conditions):
  - **Miss rate:** unchanged under every condition.
  - **Fewer false alarms where the clean-trained models suffered most:**
    - `stuck_30`: RF −0.16 per hour, XGBoost −0.20 per hour (3/3);
    - XGBoost `dropout_20` / `dropout_30`: −0.07 / −0.11;
    - Transformer `dropout_30`: −0.53 per hour (2/3).
    - No model has significantly more false alarms under any seen condition.
  - **RF calibration under noise is repaired:** ECE −0.020 at `noise_20` and −0.023 at `noise_30` (3/3). The ECE is better in 10 of 16 seen conditions. Its Mondrian FAULT coverage under noise falls by 0.08–0.09, away from the degenerate 1.00 of the clean-trained RF (sets that always contained FAULT).
  - **Stuck sensors still cost lead:** the augmentation-trained models lose 32–35 s of lead at `stuck_30` against their own clean result (3/3), about as much as the clean-trained models.
  - **Sequence models under MCAR get worse:**
    - Mondrian FAULT coverage: LSTM −0.13 / −0.16 at `mcar_20` / `mcar_30`, GRU −0.16 / −0.20, Transformer −0.09 to −0.15 at all MCAR levels.
    - Absolute coverage is 0.69–0.77 at `mcar_20` / `mcar_30`, below the 0.90 target.
    - LSTM lead: −14 / −16 s at `mcar_20` / `mcar_30`.
- **Unseen type** (`sf_drift`):
  - No significant change in early warning or calibration, apart from the RF's +7 s, which it also has on clean data, and the GRU's FAULT ECE (+0.0008).
  - False alarms inside drift segments are not reduced (point estimates, augmentation vs clean, per hour): RF 0.47 vs 0.22, XGBoost 0.94 vs 0.47, LSTM 0.22 vs 0.25, GRU 0.51 vs 0.47, Transformer 0.65 vs 0.58. The CIs are wide.
- **Mixed** (`sf_all`):
  - No significant change in early warning.
  - The one episode hidden by a 688 s all-sensor dropout is missed by all 15 augmentation-trained runs, as by the E1 models.

**Results: signal-relative noise** (exploratory; noise sigma = p % of each sensor's typical rise before FAULT; § signal-relative noise and `robustness_signal_noise.png`)

- **How big the noise is:** 10 % of the rise means 0.01 normal-operation std on some sensors (INSPECTION load) and 9.2 std on others (INSPECTION vibration). The original noise grid was at most about 5 % of the rise.
- **Early warning breaks down through false alarms, not misses.** E1 models, false alarms per hour:

| Level | Threshold | RF | XGBoost | LSTM | GRU | Transformer |
|---|---|---|---|---|---|---|
| clean | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| 10 % | 0.00 | 0.44 | 0.36 | 38.0 | 55.9 | 3.7 |
| 25 % | 1.6 | 37.1 | 34.7 | 141.2 | 159.1 | 26.8 |

  - **10 %:** macro F1 falls from 0.78–0.79 to 0.61–0.68.
  - **25 %:** macro F1 is 0.24–0.57.
  - **50–100 %:** the threshold rule reaches 221 false alarms per hour, every model's macro F1 is at most 0.32, the FAULT-class ECE reaches 0.98 (GRU), and the Mondrian sets grow to contain everything (coverage 1.00).
- **Miss rate stays 0** except for XGBoost (mean over seeds 0.04–0.09 of 26 episodes, CIs include zero). The lead grows towards the episode length because the alarm is already on. Neither counts as success: the report explains both, and the rising-edge false-alarm count falling at 100 % for the recurrent models.
- **Most robust at 10–25 %:** the threshold rule, then the Transformer and the trees. The recurrent models break first.
- **Augmentation does not help here** (`aug_snoise`, point estimates on the same perturbed runs; the report gives no paired test between the two studies). At 10 %, false alarms per hour: RF 4.6 (E1 RF 0.44), XGBoost 28.0 (0.36), LSTM 25.0 (38.0), GRU 61.4 (55.9), Transformer 3.3 (3.7). The augmentation noise was at most 30 % of the normal-operation std, far weaker than these levels on the vibration sensors.

**Changes in this part**

- **Report fix (`report_robustness.plot_snoise`):** the report crashed (`ValueError: number sections must be larger than 0`) when only persistence had finished in a signal-noise study, because persistence is never drawn and the legend had no entries. Now the legend is skipped when empty, and the dashed `aug_snoise` lines carry the labels when `e1models_snoise` is absent. Regression test: `ml/tests/test_ml_report_robustness.py::test_plot_snoise_partial_studies`.
- **Reading notes in the signal-noise section** (generated from the strength data):
  - the noise size in normal-operation std at the lowest level;
  - false alarms count rising edges;
  - lead is capped by the episode length.
  - These are methodology notes and do not change any metric.
- **Case-study wording:** "no model was trained on missing data" became "the E1 models were not trained on missing data". A generated line adds how many augmentation-trained runs (learned models only) detect the episode: 0 of 15.
- **Docs:**
  - `docs/experiments.md`: reproduce commands for all studies, the E4 / `snoise` conditions, the ablation definition, measured times, the results list;
  - ADR 0021 § Outcome; ADR 0022 § Results; ADR 0023 § Results.

**Tests:**
- `uv run poe test`: pytest 763 passed, 50 skipped (TimescaleDB, Mosquitto and Playwright tests; services not running). Vitest 163 passed.
- `uv run poe lint`: clean.
- `uv run poe minimal-data-check`: 14 passed.
- New test: `test_plot_snoise_partial_studies`.

**Known limitations (Phase 8)**

- **The ablation is exploratory and has wide CIs:** 26 episodes, 3 seeds. "Significant" uses the per-seed rule above, without correcting for multiple comparisons, across 16 seen conditions × 7 metrics × 5 models.
- **No paired test between `e1models_snoise` and `aug_snoise`:** the report puts them side by side only.
- **Augmentation levels follow the test grid,** which turned out to be weak against the fault signal. Training with signal-relative noise was not tried, so that the extension stays evaluation-only (ADR 0021 amendment).
- **The calibration and conformal maps are fitted per run and not exported.** Phase 11 has to refit or export them (ADR 0022).
- **Earlier Part A / B limitations still apply:** degenerate baseline conformal sets, approximate exchangeability, empty randomised APS sets.

**Deviations from the plan / CLAUDE.md**

- **Report generator changed after results were seen:** the plot bug fix, the reading notes and the case-study line. None of them changes a metric, a setting in `robustness.yaml` or the pre-registered primary outcomes.
- **The `aug` study took 4 h** against the 2.5 h + 1 h planned. Its plan ETA (143 min) used Phase 7 training times and left out the evaluation of 19 conditions.

### Part B (done, 2026-11-15): E4 condition shift

- **Dataset:** `uv run poe dataset-e4` → `data/ml/0b2beacae27b45d3`.

| Split | Runs | By band | Gradual episodes |
|---|---|---|---|
| train | 93 | low 53, medium 40 | 90 |
| val | 22 | low 7, medium 15 | 21 |
| `e4_high` | 62 | high 62 (none trained or validated on) | 62 |
| `e4_in` | 23 | low 12, medium 11 | 20 |

- **Study:** `uv run poe robustness --study e4 --skip-existing` trained and evaluated 17 runs in 8774 s (2 h 26 min), against a first estimate of 115 min.
  - Measured per run: RF about 150–180 s, XGBoost about 57 s, LSTM 460–650 s, GRU 570–770 s, Transformer 1195–2000 s.
  - The ETA came from Phase 7 train times scaled by the training-window ratio (0.678). It underestimated, because the evaluation on 878k + 326k test windows plus the training run's own test evaluation was not in it.
- **Code** (applied after the study finished, because the running workers import these modules):
  - the run filter (`data.build.select_runs`) and the extra split `test_in`;
  - augmentation (`augment_series`);
  - the signal-relative noise extension (`perturb.episode_rises`, `signal_scale`, `signal_noise_conditions`; extension studies).
- **Hashes:** the study config hash leaves extension studies and the `signal_noise` settings out of its shared part. All 34 e1models and E4 runs were checked against a recomputed hash before and after the patch: 34 match, 0 mismatch.
- **Report:** E4 in the PRIMARY section (`e4_high` vs threshold); the shift effect (unpaired) and the secondary E1-vs-E4 check on the 7 shared high-load runs (6 episodes) in EXPLORATORY; the plot `robustness_e4.png`.
- **Results (short record; numbers in [robustness.md](../experiments/results/robustness.md)):**
  - **Primary, `e4_high` (62 runs, 62 episodes, +30 s):**
    - No model misses an episode.
    - Lead relative to the threshold WARNING is still ahead of the threshold rule for every model (CI excludes zero in 3/3 seeds). The margin is much smaller for the trees (RF +24 s, XGBoost +14 s) than for the sequence models (+47 to +49 s).
    - False alarms per hour: the Transformer is slightly above the threshold rule (+0.03, 2/3 seeds), the others are not different.
    - ECE after calibration is about 0.003, except the RF at 0.029.
    - Mondrian FAULT coverage drops to 0.78–0.89, below the 0.90 target. LAC is about 0 and APS 0.59–0.68.
  - **Exploratory shift effect (same model, high minus low + medium, unpaired):**
    - The tree models lose lead: RF −28 s and XGBoost −43 s (CI excludes zero in 3/3 seeds). The sequence models lose 5–11 s (CIs include zero).
    - Mondrian FAULT coverage falls by about 0.09–0.12 for the sequence models.
    - XGBoost's ECE rises slightly (3/3), and the RF's clearly (+0.028).
  - **Secondary (E4 minus E1 on the 7 shared high-load runs, 6 episodes):** the E4 trees lose 29–40 s of lead, and the E4 sequence models lose 0.18–0.28 of Mondrian FAULT coverage. With 6 episodes the CIs are wide, and this comparison mixes the shift with the training-set size.
- **Tests:** `uv run poe test`: pytest 759 passed, 50 skipped (services not running). Vitest 163 passed. `uv run poe lint` clean. `minimal-data-check` 14 passed. New: `test_ml_study_datasets.py` (E1 fingerprint unchanged, E4 run rule, augmentation seeded and train-only, built quick datasets), `test_ml_signal_noise.py` (scale, conditions, hashes), `test_ml_report_robustness.py`.

### Report additions requested for Part C (implemented 2026-11-15, all exploratory, all generated)

1. **FAULT-class ECE next to top-label ECE** in every calibration table (primary, study, E4, ablation), with a note giving the NORMAL share of operating windows: 97.8 % in the e1models test set at +30 s.
2. **Case study of the episode missed under `sf_all`** by every learned model (all seeds) and the threshold rule. It is found from the per-episode results; the overlapping sensor faults and the leading-sensor rise are read from the simulator files; the explanation sentences are generated from conditions on that data.
3. **Perturbation strength relative to the fault signal:** per kind and gradual fault type, the leading sensor's precursor rise in normal-operation std, the noise sigma as a fraction of that rise, and the median precursor length against the dropout and stuck segment lengths.
4. **Drift note:** under `sf_drift`, false alarms inside drift segments are above zero for every learned model and the threshold rule. This suggests the models cannot tell sensor drift from degradation, which motivates Phase 9. The section is emitted only if the data shows this.
5. **Signal-relative noise grid** (exploratory, evaluation only; ADR 0021 amendment):
   - noise sigma = 10 / 25 / 50 / 100 % of each sensor's typical rise before FAULT, per kind, from the training episodes;
   - studies `e1models_snoise` and `aug_snoise`;
   - early warning, top-label and FAULT ECE, Mondrian FAULT coverage, and the plot `robustness_signal_noise.png`.

   The code patch waits until E4 finishes, because the running workers import those modules.
6. **Case-study note:** in the live system an all-sensor dropout is quarantined, so the machine goes STALE and then OFFLINE (thresholds read from `config/sources.yaml`) and raises the CRITICAL `machine_offline` alert. The data-quality layer covers this model blind spot.

### Part A (done, 2026-11-15): perturbations, E2 / E3 / sensor faults on the E1 models, calibration, conformal, report

**What was built**

- **Perturbations** (`twin_ml/data/perturb.py`, pure numpy):
  - noise, MCAR, single-sensor dropout and stuck-sensor segments on the raw series before windowing;
  - seeded per (condition, run, machine) with CRC-32 keys, identical for every model;
  - noise scaled by the normal-operation std per kind and sensor (RUNNING rows of the clean training split without an active fault).
- **Rebuilt test splits** (`twin_ml/data/test_sets.py`): the clean normaliser, vocabulary and config are reused, so forward fill and masks behave as in live use.
- **Simulator sensor-fault sets:** `uv run poe sim-sensor-faults` writes `data/sim_sf_{all,seen,drift}`, the 200 runs with seed 42 and scenarios that differ from `default.yaml` only in `sensor_faults`. The loader requires identical timestamps, states and fault types, and they are identical for all 30 test runs. About 10 % of the test rows are inside a sensor fault.
- **Model loading:** `TreeModel.load` (XGBoost heads; trained classes recomputed from the training split) and `SequenceModel.load`. Sequence models return their logits in `Predictions.extra`.
- **Calibration** (`twin_ml/calibrate/`): temperature scaling (NN); one-vs-rest isotonic or Platt, chosen by run-grouped CV NLL on validation (trees); ECE from per-run bin sums. [ADR 0022](adr/0022-calibration-conformal.md).
- **Conformal** (`twin_ml/conformal/`):
  - LAC, randomised APS and Mondrian LAC (requested in the plan review) at α = 0.1, with per-class sums;
  - time-to-fault absolute-residual intervals, with the certain-miss rate on censored rows.
- **Run-level statistics** (`twin_ml/evaluate/run_sums.py`): every metric is computed from per-run sums, so the bootstrap, the paired difference (same resampled runs) and the unpaired difference (independent resampling, for E4) all work the same way.
- **Study runner:**
  - `python -m twin_ml.experiments.robustness --study …` (poe `robustness`), one worker process per (study, model, seed) (`robust_worker.py`);
  - resumable through a study config hash (model config, robustness and evaluation settings, dataset key, reused E1 run);
  - progress JSONL with ETA in `experiments/progress/`.
  - E1 runs logged before the `model_config_sha` tag existed (baselines, RF, XGBoost) are matched by their logged `model.*` params, which must be exactly equal.
- **Report:** `python -m twin_ml.experiments.report_robustness` (poe `report-robustness`) reads only MLflow (`metrics.json` and `runstats.npz` per run). It writes [robustness.md](../experiments/results/robustness.md), a compact `robustness.json`, degradation curves per perturbation type, reliability diagrams and coverage plots. Sections are labelled PRIMARY or EXPLORATORY, and only runs whose hash matches the current configs are used.
- **Configs:** `config/models/robustness.yaml`, `config/scenarios/sf_{all,seen,drift}.yaml`; prepared for B and C: `dataset_e4.yaml`, `dataset_aug.yaml`, poe `dataset-e4` / `dataset-aug`.
- **Docs:** ADR 0021 (protocol and pre-registration), ADR 0022 (calibration and conformal), ADR 0023 (E4, draft), `docs/experiments.md` § Phase 8, CLAUDE.md §5.

**How to run**

```text
uv run poe sim-sensor-faults                            # once, about 30 s
uv run poe robustness --study e1models --skip-existing  # 17 study runs, 19 conditions each
uv run poe report-robustness                            # about 30 s
```

**Measured times (this laptop, CPU only)**

- **Study:** the e1models study took about 73 min for 17 runs: 3954 s after the resume, plus about 7 min before the interruption.
- **Per run:**
  - persistence about 70 s; threshold 40 s;
  - RF 300–370 s (retraining about 2 min, then evaluation);
  - XGBoost 110–130 s; LSTM and GRU 280–300 s; Transformer about 385 s.
- **Per condition:** a perturbed test split takes about 5 s (trees) to 15 s (NN) to build, predict and evaluate.

**Results (all numbers in [robustness.md](../experiments/results/robustness.md); short record only)**

- **Reused models:** every one reproduces its stored E1 clean test predictions exactly (max |Δp| 0.0), and the RF retrain reproduces E1 bit for bit.
- **Primary, `noise_20` / `mcar_20` / `sf_all`, +30 s:**
  - Every learned model keeps its earlier warning than the threshold baseline. Lead relative to the threshold WARNING improves by 53–67 s, and the CI excludes zero in 3/3 seeds for every model and condition.
  - Miss rate and false alarms per hour are not different from the threshold baseline (CIs include zero). Under `sf_all` every model and the threshold rule miss 1 of 26 episodes.
- **Calibration (primary):** ECE after calibration is about 0.001–0.003 for XGBoost and the sequence models under all three conditions. The RF is the exception under noise: its ECE is 0.021 at `noise_20`. Its probabilities move away from the validation-fitted map, and its marginal Mondrian coverage collapses (about 0.02), although its alarms are unaffected.
- **Conformal (primary), FAULT-class coverage at a target of 0.90:**
  - **Marginal LAC: 0.** The NORMAL-dominated threshold excludes FAULT.
  - **APS: about 0.50–0.62.**
  - **Mondrian LAC: 0.81–0.93** for the learned models. The exception is the RF under `noise_20` at 1.00, because its sets grow to include FAULT while NORMAL coverage collapses. The Transformer is lowest, under `mcar_20` and `sf_all`.
  - Only the class-conditional method gives a usable per-class guarantee, and even it falls below target under perturbation.
- **Exploratory:** stuck sensors hurt most.
  - At `stuck_30`, XGBoost loses about 0.10 macro F1 and 40 s of lead, with more false alarms (CI excludes zero in 3/3 seeds), and misses 2 of 26 episodes.
  - Noise and MCAR barely change anything.
  - Dropout raises the Transformer's false alarms (0.63 per hour at `dropout_30`).
  - Simulator drift (`sf_drift`) raises false alarms inside sensor-fault segments for every model and for the threshold rule.

**Tests:** `uv run poe test`: pytest 749 passed, 50 skipped (TimescaleDB, Mosquitto and Playwright tests; services not running). Vitest 163 passed. `uv run poe lint` clean. `uv run poe minimal-data-check` 14 passed. New test files:
- `test_ml_perturb.py` (also checks the pre-registration against ADR 0021);
- `test_ml_calibration.py`;
- `test_ml_conformal.py`;
- `test_ml_robustness_e2e.py`: E1 runs → study runs in workers → resume → report numbers equal the MLflow metrics; sensor-fault labels kept;
- import guards for the new modules.

**Known limitations (Part A)**

- **Baseline conformal sets are degenerate,** because the baselines output one-hot probabilities (noted in the report).
- **Exchangeability:** split conformal assumes exchangeable windows; windows within a run are dependent (ADR 0022).
- **APS** can return empty sets (randomised); they count as not covered.
- **26 gradual episodes** give wide CIs on miss rate and lead.
- **Interruption:** the e1models study was interrupted once by my own schema change mid-run. It resumed with `--skip-existing`; the hashes are unchanged and no run is duplicated in the report.

## Backlog (not scheduled)

- **Buffer telemetry:** a BufferLevel event (buffer id, level, capacity, ts) published by the simulator publishers, consumed by the adapters, stored as an event stream and sent on the WebSocket. Until then the 3D twin shows buffers as FULL / EMPTY / FLOWING / UNKNOWN, inferred from the neighbours' states (ADR 0016). The simulator already records `buffer_in` / `buffer_out`, but only as dataset label columns.

## Company deliverable: Machine Digital Twin Dashboard (Milestone A)

The original brief:

> Machine Digital Twin Dashboard: A 3D conveyor or CNC station on desktop shows running, idle and fault states driven by a JSON/CSV file or mock data server. It tests architecture, data handling and connecting the engine to outside systems.

| Part of the brief | Features that meet it | Where |
|---|---|---|
| **A 3D conveyor or CNC station** | Procedural 3D conveyor, CNC and inspection station with addressable parts, laid out from the line config. A single-station mode (one conveyor) is available, and serpentine lines up to 20 machines. | `frontend/src/engine/` (ADR 0016), `config/line.yaml`, `config/line_single.yaml` |
| **on desktop** | A browser dashboard (React + Three.js) at 1280 and 1920 px, light and dark, with basic keyboard support | `frontend/src/dashboard/` (ADR 0018), `docs/frontend.md` |
| **shows running, idle and fault states** | Every machine state (RUNNING, IDLE, WARNING, FAULT, STARVED / BLOCKED with root cause, MAINTENANCE, STALE, OFFLINE) in the 3D scene (motion, outlines, flashing, beacons, labels), in the machine list and in the details. Alerts follow ISA-18.2. | `engine/visuals.ts`, `engine/rootCause.ts`, dashboard panels |
| **driven by a JSON/CSV file** | The file source replays the simulator's CSV / JSONL exports (or any mapped CSV) into the twin. Minimal mode needs no Docker, broker or ML (tested in CI). | `uv run poe minimal`, `backend/twin_backend/adapters/file.py`, `tests/test_phase6_e2e_minimal.py` |
| **or mock data server** | The mock machine server (REST + WebSocket) drives the same twin live, accepts commands and scenario injection | `uv run poe dev --source mock`, `twin_sim mock-server` |
| **tests architecture** | Layers: adapters → ingest → event bus → event store / twin engine → REST + WebSocket API → frontend. Everything is swappable behind interfaces (file, mock, MQTT, OPC UA; SQLite or TimescaleDB). Decisions are recorded in ADRs 0001–0018. CI runs three OSes, with services and end-to-end tests on Ubuntu. | `docs/architecture.md`, `docs/adr/` |
| **data handling** | Validation and quarantine; dedup and reorder; missing values kept as SUSPECT and drawn as chart gaps (never zero); staleness; per-machine data quality; event store with history at raw / 1 min / 10 min; replay to any time; export | `backend/twin_backend/ingest/`, `storage/`, `/history`, `/replay`; dashboard history, sources and replay panels |
| **connecting the engine to outside systems** | Interchangeable live sources (mock REST/WS, MQTT, OPC UA) with hot switching. Typed REST and WebSocket API with API-key roles. Bidirectional commands back to the machine (safety rules, idempotency, audit), and scenario injection into the simulator. | `docs/api.md`, dashboard command, scenario and sources panels |

### How to run the demo

```text
uv run poe install                     # once: Python + frontend dependencies

# 1. Minimal mode: the twin from a CSV file, no Docker
uv run poe sim-quick                   # once: simulated runs -> data/sim_quick/
uv run poe minimal                     # terminal 1: backend on http://127.0.0.1:8000
uv run poe frontend                    # terminal 2: open http://127.0.0.1:5173/
#   Select a machine (list or 3D view): state, readings with units, live charts, history.
#   "Replay": scrub back through the stored run and watch states change in 3D.

# 2. Live mode: the mock machine server, commands and fault injection
uv run poe up                          # TimescaleDB + Mosquitto (Docker Compose)
#   add an operator key to .env, e.g. LINETWIN_API_KEYS_OPERATOR=demo:<a long random key>
uv run poe dev --source mock           # terminal 1
uv run poe frontend                    # terminal 2: open http://127.0.0.1:5173/, enter the key
#   Scenario (sim) tab: inject bearing_wear on CNC-01 -> WARNING, then FAULT and its alert.
#   STOP (confirm) -> neighbours BLOCKED / STARVED, line_stoppage alert -> Acknowledge.
#   Maintenance -> the machine restarts. Replay -> scrub back to watch the fault build up.

uv run poe screenshots                 # screenshots of all of the above (dashboard: real backend)
```

The same two flows run automatically as the Playwright end-to-end tests (`uv run poe browsers` once, then `uv run poe test`).

## Phase 7: ML dataset and models, E1 (done, 2026-11-14)

The first phase of Milestone B (research). It was delivered in two parts:

- **Part A:** setup, dataset, features, baselines, Random Forest and XGBoost, the evaluation and report pipeline.
- **Part B:** LSTM, GRU and Transformer, the full E1 run over three seeds, the final report and the docs.

The review changes are included: sample weights compensate the subsample, the report has bootstrap CIs, and `twin_sim` is isolated.

### What was built

- **Setup:**
  - This machine has no NVIDIA GPU (`nvidia-smi` is absent; Intel Arc graphics), so everything uses the CPU build of PyTorch.
  - New dependencies in `twin-ml`:
    - `torch` 2.15.0 (CPU index on Linux and Windows, PyPI on macOS);
    - `scikit-learn` 1.9.1;
    - XGBoost 3.2.0 (`xgboost-cpu` on Linux and Windows);
    - `mlflow-skinny` 3.17.2;
    - plus `twin-sim`, numpy, pyarrow and matplotlib.
  - Details and the reasons for each version: [ADR 0020](adr/0020-ml-models-training-tracking.md).
- **Simulator manifest:** a sha256 per data file and a `content_sha256` (the manifest without its volatile fields), so identical regenerations hash the same. The dataset build verifies the files against it.
- **Dataset** (`twin_ml.data`, `python -m twin_ml.data build`; [ADR 0019](adr/0019-ml-dataset-labels-splits.md)):
  - 30 s windows per machine series; strides train 5 s, val 5 s, test 1 s.
  - Targets: health at +10 s and +30 s; time to fault capped at 600 s with a censoring flag; fault type.
  - Split by run, 70 / 15 / 15 (140 / 30 / 30 runs).
  - Normaliser fitted on training rows only.
  - Forward fill plus missing masks; the engineered `vib_norm` channel.
  - Training-only subsample of far-normal windows (keep 0.25), with compensating sample weight 4.
  - Output: `data/ml/<dataset_key>/` (`meta.json` + one `.npz` per split).
- **Inputs:** `ALLOWED_INPUT_COLUMNS` (5 sensors) plus the kind is the only door into features. Sequence models see (30 × 14) windows; trees see 50 summary features.
- **Baselines:** persistence (reads `state` by design) and the static threshold rule (nominal per-kind limits from `simulator.yaml`). The simulator's own WARNING onset is the reference for relative lead times.
- **Models:**
  - RF and XGBoost (`models/trees.py`): one head per target; the time-to-fault regressor trains on known rows only.
  - LSTM, GRU and Transformer (`models/nn.py`): shared encoder, four heads, censoring-aware time-to-fault loss (`train/losses.py`), early stopping on validation macro F1.
  - Every classifier gets class weights on top of the sample weights.
- **Training and tracking:**
  - `python -m twin_ml.train` (config-driven, `config/models/*.yaml`) logs to MLflow (local file store `mlruns/`): tags with both manifest hashes, flattened params, about 200 metrics, and artifacts (`metrics.json` with per-episode lead times, `splits.json`, `test_predictions.npz`, XGBoost/torch models).
  - `twin_ml.experiments.e1` runs the whole of E1, with `--skip-existing` to resume.
- **Evaluation** (`twin_ml.evaluate`):
  - classification metrics per horizon and subset: (a) operating, (b) FAULT/MAINTENANCE, (c) before a gradual fault vs before a belt jam, and all windows;
  - time-to-fault MAE on known rows; fault-type macro F1;
  - early warning: lead time, lead relative to the simulator WARNING, miss rate, false alarms per hour;
  - latency p50/p95 at batch size 1 on one thread;
  - seeded bootstrap over whole test runs (1000 resamples, 95 % CI).
- **Report:** `python -m twin_ml.experiments.report_e1` reads only MLflow and writes [E1_clean.md](../experiments/results/E1_clean.md), `E1_clean.json` and `E1_clean_lead_times.png`, with mean ± std over seeds, the bootstrap CIs and the episode counts per machine kind.
- **poe tasks:** `dataset`, `train --model X [--seeds …]`, `train-all`, `report-e1`, `mlflow-ui`.
- **Docs:** [experiments.md](experiments.md) (new), ADRs 0019 and 0020, setup.md §7, CLAUDE.md §4 and §5.

### How to run

```text
uv run poe dataset                 # data/sim (200 x 1 h, default scenario, seed 42) + data/ml/<key>/
uv run poe train --model xgb       # one model, seeds 0,1,2 (or --seeds 0)
uv run poe train-all               # E1: baselines + 5 models x 3 seeds (--skip-existing to resume)
uv run poe report-e1               # experiments/results/E1_clean.md / .json / lead-time plot
uv run poe mlflow-ui               # optional: browse mlruns/ at http://127.0.0.1:5000
```

### Measured times (this laptop: Intel Core Ultra 5 125H, 18 logical CPUs, 16 GB RAM, Windows 11, CPU only)

| Step | Time |
|---|---|
| `poe dataset`: simulator, 200 runs × 3600 s | 11.8 s |
| `poe dataset`: windowed build (5.5 s inside the builder) | about 9 s |
| Full `train-all` (2 baselines + 15 trained runs, each with test evaluation, bootstrap and latency) | 74 min |

Train time per run (mean ± std over the three seeds, from `E1_clean.md`):

| Model | Train time |
|---|---|
| RF | 123 ± 3 s |
| XGBoost | 19 ± 1 s |
| LSTM | 228 ± 52 s (7–11 epochs at about 24 s) |
| GRU | 328 ± 26 s (5–9 epochs at 37–68 s) |
| Transformer | 674 ± 106 s (7–9 epochs at 76–87 s) |

- **Test evaluation per run:** 3–23 s of prediction over 424,920 test windows (XGBoost 3.4 s, Transformer about 20 s), plus the bootstrap.
- **Epoch-time variance:** GRU epoch times varied between runs (27–33 s in a first measurement run, 37–68 s inside `train-all`), with identical results: the seed-0 run reproduced its epochs and validation scores exactly. Wall-clock times on this laptop depend on load and are approximate.

### Results

Every number is in [E1_clean.md](../experiments/results/E1_clean.md), generated from MLflow; none is repeated here. In short, for the record:

- **Operating windows, +30 s:** all five models beat both baselines on macro F1. At +10 s the threshold rule is the strongest; the learned models rarely predict FAULT 10 s ahead.
- **Early warning:** every model detects all 26 gradual episodes and raises its sustained alarm before the simulator's own threshold WARNING, by a few tens of seconds on average (CIs in the report). There are almost no false alarms on clean data.
- **Belt jams** are not anticipated by any model or baseline, as expected: jams have no precursors.
- **Already faulted windows** (subset b): persistence is best by construction.

These are clean-data results. E2–E5 will stress them.

### Tests

pytest collects 745 tests (693 in Phase 6). With the services down, 695 passed and 50 skipped. 61 of the tests are in `ml/tests`; the extended guards and the manifest test come on top. The service-dependent tests did not run this phase because Docker was not up; nothing they cover changed.

- **Windows and labels** (`test_ml_labels_windows.py`):
  - window ends and strides; tail windows dropped; the +h target equals the state h rows ahead;
  - gathers stay inside the window; the §6 mapping;
  - censoring rule; episodes and episode types; forward fill with a leading gap.
- **Sample weights:**
  - the subsample keeps every fault-near window;
  - **the weighted class distribution after subsampling equals the unsubsampled one** (synthetic, and on the built dataset against a keep-1.0 build);
  - class weights from the subsample match the full set.
- **Leakage** (`test_ml_dataset_build.py`):
  - run-level split disjoint, complete and deterministic (140 / 30 / 30 for 200 runs);
  - no run in two splits; windows never cross series;
  - the normaliser equals a refit on training rows, and is unchanged when the val/test data is multiplied by 3.
- **Input restrictions:** permuting every forbidden column (state, run_time, seq, quality, fault_type, time_to_fault_s, ttf_censored, root_cause_machine, load_range, sensor_fault, buffers) in the Parquet files leaves the raw, normalised, mask, summary and sequence inputs bit-identical, while the labels change. Feature names are restricted.
- **Baselines** (`test_ml_baselines.py`): persistence on a hand-built series; the threshold rule at and just below every warning and fault limit for each kind; the limits equal the nominal config.
- **Metrics** (`test_ml_metrics.py`):
  - confusion matrix, P/R/F1 and macro F1 by hand; the macro over present classes;
  - average precision against scikit-learn's documented example (0.8333), ties and no positives;
  - MAE excluding censored and capped rows;
  - sustained alarms, rising edges, lead and relative lead, false alarms per hour, the summary;
  - bootstrap determinism, the CI brackets the point estimate, resampling of whole runs;
  - latency percentiles.
- **Losses** (`test_ml_losses.py`): censored rows cost nothing at or above the bound and the hinge below it; Huber on known rows; weighted means; focal loss against a hand computation.
- **End to end** (`test_ml_e2e_train.py`, quick-scenario data, 6 runs × 1500 s):
  - persistence, threshold, rf, xgb, lstm, gru and transformer each train, evaluate and log one MLflow run with the expected metric keys, tags (dataset key, content hash) and artifacts;
  - the report's numbers equal the MLflow metrics;
  - a report without runs fails clearly;
  - GRU training gives identical predictions for the same seed and different ones for another seed.
- **Imports** (`test_ml_imports.py`, fresh subprocess):
  - features, normalize, summary, dataset, labels, windows, models.base / trees, evaluate and persistence load neither `twin_sim` nor torch, sklearn, xgboost or mlflow;
  - only `data.build` and `baselines.threshold` load `twin_sim`;
  - only `models.nn` loads torch.
- **Guards:** every minimal-mode and backend import guard now also lists `mlflow` and `twin_ml` (`OPTIONAL_MODULES` too).
- **Simulator:** the file hashes match the files; `content_sha256` is equal for two generations with different paths and worker counts, and differs for another seed.

### Verified here (Windows 11, services not running)

| Check | Result |
|---|---|
| `uv run poe lint` | passes (ruff, format, mypy strict incl. `ml/tests`, eslint, tsc) |
| `uv run poe test` | pytest 695 passed, 50 skipped (TimescaleDB / Mosquitto / Playwright tests; services were not running); Vitest 163 passed |
| `uv run poe minimal-data-check` | 14 passed |
| `uv run poe report-e1` run twice | the outputs differ only in the generation timestamp |

### Known limitations

- **Clean data only:** noise, missing data, shifted conditions and unseen faults are E2–E5. The missing-mask channels are in place but all zero on this data.
- **Small episode count:** 26 gradual episodes in 30 test runs (CNC 12, conveyor 9, inspection 5). The bootstrap CIs on the lead times are correspondingly wide.
- **Noisy validation macro F1:** it moves by about ±0.05 between epochs, so early stopping with patience 3 often stops after 5–11 epochs. The best epoch's weights are kept. Not tuned further.
- **No calibration yet:** probabilities are uncalibrated (`Prediction.calibrated = false`). Calibration and conformal sets are the next phases.
- **Random Forests are not saved** (hundreds of MB); retrain from the seed.
- **MLflow file store:** in maintenance mode in MLflow 3.x and enabled with `MLFLOW_ALLOW_FILE_STORE=true`. `mlflow migrate-filestore` moves the runs to SQLite if needed.
- **Not run:** the CI matrix (there is no remote). Linux and macOS rely on the platform markers for torch and xgboost.

### Deviations from the plan / CLAUDE.md

- **`mlflow-skinny` instead of `mlflow`:** full mlflow 3.17 pins `pyarrow < 26`, which conflicts with the simulator. The full UI runs isolated via `uv run poe mlflow-ui`.
- **XGBoost 3.2.0, not the latest:** 3.3 and later need Python 3.12. `xgboost-cpu` is used on Linux and Windows to avoid the CUDA/NCCL dependency.
- **Subset (c)** (gradual vs jam) is restricted to operating windows, so it measures warning before the fault. A first version also counted windows already in FAULT, which made jam "recall" look high.
- **Simulator change:** `batch.py` writes file hashes and `content_sha256` into the manifest.
- **Test fixtures:** a new plugin `twin_ml.testing` is registered in the root `conftest.py`.
- **New files outside the CLAUDE.md layout** (§5 updated): `ml/twin_ml/baselines/`, `twin_ml/config.py`, `twin_ml/testing.py`, `config/models/dataset.yaml` and `eval.yaml`.
- **Results committed:** `experiments/results/E1_clean.md`, `.json` and `.png`.

## Phase 6: Dashboard (done, 2026-11-14)

Delivered in two parts:

- **Part A:** Phase 5 follow-ups, layout, auth, machine list and details, charts and history, alerts, connection state, code split.
- **Part B:** commands, scenarios, sources, replay, placeholders, end-to-end tests, screenshots, docs.

### What was built

- **Phase 5 follow-ups (engine):**
  - **Label declutter** (`engine/labelPolicy.ts`):
    - machine labels show only for non-normal states (anything but RUNNING / IDLE; NO DATA and LINK LOST count), buffer labels only for FULL / EMPTY / UNKNOWN;
    - hover or selection shows a label, and `setLabelMode("all")` (dashboard toggle) shows all of them;
    - labels keep a constant on-screen height (22 px, never below 14 px on small viewports), fade from 45 m and are hidden from 80 m; hovered and selected labels never fade.
  - **States scene:** rebuilt from short independent segments the simulator can produce: RUNNING → WARNING → RUNNING (a WARNING machine still runs), BLOCKED → FAULT → STARVED, BLOCKED → MAINTENANCE → STARVED, and STALE / OFFLINE alone.
    - Root causes come from `engine/rootCause.ts`, a port of the backend propagation. `uv run poe api-docs` writes `root_causes.json` (208 cases run through the backend), and a Vitest test checks the port against it.
  - **Serpentine connection:** where a row continues into the next (or an override puts machines out of line), a transfer track with flow chevrons connects them (`layout.ts` `linkPath`, `geometry/transfer.ts`). Buffers sit halfway along each link.
  - **Code split:**
    - the dashboard and the dev page load the engine with a dynamic `import()`;
    - three.js is its own chunk group, kept as two chunks along its module split, under the warning limit without raising it;
    - Chart.js is its own chunk;
    - the build has no chunk-size warning.
  - New engine API: `setLabelMode`, `setTheme`.
- **Backend additions** (small, all tested; ADR 0018):
  - `GET /auth/me`
  - `GET /commands/limits`
  - `GET /replay/range`
  - `/replay` returns `line` (root causes) and `alerts` (unresolved at t)
  - `GET /sources` has an `ingest` block (counters + latency)
  - `GET /scenario/options` (from `simulator.yaml`, checked against `twin_sim`)
  - `serve --cors-origin` and `devstack --cors-origin`
- **Dashboard** (`frontend/src/dashboard/`, `index.html`; details in [frontend.md](frontend.md#dashboard-phase-6)):
  - **State:** one `DashboardModel` (the stream's `EngineSink`) with small typed stores read through `useSyncExternalStore`. No state library (ADR 0018).
  - **Layout:**
    - header (connection pill, Replay, API key and role, label toggle, theme);
    - machine list (badges, root cause, DQ flag, alert count);
    - 3D view, with tabs below it for Live charts, History, Alerts, Sources, Scenario (sim), Prediction and What-if;
    - details and commands on the right.
    
    Responsive at 1280 and 1920 px; light and dark; skip link, listbox and tab keyboard navigation, focus-managed dialogs.
  - **Connection:** Connected / Reconnecting… / Link lost (after 5 s) / Key rejected, with a banner. Operator buttons are disabled while not connected.
  - **Auth:** key entry checked with `/auth/me`, the role shown, operator controls hidden for viewers. The key lives in memory or sessionStorage ("remember in this tab"), never in localStorage or a URL.
  - **Charts (Chart.js):**
    - live sensor charts for the selected machine (prefilled from `/history`);
    - history with automatic resolution: raw ≤ 30 min, 1 min ≤ 24 h, 10 min beyond, with a min–max band; presets, drag to zoom, reset;
    - missing values are gaps (`spanGaps: false`, null points at missing values and long time gaps), never zero;
    - units on the axes;
    - no threshold lines, because the backend provides none (stated in the UI).
  - **Alerts:** the ISA-18.2 groups, machine / severity filters, Acknowledge (operator), click to select.
  - **Commands:**
    - STOP always available, with a confirmation;
    - START / RESET / Maintenance / Set speed / Set load with the backend's limits;
    - a new idempotency key per click;
    - result PENDING → ACCEPTED → EXECUTED / REJECTED (reason) / TIMEOUT;
    - recent commands from the audit trail.
  - **Scenario:** only with the `scenario` capability, labelled SIMULATION ONLY. Fault types filtered by machine kind, optional progression time within the configured range, recent requests.
  - **Sources:** the active adapter, health, ingestion metrics (late %, quarantined by reason, partial, duplicates, latency mean / p95), per-machine data quality, and source switching (operator, with confirmation).
  - **Replay:**
    - a REPLAY bar with a scrubber over `/replay/range`;
    - the 3D scene, list, details and alerts jump to `/replay?t=` (engine replay mode);
    - play at 1× / 5× / 20× with coalesced requests;
    - one click back to live (resubscribe, fresh snapshot);
    - data quality shown as "not replayed".
  - **Prediction / What-if:** placeholders that turn the 501 into "available in Phase 11".
- **End-to-end tooling:**
  - **`twin_backend/e2e_stack.py`:** starts the backend and Vite on free ports and runs Playwright. In minimal mode it replays a fixed seeded file with `--no-loop`; in dev mode it starts the mock server itself and runs `devstack --no-sim` on a throw-away TimescaleDB database. It also provides `python -m twin_backend.e2e_stack screenshots`.
  - **Playwright:** `frontend/playwright.config.ts` (retries 0, trace and screenshot on failure) and `frontend/e2e/*.spec.ts`.
  - **Quiet scenario:** `config/scenarios/e2e_quiet.yaml` has no random faults, no auto-recovery and no speed changes.
- **poe and CI:**
  - new poe task `browsers` (Playwright Chromium, with system libraries on Linux);
  - `screenshots` also takes the dashboard screenshots;
  - new pytest marker `e2e`;
  - CI installs the browser and runs the e2e tests on Ubuntu only. Windows and macOS keep the unit tests and the build.
- **Docs:**
  - [ADR 0018](adr/0018-dashboard-state-auth-replay.md);
  - [frontend.md](frontend.md): dashboard section, labels, serpentine, bundle;
  - [api.md](api.md): dashboard endpoints, regenerated reference;
  - [architecture.md](architecture.md), [setup.md](setup.md), CLAUDE.md §4 and §5;
  - the Company deliverable section above.

### How to run

```text
uv run poe sim-quick; uv run poe minimal          # terminal 1 (no Docker)
uv run poe frontend                               # terminal 2 -> http://127.0.0.1:5173/ (dashboard), /dev.html (3D only)
uv run poe up; uv run poe dev --source mock       # live simulator instead (commands, scenario injection)
uv run poe browsers                               # once: Chromium for the e2e tests
uv run poe screenshots                            # dev page + dashboard PNGs (dashboard dev-mode shots need `up`)
```

### Tests

pytest runs 693 tests (683 before this phase) and Vitest 155 in 17 files (92 before). The most important new ones:

- **Backend** (`backend/tests/test_api_dashboard.py`):
  - `/auth/me` for each role and anonymous;
  - `/commands/limits` equals `commands.yaml`, with STOP never in a rule;
  - `/replay/range` equals the store's span;
  - `/replay` `line` and `alerts` (highest rev ≤ t, resolved alerts dropped, nothing before the first event);
  - the `/sources` ingest block matches `IngestMetrics`;
  - `/scenario/options` matches `twin_sim` (fault set, gradual faults, applicability per kind, progression range);
  - `--cors-origin`.
  
  The existing replay test now also checks that the replay's `line` equals the live engine's at the watermark. `root_causes.json` staleness is checked with the other generated files.
- **Engine:**
  - label policy (states, buffers, hover / selection / mode all), constant on-screen size (the projected height is checked), fading;
  - engine labels for non-normal machines only;
  - the root-cause port equals the backend on all 208 fixture cases;
  - the states scene is plausible: every state covered, no moving machine blamed, STARVED / BLOCKED neighbours stopped, the WARNING machine's neighbours running;
  - link paths: straight in a row, one transfer path per row turn clear of all machines, override turns, midpoint;
  - transfer tracks only at row turns, released on `dispose()`.
- **Dashboard components** (Vitest + Testing Library, jsdom):
  - role gating;
  - connection banner (connected → reconnecting → link lost → connected; key rejected), operator buttons disabled;
  - ISA-18.2 groups, filters, acknowledge, click to select;
  - details with units and missing values as "–";
  - missing values as chart gaps (null, `spanGaps: false`, axis unit);
  - command flow (pending → ACCEPTED → EXECUTED, REJECTED with reason, TIMEOUT, HTTP error), STOP confirmation, Escape cancels, a fresh idempotency key per click, limits enforced, disabled while disconnected;
  - scenario panel (capability, SIMULATION ONLY, faults filtered by kind, request body, viewers cannot inject);
  - sources metrics;
  - replay banner, engine replay mode, data quality not replayed, commands disabled, return to live resubscribes, play coalesces requests;
  - 501 placeholders;
  - session: the key in the header and the first WebSocket frame, never in a URL or localStorage, sessionStorage only when remembered, a rejected key opens no stream.
- **End to end (Playwright, retries 0):**
  - `tests/test_phase6_e2e_minimal.py`: dashboard on minimal mode, machines listed, details and charts, replay scrub changes the 3D labels (RUNNING → FAULT → STARVED by …);
  - `tests/test_phase6_e2e_dev.py` (services): inject → WARNING → FAULT + alert → STOP → BLOCKED / STARVED + `line_stoppage` → acknowledge → maintenance → RUNNING → replay back through WARNING and FAULT.

### Verified here (Windows 11, Docker services up)

| Check | Result |
|---|---|
| `uv run poe lint` | passes |
| `LINETWIN_REQUIRE_SERVICES=1 uv run poe test` | pytest 693 passed, 0 skipped (both e2e suites ran); Vitest 155 passed |
| `uv run poe build-frontend` | builds, no chunk-size warning (largest chunks: three 353 kB, main 280 kB, three 206 kB, Chart.js 167 kB) |
| `uv run poe minimal-data-check` | 14 passed |
| `uv run poe screenshots` | 12 dev-page PNGs + 11 dashboard PNGs (`frontend_screenshots/dashboard/`, with `summary.json`); no page errors; reviewed by eye (below) |
| e2e durations | minimal ≈ 55 s, dev ≈ 85 s |

**Screenshot review.** I looked at each PNG and fixed what was wrong:

- in replay, the live data quality (DEGRADED after the replay ended) showed as halos and DQ flags as if it were history; replay now clears it and says "not replayed";
- the command result was cut off at the bottom of the right column; it now sits under the buttons;
- the "likely rejected" hints took four lines; they are now one line, with the reasons in the tooltips;
- an EXECUTED result showed the ACCEPTED ack's reason; reasons now show only for REJECTED / TIMEOUT / errors;
- the history range reached back before the first stored event (an empty strip on the left);
- machine ids and times wrapped in the alerts table at 1280 px;
- the scenario progression placeholder was clipped;
- the replay speed was shown twice.

**Bug found by the e2e run:** devstack's mock-server child survived the test on Windows (terminating a process there runs no cleanup) and kept the mock port, so the next run talked to a stale server. The stack now starts the mock server itself.

### Follow-up fixes after review (2026-11-14)

- **Alert times:**
  - The alerts list showed each alert's latest transition time, which read like the raise time (a cleared WARNING showed the FAULT time). The alerts and details panels now show "raised", plus "cleared" and "acked … by …" when they exist. The times come from the alert's lifecycle events (`alertTimes`).
  - For alerts first seen at a later rev, the raise event is fetched from `/alerts/history`, once per alert.
  - Tests: unit tests for raise, clear, ack, re-raise, replay cut-off and an unknown raise; component tests for both panels, the ack time and the backfill request.
- **History chart fit:**
  - At 1280 px the x-axis labels were cut off. The bottom panel is now a flex column and the History chart fills the remaining height.
  - The minimal e2e and the screenshot specs (1280 and 1920) check that the chart canvas lies inside the panel.
- **Light-theme replay cue:**
  - The warm scene tint looked muddy. The light theme no longer tints the 3D scene; replay is shown by the banner plus an amber frame with a soft vignette on the 3D view, in both themes.
  - Tests check that the view gets `data-mode="replay"` and back, and that the light replay background equals the normal one.
- **Verified:**
  - `uv run poe lint` passes;
  - `LINETWIN_REQUIRE_SERVICES=1 uv run poe test`: pytest 693 passed, 0 skipped; Vitest 163 passed;
  - dashboard screenshots regenerated and reviewed.
  
  The first screenshot run of the dev-mode spec hit the 180 s Playwright test timeout inside `page.screenshot` (the spec normally takes 35–75 s). Two reruns passed, and the cause was not identified; a stalled software-WebGL frame under load is the likely one.

### Known limitations

- **Data quality is not replayable:** it is a rolling wall-clock window. Replay shows it as "not replayed".
- **No threshold lines:** the backend has no alarm limits to show. The simulator's thresholds are perturbed per run and not part of the twin's contract.
- **Replay plays by polling `/replay`:** at most one request in flight, a 500 ms tick. Fine for the stores used here; not measured on large stores.
- **Latency p95** is the upper bound of the histogram bucket, not an exact percentile.
- **Dev-mode e2e and screenshots use the mock server's fixed port** from `sources.yaml`, so they cannot run while `uv run poe dev` is running. The stack checks the port first.
- **Layout:** the dashboard targets desktop widths (minimum 1200 px). Narrow and mobile layouts are out of scope.
- **Accessibility** is basic (keyboard paths, labels, live regions). It was not audited with a screen reader.
- **Synthetic states overview** (dev page): with constant-size labels some labels overlap at the overview distance; the per-state close-ups are clean.
- **Not run:** the CI matrix (there is no remote). Linux and macOS behaviour relies on poe tasks, `subprocess.Popen` and `node` with no shell; only Windows was exercised.
- **Vitest drive-letter quirk** (Phase 5 note) still applies to this tool's shell; tests run from `C:\` paths.

### Deviations from the plan / CLAUDE.md

- **New dependencies (approved in planning):**
  - runtime: `chart.js` 4.6.1 (in the CLAUDE.md stack);
  - dev: `@testing-library/react` 16.3.3, `@testing-library/dom` 10.4.2, `jsdom` 30.1.3 (31.x was under two weeks old), `@playwright/test` 1.64.1.
  
  No zoom plugin.
- **API changes:**
  - new endpoints and fields (listed above);
  - the `/replay` response model is now `ReplayResponse`, a superset of `LineState`.
- **New CLI option:** `--cors-origin` on `serve` and `devstack`.
- **Engine behaviour:**
  - a snapshot that carries a `dataQuality` map is treated as complete (a missing machine has no report);
  - buffers now sit halfway along the link between outlet and inlet, instead of between machine centres.
- **"Restart" in the dev e2e** is Maintenance. RESET keeps the wear, so the machine would fault again. Maintenance clears the fault and wear, and the simulator runs the machine again afterwards.
- **Code split:** three.js is split into chunks along its own module boundary (`maxSize` on the chunk group). The warning limit was not raised.
- **Refactors:**
  - the API test helpers (`running_api`, `api_client`, keys) moved from `test_api.py` into `twin_backend/testing.py`, so the new test module can use them (pytest's importlib mode);
  - `twin_common.config.loader._read_yaml` became `read_yaml`.
- **New files outside the CLAUDE.md layout** (§5 updated):
  - frontend: `playwright.config.ts`, `tsconfig.e2e.json`, `e2e/`;
  - backend: `twin_backend/e2e_stack.py`, `api/routes/auth.py`;
  - common: `twin_common/config/scenario_options.py`;
  - config: `config/scenarios/e2e_quiet.yaml`.

## Phase 5: 3D twin engine, frontend (done, 2026-11-14)

Delivered in two parts:

- **Part A:** API client, `GET /line`, layout, geometry, state visuals, dev page.
- **Part B:** interaction, performance, highlight API, dispose, screenshots, docs.

Picking, the tooltip and the FPS counter were written with the engine in part A; part B added their tests and refinements.

### What was built

- **Backend:**
  - **`GET /line`** (`api/routes/line.py`, viewer) returns the loaded line config: machines (kind, name, sensors, parts, `sensorParts`, `layout`), buffers, the sensor catalog, and `topology` from `Topology.from_line`, the same one the engine uses for root causes.
  - **`MachineLayout`** (`twin_common.config.line`): an optional per-machine `layout: {position: [x, y, z], rotation_deg}` in the line YAML. The simulator ignores it.
  - **WebSocket frames as Pydantic models** (`api/ws_models.py`, [ADR 0017](adr/0017-generated-api-types.md)): `WsClientMessage` / `WsServerMessage` discriminated unions. The hub builds its control frames through them, and they are merged into the OpenAPI components.
  - **`api/docs.py`** also writes `frontend/src/api/openapi.json` and the `GET /line` fixtures for both shipped configs (`frontend/src/engine/__fixtures__/`).
  - **`serve --loop/--no-loop`:** overrides the file source's `loop` (the shipped `sources.yaml` loops), so a replay can end and the twin shows the machines going STALE and OFFLINE.
- **Frontend engine** (`frontend/src/engine/`, [ADR 0016](adr/0016-3d-twin-engine.md)): **`TwinEngine`**, Three.js used directly, no React.
  - **Pure modules:** `topology.ts` (flow order, branch, merge and cycle handling), `layout.ts` (serpentine auto layout plus overrides), `visuals.ts` (state → visual, labels, buffer inference, animation rate) and `tooltip.ts`.
  - **Procedural geometry** (`geometry/`):
    - conveyor: frame, instanced rollers, scrolling `DataTexture` belt, drums, motor;
    - CNC: bed, column, rotating spindle, glass enclosure;
    - inspection: rotating indexer, gantry with a sweeping camera head.
    
    Every config part is a named group with its own materials; unknown parts get a generic housing.
  - **State visuals:**
    - RUNNING animated at speed / reference RPM, IDLE stopped;
    - WARNING: orange inverted-hull outline on the at-risk part (top attribution, else the whole machine);
    - FAULT: red flash;
    - STARVED / BLOCKED: yellow beacon; MAINTENANCE: blue beacon and tint;
    - STALE semi-transparent, OFFLINE grey, link lost semi-transparent;
    - data quality DEGRADED: a floor halo; active alerts: a severity marker;
    - camera-facing text labels for every state ("BLOCKED by CNC-01"), turned off with `labels.enabled`.
  - **Buffers:** inferred FULL / EMPTY / FLOWING / UNKNOWN, never a made-up count. Parts on conveyors move with the belt and are hidden on STARVED, STALE or OFFLINE conveyors.
  - **Interaction:** orbit controls, click to select (a drag does not count), hover tooltip in plain DOM (text only). One invisible pick box per machine and buffer.
  - **Public API:** `loadLayout`, `applySnapshot`, `applyReading`, `applyStateChange`, `setPrediction`, `highlightParts`, `setMode`, `on('select' | 'hover')`, `resize`, `dispose`. Added: `applyDataQuality`, `applyAlert`, `applyLine`, `setConnected`, plus diagnostics (`resourceCounts`, `pickAt`, `screenPosition`, `focus`).
  - **Disposal and FPS:** `ResourceTracker` and `ListenerRegistry` make `dispose()` checkable. `FpsCounter` / `FpsMeter` provide the dev FPS overlay.
- **API client** (`frontend/src/api/`):
  - typed `RestClient`;
  - `StreamClient`: auth first frame, subscribe after the auth reply, snapshot. It reconnects with seeded exponential backoff after any close except 4401 (4408 included), re-subscribes and delivers a fresh snapshot, and has an optional heartbeat;
  - `bindStream`: re-fetches `/line` on reconnect and reloads the scene only if it changed; replay mode ignores live data.
- **Generated types:** `openapi.json` → `schema.gen.ts` (openapi-typescript), aliased in `types.ts`.
- **Dev page** (`frontend/dev.html`, `src/dev/`):
  - full-screen engine against `VITE_LINETWIN_API_URL`, with an optional `VITE_LINETWIN_API_KEY` from `frontend/.env.local`;
  - status pill; a clear message when the backend needs a key;
  - `?fps`; `?synthetic=20|line|single|states`; `&freeze=<s>`.
- **poe tasks:**
  - `frontend`: the Vite dev server;
  - `screenshots`: Playwright Chromium install plus `npm run screenshots`;
  - `api-docs` now also regenerates `openapi.json`, the fixtures and `schema.gen.ts`;
  - `minimal-data-check` also runs `tests/test_phase5_frontend_contract.py`.
- **Docs:**
  - new `docs/frontend.md`;
  - ADRs 0016 and 0017;
  - `setup.md` §5 (backend and frontend together);
  - `api.md` (`GET /line`, WS schema note, regenerated reference);
  - `architecture.md`;
  - CLAUDE.md §4 and §5.

### How to run

```text
uv run poe sim-quick; uv run poe minimal          # terminal 1 (add --no-loop to let the replay end)
uv run poe frontend                               # terminal 2 -> http://127.0.0.1:5173/dev.html
uv run poe up; uv run poe dev --source mock       # live simulator instead (commands work)
uv run poe screenshots                            # PNGs -> experiments/results/frontend_screenshots/
```

### Tests

pytest runs 683 tests (671 before this phase) and Vitest 92 in 10 files (none before). The most important new ones:

- **Frontend contract in minimal mode** (`tests/test_phase5_frontend_contract.py`, fresh process, `--no-loop`):
  - `GET /line` equals the committed fixture;
  - every WebSocket frame validates against `WsServerMessage`;
  - after the replay ends, every machine goes STALE (3 s) and then OFFLINE (5 s after its last reading), within 2.5 s of slack.
- **`GET /line`** for `line.yaml` and `line_single.yaml`; an added machine with a layout override appears with its topology; `MachineLayout` validation. All frames in the existing WebSocket tests now validate against the schema.
- **Staleness:** `openapi.json` and the fixtures match the code (pytest); `schema.gen.ts` equals openapi-typescript's output (Vitest); every REST client endpoint exists in the schema.
- **Pure functions:**
  - layout: flow order, gaps, single station, added machine, overrides, 20 machines in a serpentine without overlaps, unknown kind;
  - topology: reordered config, cycle fallback, branch / merge warnings;
  - visuals: one test per state; DQ and alert overlays; link lost; labels including "BLOCKED by CNC-01", "(unknown)" and turned off; attribution vs whole machine; animation rate; buffer status for 10 neighbour combinations;
  - tooltip: units, `–` for null, absent and NaN values.
- **WebSocket client** (fake socket, fake timers):
  - auth is the first frame and subscribe waits for the auth reply;
  - backoff 100 / 200 / 400 / 800 ms, then success with re-auth, re-subscribe with the changed selection and a fresh snapshot (`reconnect: true`);
  - 4408 reconnects, 4401 does not;
  - backoff reset; heartbeat timeout; `close()` leaves no timers.
- **Engine on a fake renderer:**
  - every config part of both fixtures is an addressable group with its own material;
  - readings and StateChanges map onto materials (OFFLINE desaturated, STALE transparent), the beacon and labels;
  - attributed WARNING outline; FAULT flash; DQ halo; alert marker cleared on resolve; buffers FULL / EMPTY / FLOWING / UNKNOWN; items hidden while the link is down;
  - picking of machines and buffers; click vs drag; one hover event per target with a tooltip; `highlightParts` (clamping, unknown parts, FAULT precedence); `setMode`.
- **`dispose()`:**
  - every geometry, material and texture reachable from the scene receives its dispose event;
  - every resource and listener count is 0, and no frame is pending;
  - renderer, controls and context are released, and the canvas is removed;
  - it is re-entrant, and repeated `loadLayout` does not grow the counts.

### Verified here (Windows 11, Docker services up)

| Check | Result |
|---|---|
| `uv run poe lint` | passes |
| `LINETWIN_REQUIRE_SERVICES=1 uv run poe test` | pytest 683 passed, 0 skipped; Vitest 92 passed |
| `uv run poe build-frontend` | builds (chunk-size warning, see limitations) |
| `uv run poe minimal-data-check` | 14 passed (incl. the Phase 5 contract test) |
| `uv run poe screenshots` | 12 PNGs, WebGL via SwiftShader, no page errors; reviewed by eye (below) |
| Dev page live against `serve --mode minimal --no-loop` (60 s file at 2x), headless Chromium | all machines RUNNING; at ~31 s all STALE (semi-transparent), ~2 s later all OFFLINE (grey, offline alert markers, DQ halos, buffers UNKNOWN); no page errors |
| Dev page live against `devstack --source mock` (TimescaleDB), STOP CNC-01 via `POST /commands` with a temporary operator key | ack EXECUTED; CNC-01 IDLE at once, then `BLOCKED by CNC-01` on CNV-01 (yellow beacon, buffer FULL) and `STARVED by CNC-01` on INSP-01 and CNV-02 (buffers EMPTY), all within ~19 s; START restored the line |

**Screenshot review.** I looked at each PNG and fixed what was wrong:

- the CNC roof was an opaque slab hiding the spindle; it is now glass with a frame;
- labels were hidden behind enclosure posts; they now draw on top;
- the close-ups cropped labels;
- the overview cut off the 20-machine line; the camera distance is now fitted to both fields of view;
- the belts were too dark;
- the inspection station could not be hovered in its open centre; found by the picking test, fixed with pick boxes.

The final images show every state with its label, the WARNING outline on the spindle head, the red FAULT spindle, beacons, the DQ halo, alert markers, all four buffer states, the single-station line, the 20-machine serpentine and the tooltip with units.

### Known limitations

- **FPS (measured by the user on their GPU after this phase):** `dev.html?synthetic=20&fps` runs at 60 fps with a worst frame of 21 ms. Headless Chromium here uses software rendering, so it was not measured in this environment.
- **Buffers are qualitative** (see the backlog); a FLOWING buffer may be nearly full or nearly empty.
- **Reading age** in the tooltip uses the wall clock, which matches event time only for live sources (a replay at speed ≠ 1 or an old export shows large ages).
- **Replay mode** was only a visual cue plus the live-data gate in `bindStream`; the dashboard (Phase 6) now drives the engine from `/replay?t=`.
- **Bundle size:** one chunk of about 600 kB (three.js); Vite warned. Fixed in Phase 6 (code split, no warning).
- **Vitest under this tool's shell:** Vitest fails on every test file, even a trivial one, when it is started with a lower-case drive letter in the working directory (`c:\dev\...`), which is how this session's shell starts; with `C:\` (any normal terminal, PowerShell, CI) it passes. It looks like two Vitest module instances when the path case differs. Not a project issue; noted in case it shows up elsewhere.
- **Not run:** the CI matrix (there is no remote).

### Deviations from the plan / CLAUDE.md

- **New dependencies:**
  - `three` 0.187.0 and `@types/three` (in the CLAUDE.md stack, newly installed);
  - dev: `openapi-typescript` 7.13.0 (approved in planning; npm `overrides` lets it use TypeScript 6) and `playwright` 1.64.1 (approved; 1.65.0 was under two weeks old).
- **Contract and config:** `MachineConfig.layout` (optional); the WebSocket frame models are new types in the OpenAPI schema, not in CLAUDE.md §6, since they are API envelopes, not domain contracts.
- **New CLI option** `serve --loop/--no-loop`, needed to end a replay in minimal mode (`sources.yaml` loops by default).
- **New files outside the CLAUDE.md layout** (§5 updated): `frontend/dev.html`, `frontend/scripts/`, `frontend/src/dev/`, `frontend/src/test/`, `backend/twin_backend/api/ws_models.py`, `api/routes/line.py`.
- **Engine API additions:** `applyDataQuality`, `applyAlert`, `applyLine`, `setConnected` and diagnostics. Picking uses invisible pick boxes rather than the meshes.
- **WebSocket subscribe validation:** unknown channels and machines are now checked on the raw message before typed validation. The error frame is unchanged.
- **Synthetic mode** (`?synthetic=`) generates data in the browser for the performance check and screenshots only; it never reaches the backend or any experiment.

## Phase 4: Twin engine, API and bidirectional commands (done, 2026-11-13)

Delivered in two parts: part A (engine, alerts, API, WebSocket stream, auth, minimal mode) and part B (commands end to end over all adapters, scenario injection, dev launcher, docs). A review fix between the parts makes restored alerts wait for fresh data after a restart.

### What was built

- **Twin engine** (`backend/twin_backend/engine/`, [ADR 0012](adr/0012-twin-engine.md)). The engine observes the `LineState` fold and never changes it, so live state still equals `reconstruct(t)`.
  - `state_machine.py`: allowed transitions for all nine states from `config/engine.yaml`. Invalid transitions are applied anyway, logged and counted (`linetwin_engine_invalid_transitions_total`).
  - `topology.py` + `propagation.py`: root cause of every STARVED/BLOCKED machine from the line config. Buffers define the links; without buffers the machine order does (`line_single.yaml`). A neighbour whose link is down is reported with `unknown: true`.
  - `data_quality.py`: OK / DEGRADED per machine from late %, quarantine %, partial-reading % (rolling wall-clock window) and link state, with `min_samples` and recovery hysteresis from config. Quarantined messages reach it on a new bus topic `quarantine`.
  - `twin.py`: when a reading changes a machine's state, the engine publishes a StateChange with a reason (`"BLOCKED by CNC-01"`). Pipeline StateChanges are not re-emitted.
- **Alerts** (`engine/alerts.py`, [ADR 0011](adr/0011-alert-lifecycle.md)):
  - an ISA-18.2 lifecycle with independent `active` / `acknowledged`, plus `ackedBy`, `ackedAt` and `rev`;
  - every transition is an append-only Alert event; deterministic ids;
  - kinds `machine_warning`, `machine_fault`, `machine_offline`, `data_quality` and `line_stoppage` (event-time debounce);
  - restored from the store on startup. A restored alert is re-evaluated only once its machine has fresh data since startup (the whole line for `line_stoppage`, `min_samples` messages for `data_quality`).
- **API** (`backend/twin_backend/api/`, [ADR 0013](adr/0013-api-auth-websocket.md)):
  - FastAPI, started with `uvicorn.Server.serve()` inside `twin_common.aio.run()`, never `uvicorn.run`. `python -m twin_backend serve --mode minimal|dev|config`.
  - Routes: `GET /machines`, `/machines/{id}/state`, `/history` (raw / 1s / 1m / 10m), `/replay?t=`, `/export`, `POST /commands`, `GET /commands/{id}`, `GET /commands`, `GET /alerts`, `/alerts/history`, `POST /alerts/{id}/ack`, `POST /scenario`, `GET /scenario[/{id}]`, `GET /sources`, `POST /sources/switch`, `GET /predict` and `POST /whatif` (501 stubs), `GET /health`, `GET /metrics`.
  - `WS /stream`: first-message auth, subscriptions by machines and channels, a snapshot on subscribe, per-client outboxes with coalescing (readings, line view, data quality) and drop-oldest with `dropped` notices (state changes, alerts, commands). A client whose send takes longer than `send_timeout_s` is closed (4408).
  - CORS for the Vite dev server, configurable in `config/api.yaml`.
- **Auth** (`api/auth.py`):
  - API keys from `.env` (`LINETWIN_API_KEYS_OPERATOR` / `_VIEWER`, `name:key,...`; empty placeholders in `.env.example`), mapped to roles viewer < operator.
  - Without keys: read-only and loopback only. A non-loopback bind without keys refuses to start, and a warning is logged.
  - WebSocket keys go in the first message, never in a URL.
  - `twin_common.logging` redacts every configured key, credential-named fields and `apiKey=` query parameters from structlog and uvicorn logs.
- **Commands** (`backend/twin_backend/commands/`, [ADR 0014](adr/0014-command-path.md)):
  - Pydantic validation; idempotency keys with replay (`Idempotent-Replay` header), in-flight dedup, 409 on reuse with a different request, a TTL, and reload from the store.
  - Safety rules and limits from `config/commands.yaml`. **STOP is always allowed** (fail-safe): exempt in code, and the loader rejects any rule that names it.
  - Capability check with a clear reason for the file source; routing via `SourceManager.route_command` with the existing timeout.
  - Audit trail: Command, ACCEPTED and the final ack on the bus and in the store, rejected commands included, `ts` on the event-time axis.
- **Scenario injection, simulation only** ([ADR 0015](adr/0015-scenario-injection.md)):
  - `Capability.SCENARIO` on the mock server (`POST /scenario`), MQTT (`{prefix}/scenario` + `/scenario/ack`) and OPC UA (`InjectScenario` method). The mock, MQTT and OPC UA adapters implement `send_scenario`.
  - Actions `INJECT_FAULT` (`LineSimulator.inject_fault`) and `LOAD_SCENARIO` (`LineSimulator.schedule_plan`), run at a tick boundary (`RealtimeRunner.submit_call`).
  - New contracts `ScenarioRequest` / `ScenarioAck`, stored in the new table `scenario_requests` (SQLite 0003, TimescaleDB 0005), rejected requests included. Metric `linetwin_scenario_requests_total`.
- **Storage:** alert lifecycle migrations (SQLite 0002, TimescaleDB 0004, unique `(alert_id, rev, ts)`). The TimescaleDB settings fingerprint now includes the stream list, so new tables get the configured policies.
- **Runtime:** `build_runtime` also builds the engine, the command and scenario services, and loads `engine.yaml`, `api.yaml` and `commands.yaml`. The line file comes from `TwinSettings.line_file` (`LINETWIN_LINE_FILE`, `--line`). `SourceManager.stop(before_storage=...)` lets the engine publish its last events before the writer drains.
- **poe tasks:** `minimal` (`serve --mode minimal`), `dev` (`python -m twin_backend.devstack`) and `api-docs`. `minimal-data-check` also runs `tests/test_phase4_minimal.py`.
- **Dev launcher** (`devstack.py`):
  - probes TimescaleDB (and Mosquitto for MQTT) and prints the `uv run poe up` hint;
  - starts the matching `python -m twin_sim mock-server | mqtt | opcua` with `subprocess.Popen`: no shell, no asyncio subprocess;
  - waits until it is ready, runs the backend in process, and stops the child in `finally`.
- **Docs:**
  - `docs/api.md` has hand-written sections for running, auth, the WebSocket protocol, commands, alerts and scenario injection, plus a REST reference generated from OpenAPI (`uv run poe api-docs`; a test fails when it is stale).
  - ADRs 0011–0015; updates to architecture.md, setup.md, simulator.md and CLAUDE.md (§5 config files, §6 Alert, scenario contracts and topics).

### How to run

```text
uv run poe sim-quick; uv run poe minimal      # API on http://127.0.0.1:8000, file replay + SQLite
uv run poe up; uv run poe dev                 # simulator publisher (sources.yaml active) + TimescaleDB
uv run poe dev --source opcua --speed 5       # other source / speed
uv run poe api-docs                           # refresh docs/api.md after API changes
```

Writes need an operator key in `.env`: `LINETWIN_API_KEYS_OPERATOR=name:<key>`, sent as `X-API-Key`.

### Tests

pytest runs 671 tests (557 before this phase). The most important new ones:

- **End to end** (`tests/test_phase4_e2e.py`, over **MQTT**, **OPC UA** and the **mock server**). A simulated full line runs at 20x behind the publisher; the backend runs on a real server with SQLite and API keys.
  - STOP CNC-01 → ack EXECUTED → CNC-01 IDLE → CNV-01 BLOCKED, INSP-01 and CNV-02 STARVED, all with root cause CNC-01. The StateChanges with their reasons, the command and its acks (ACCEPTED, EXECUTED) are in the API and the store.
  - A `line_stoppage` alert is raised; after START the line runs again and the alert clears, stays listed, is acknowledged and resolved. The store holds raise, clear and ack.
  - A repeated idempotency key returns the original ack without a second send; reuse with a different request is 409.
  - An injected belt jam is EXECUTED. CNV-01 goes to FAULT and a CRITICAL `machine_fault` alert appears. The stored request precedes the FAULT on the event-time axis. SET_SPEED on the faulted machine is rejected by the safety rule; STOP passes it.
  - Measured: about 9 s (mock), 14 s (OPC UA) and 9 s (MQTT) per run.
- **STOP fail-safe:** STOP passes the safety evaluator in every state (unit tests). Over the mock server with the publisher paused, STOP is ACCEPTED while CNC-01 is STALE and OFFLINE, and SET_SPEED is rejected ("data link is down"). An OFFLINE alert is raised.
- **Rejections:** the file source has no command capability (REJECTED with reason, audited); viewer 403; no keys 403 with a hint; unknown machine 404; missing SET_SPEED param 422; scenario with the file source 409, stored.
- **`/replay?t=` equals `reconstruct(t)`** at 9 times, and `/history` raw / 1m / 10m equals the store queries, on both backends.
- **Slow WebSocket client** (`backend/tests/test_api_ws_backpressure.py`): against 160 readings/s, one client that never reads and one that reads every 0.5 s. The normal client receives every reading, ingestion keeps its pace, the tracker and writer drop nothing, and the slow clients' buffers stay bounded. Also: outbox coalescing and drop-oldest; a stalled client is closed (4408).
- **Root cause matches the simulator:** on generated `cnc_stop`, `quick` and `quick_single` runs, the engine's root cause equals the simulator's `root_cause_machine` on every sample. I also ran the same check on 40 runs of `data/sim` by hand: 576,000 samples, 0 mismatches. The shipped transition table accepts every transition in those runs.
- **Engine:** StateChange reasons, line stoppage debounce, fault / warning / offline alerts, data quality from quarantine, live state == reconstruct with the engine running and alerts restored after a restart (both backends).
- **Restored alerts wait for fresh data** (review fix): a raised `machine_fault` and `line_stoppage` survive a restart through the store with the same ids. They are not cleared by the empty state, partial data or a data-quality tick, are confirmed by fresh data, and clear with the same id once the data ends the condition. Both variants fail with the gate removed.
- **Commands unit tests:** audit order and event time, safety rejection not sent, capability, in-flight dedup, TTL, timeout, restore of idempotency keys from the store (both backends).
- **Simulator:** `inject_fault` (jam trips, kind rules, duplicates, short gradual fault, snapshot/restore), `schedule_plan` offsets, `apply_scenario` validation, the request contract (no path traversal), `submit_call` error handling.
- **Auth and logs:** role matrix, loopback rule, WebSocket auth (wrong key, subscribe first, missing auth, timeout → 4401, anonymous only without keys). **No key appears in any log line** (structlog, uvicorn access and error logs, a key in a query string, the WebSocket auth message).
- **Minimal mode** (`tests/test_phase4_minimal.py`): `serve --mode minimal` in a fresh process (plain `Popen`), no keys. The API serves live state for every machine, history comes from SQLite, the WebSocket streams anonymously, writes are refused, no broker, DB driver or ML module is loaded, and on Windows the loop is a selector loop.
- **Docs and launcher:** `docs/api.md` is up to date; launcher argument building, service probes, cleanup on exceptions, dead publisher, and a smoke run with a real mock-server child process.
- **Config:** shipped configs load; missing states or kinds are rejected; a rule naming STOP is rejected; the line file and API key settings; `.env.example` placeholders.

### Verified here (Windows 11, Docker services up)

| Check | Result |
|---|---|
| `uv run poe lint` | passes |
| `LINETWIN_REQUIRE_SERVICES=1 uv run poe test` | 671 passed, 0 skipped (vitest: no test files yet) |
| `uv run poe minimal-data-check` | 12 passed |
| e2e, WebSocket backpressure, engine and minimal tests | passed 3 runs in a row |
| `uv run poe minimal` by hand | serves live state; `eventLoop` `_WindowsSelectorEventLoop`; no optional heavy module loaded; writes 403 without keys |
| `uv run poe dev --source mqtt --speed 10` by hand | publisher started, backend on TimescaleDB; STOP EXECUTED; belt-jam injection EXECUTED |

### Known limitations

- **Hard kill of the dev launcher:** Ctrl+C stops the simulator child (cleanup in `finally`), but killing the launcher process outright leaves the child running.
- **Audit queries are unpaged:** `GET /commands`, `GET /commands/{id}` (after a restart) and `GET /scenario` read their streams in full from the store. That is fine at operator command rates; large volumes would need indexed lookups.
- **Live state starts empty after a restart:** only alerts are restored. Seeding the state from the store would hide a restarted file replay, whose readings are older than the stored ones.
- **Data quality is live-only:** it uses a wall-clock window and is not stored; only its alerts are.
- **Command `ts` before any data:** with no reading yet, audit timestamps fall back to the wall clock.
- **WebSocket coalescing:** a slow client sees the newest reading per machine, not every reading; `/history` has them all.
- **Not run:** the test suite with the services stopped (the MQTT e2e and TimescaleDB tests are marked `services` and skip with the start hint, as before), and the CI matrix (there is no remote).

### Deviations from the plan / CLAUDE.md

- **Contracts:** `Alert` gained `active`, `acknowledged`, `ackedBy`, `ackedAt` and `rev`; the unique key is `(alert_id, rev, ts)` because TimescaleDB needs `ts` in it. New contracts `ScenarioRequest` / `ScenarioAck`; new enums `DataQuality`, `Role` and `ScenarioAction`; new `Capability.SCENARIO`. CLAUDE.md §6 is updated.
- **New bus topics** `quarantine` and `scenarios`, and the new stream `scenario_requests`.
- **New config files** `engine.yaml`, `api.yaml` and `commands.yaml` (CLAUDE.md §5 updated). The API key header is fixed (`X-API-Key`) rather than configurable. `sources.yaml` mock gained `scenario_path`.
- **New settings:** `TwinSettings.line_file`, `api_keys_operator` and `api_keys_viewer`.
- **Dependencies:** `fastapi` and `uvicorn` added to `twin-backend` (already in the stack and the lockfile via the simulator).
- **New files outside the layout:** `twin_backend/__main__.py`, `devstack.py`, `api/docs.py`, `api/routes/`, `commands/scenarios.py`, `twin_sim/publishers/scenario.py`.
- **Additions beyond the task list:** `GET /alerts/history`, `GET /commands` (list), `GET /scenario[/{id}]`; the CLI options `--replay-speed`, `--sqlite-path`, `--file`, `--source`, `--line`.
- **Command service:** the module is `commands/service.py`, not `bus.py` as named in the plan. `SourceManager.send_command` was split into `route_command` (no publishing) plus publishing, so the service controls the order of audit events.
- **Simulator:** `LineSimulator.inject_fault` / `schedule_plan`, `RealtimeRunner.submit_call` (commands now go through it), `SimSession.submit_scenario`.
- **Audit timestamps:** command and scenario audit `ts` values are on the event-time axis, and the simulator's scenario acks carry simulation time like its command acks. Neither was specified for commands; I chose this for consistency with ADR 0010.
- **Fixes made along the way:**
  - engine shutdown order, so the last StateChanges are folded before storage stops;
  - uvicorn turning off access-log propagation process-wide;
  - log redaction of non-string arguments;
  - read-after-write lag in the audit listings (they now merge this process's in-memory records).

## Phase 3: Storage and event sourcing (done, 2026-11-12)

Delivered in two parts: part A (SQLite, writer, snapshots, replay, export, wiring) and part B (TimescaleDB, test-database isolation, benchmark, docs).

### What was built

- **Contracts and Phase 2 changes:**
  - New `Alert` contract and `AlertSeverity` enum.
  - `SourceAdapter.time_scale`: the replay speed for the file adapter, 1.0 for live sources. `SourceManager` passes it to the pipeline.
  - **Staleness `StateChange.ts` is now event time:** the last reading ts + wall silence × time scale. A recovery gets the resuming reading's ts. See the ADR 0007 amendment and [ADR 0010](adr/0010-event-time-replay-and-snapshots.md).
  - **Bus** (ADR 0006 amendment): `subscribe(topic, maxsize=...)`; publish-time stamps (`get_stamped()`), stored as `ingested_at`; `qsize()`.
- **Line state** (`engine/line_state.py`): `LineState` and an order-independent fold (`apply_event`) that never applies LATE readings, plus `LiveStateTracker`. The live state and `reconstruct` use the same fold.
- **Storage** (`storage/`, [ADR 0009](adr/0009-event-store-and-migrations.md)):
  - `EventStore` interface with seven event streams plus snapshots. Inserts are idempotent; every row keeps event-time `ts` and wall-clock `ingested_at`.
  - **SQLite** (`sqlite.py`): standard library only, one worker thread, WAL, integer-microsecond timestamps. Downsampling is done in SQL.
  - **TimescaleDB** (`timescale.py`, asyncpg, imported only when selected):
    - hypertables for all event tables;
    - continuous aggregates `readings_1m` and hierarchical `readings_10m` (sum, count, min, max, `n_late`, `n_suspect`), with real-time aggregation;
    - refresh policies whose windows cover `max_late_arrival_s`; optional compression and retention policies; chunk intervals from config;
    - inserts via COPY into a staging table, then `ON CONFLICT DO NOTHING`;
    - readings older than the policy window (replays, backfills) are queued in the table `aggregate_refresh_queue`, written in the same transaction as the readings, and refreshed before the next 1 min / 10 min history query that covers them, in any process and across restarts (ADR 0010 amendment).
  - **Migrations:** versioned plain SQL (`migrations/{sqlite,timescale}/NNNN_name.sql`, runner `migrator.py`) with CRLF-insensitive checksums and a no-transaction marker for continuous aggregates.
  - **`StorageWriter`:** own bounded bus queues (drop-oldest), batches, retries, metrics (`linetwin_storage_dropped_writes_total{stream,reason}`, written, duplicates, batch latency, queue depth). Quarantine records reach it through a tee sink.
  - **`Snapshotter`:** epoch-aligned snapshots every `interval_s` of event time, up to the watermark minus `settle_s`. Each snapshot is computed by `reconstruct`. Snapshots are deleted when older non-LATE data arrives later.
  - **`replay.py`:** `reconstruct(t)` (snapshot + replay) and `iter_states(t0, t1, step)`.
  - **`export.py`** and the CLI `python -m twin_backend.storage migrate | info | export`: canonical Reading CSV/JSONL that the file adapter replays; `--all` also writes the other streams.
  - **`service.py`:** `StorageService` and `build_storage` / `build_store`.
- **Wiring:**
  - `SourceManager(..., storage=...)` starts storage before the adapter, stores quarantined messages, and on stop publishes what the reorder buffer holds, then drains the writer.
  - `twin_backend.runtime.build_runtime()` loads `line.yaml`, `sources.yaml` and `storage.yaml` and checks `settle_s ≥ lateness_window_s + flush_interval_s`.
- **Config:** new `config/storage.yaml` (`backend: sqlite` by default) and `StorageConfig`. Cross-field rules: refresh `start_offset_s ≥ max_late_arrival_s + end_offset_s`; retention longer than every refresh window. New `TwinSettings.storage_config_path`.
- **Tests:**
  - **Test database:** session fixture `timescale_test_db` creates `linetwin_test_<hex>`, migrates it and drops it with `WITH (FORCE)`.
  - **Both backends:** `STORE_BACKENDS` runs every store-level test on SQLite and on TimescaleDB (`services` marker).
- **Benchmark:** `storage/bench.py` (`uv run poe storage-bench`) → `experiments/results/storage_bench.md` (committed).
- **Results in git:** `.gitignore` now tracks reports, summaries and plots under `experiments/results/` and ignores large or regenerable artefacts. `tests/test_results_policy.py` enforces the types and a 5 MB limit (CLAUDE.md §7).
- **poe tasks:** `db-migrate`, `db-info`, `db-export`, `storage-bench`. `minimal-data-check` now also runs `tests/test_phase3_minimal.py`.
- **Docs:**
  - ADRs 0009 and 0010, and amendments to 0006 and 0007.
  - architecture.md: a storage section.
  - api.md: Alert, the store CLI and export format, and how to configure the file adapter (default mapping) to replay an export.
  - setup.md: the test database and how to switch to TimescaleDB.
  - CLAUDE.md §5 (layout) and §6 (StateChange, Alert).

### How to run

```text
uv run poe minimal-data-check             # file -> ingest -> bus -> SQLite, no services
uv run poe db-info                        # what the configured store holds (SQLite by default)
uv run poe db-export --format csv --out data/export/run.csv --from 2026-01-01T00:00:00Z
uv run poe up; uv run poe storage-bench   # both backends -> experiments/results/storage_bench.md
```

- **Replay an export:** set `file.path`/`format` to the export and `file.mapping: {}` in `sources.yaml`. The shipped mapping is for the simulator CSV; see [api.md](api.md#replaying-an-export).
- **TimescaleDB as the live store:** set `backend: timescale` in `config/storage.yaml`.
- **Backend entry point:** `build_runtime()` is what the Phase 4 `minimal` / `dev` entry points will use. Until then storage is used from code and tests.

### Tests

pytest runs 557 tests. The most important new ones:

- **reconstruct equals live** (`tests/test_phase3_integration.py`, both backends):
  - A seeded 120 s full-line run, edited to contain a 16 s CNC-01 outage (STALE, OFFLINE, then recovery) and one reading that arrives late (LATE), goes through the file adapter, ingest and the writer.
  - `reconstruct(t)` equals the live state recorded at 25 random settled times, both from raw replay and from snapshots. CNC-01 shows OFFLINE during the outage, and the LATE reading is never applied.
- **Export → replay → store** (CSV and JSONL, both backends): identical readings, including nulls, SUSPECT and LATE.
- **Round trips** (both backends): partial readings and every quality flag round-trip exactly, as does every other stream (including quarantine payloads with NUL bytes). Re-inserts are idempotent.
- **Backpressure** (`backend/tests/test_storage_writer.py`):
  - With the store blocked for the whole run, 2000 messages still go through the pipeline, and other subscribers get all of them.
  - Afterwards written + dropped = published, the survivors are the newest rows, and drops are counted.
  - Also covered: retry then success; drop after all retries; bounded quarantine queue; stop drains.
- **Downsampling** (both backends): 1 s / 1 min / 10 min match a numpy reference (`nanmean`, `nanmin`, `nanmax`, non-null counts) on 1500 s of seeded data with jitter, about 20 % nulls, a 100 s stretch with a single sensor, a 700 s gap, and LATE/SUSPECT rows. Also tested: unaligned ranges and the machine filter.
- **TimescaleDB specifics** (`backend/tests/test_storage_timescale.py`):
  - **Late arrival:** a LATE reading inserted into an already materialized 1-min bucket is absent until the refresh policy job runs (`CALL run_job`), then present in both the 1 min and 10 min aggregates. Mean, max, counts and `n_late` are checked against numpy.
  - **Backfill below the watermark** is refreshed on read, and the queue is empty afterwards.
  - **Across restarts and processes:** one store instance writes 20 minutes of old sim-time data in three batches and closes. A new instance finds three queued refreshes, returns all 20 one-minute buckets (60 readings each, temperature mean checked against numpy) and both 10-minute buckets, then has an empty queue. A control row written without a queue entry stays invisible.
  - **Recent data is not queued**, because the policies cover it.
  - **Policies follow config:** hypertables, real-time aggregates, refresh offsets and schedule, chunk interval; compression and retention policies added and removed with the flags; unchanged settings are not re-applied.
  - **Isolation:** the test database is not the configured one.
- **Snapshots** (both backends): aligned times; snapshot equals replay; deletion when older data arrives later, but not for LATE readings; `iter_states`; the service persists bus events across a restart.
- **Units:** reducer rules and order independence (20 shuffles); event-time staleness at speeds 1, 10 and 0.5; bus per-subscription bounds and publish stamps; migration runner (checksums, CRLF, tampering, re-runs); `StorageConfig` validation; export format; CLI (subprocess); benchmark script.
- **Minimal mode** (`tests/test_phase3_minimal.py`): in a fresh interpreter, `build_runtime` with the shipped configs persists the sim-quick CSV and JSONL to SQLite, and `reconstruct` works. No broker, database driver beyond `sqlite3`, HTTP or ML module is loaded.

### Verified here (Windows 11, Docker services up)

| Check | Result |
|---|---|
| `uv run poe lint` | passes |
| `LINETWIN_REQUIRE_SERVICES=1 uv run poe test` | 557 passed, 0 skipped (vitest: no test files yet) |
| `uv run poe minimal-data-check` | 7 passed |
| Test/bench databases left after the run | none (`linetwin`, `postgres`, templates only) |
| Storage tests with TimescaleDB unreachable (`LINETWIN_PG_PORT=1`) | 21 passed, 22 skipped with the start hint; with `LINETWIN_REQUIRE_SERVICES=1` they error |
| TimescaleDB-specific tests | passed 3 runs in a row |
| Documented export-replay YAML (api.md) | replays CSV and JSONL exports identically, 0 quarantined |
| `uv run poe storage-bench` | ran; numbers below |

### Measured write throughput

From `experiments/results/storage_bench.md`, generated by the script on this machine: 18 logical CPUs, SQLite 3.50.4, PostgreSQL 16.15 / TimescaleDB 2.31.0 in Docker Desktop, shipped writer settings, 10 s per rate.

| Backend | Sustained through StorageWriter (kept up, no drops) | Direct insert, batch 1000 | Direct insert, batch 5000 |
|---|---:|---:|---:|
| SQLite | 9,969 rows/s | 63,732 rows/s | 68,431 rows/s |
| TimescaleDB | 4,992 rows/s | 13,895 rows/s | 17,126 rows/s |

- **Rates tested:** 1k, 2k, 5k, 10k, 20k and 40k rows/s. The next rate above each sustained value (20k for SQLite, 10k for TimescaleDB) fell behind, so the real limits lie somewhere between those steps.
- **COPY vs executemany:** TimescaleDB with plain `executemany` at batch 1000 reached 5,373 rows/s; the COPY path the store uses reached 13,895 rows/s.
- **For scale:** the simulated line produces 4 readings/s.

### Known limitations

- **Aggregates can be briefly stale:** a late reading inside the policy window reaches the 1 min / 10 min aggregates only on the next refresh-policy run (≤ `schedule_s`, 60 s / 300 s). Raw data and `reconstruct` are always current.
- **Network adapters assume `time_scale = 1`:** a simulator publisher run with `--speed ≠ 1` behind MQTT, OPC UA or mock gets approximate staleness timestamps.
- **Backlog after STALE:** readings that arrive in order but delayed after a STALE mark show STALE until a reading newer than the STALE ts arrives. This is the same live and in replay.
- **reconstruct is unpaged:** it loads all events since the snapshot into memory. That is fine with snapshots every 60 s; with snapshots off, very long ranges would need paging.
- **NUL characters:** a NUL in a plain text column (e.g. quarantine `detail`) is stored as U+FFFD on TimescaleDB. JSON payloads keep it.
- **Retention:** with retention enabled, `reconstruct` before the retention horizon is incomplete.
- **Benchmark caveats:**
  - Producer and writer share one Python process and event loop, so at overload (40k rows/s offered) the producer starves the writer and almost nothing is written. That is a property of the benchmark setup, not of the store.
  - TimescaleDB ran in Docker Desktop on Windows, which adds network overhead.
- **CI not run:** the Ubuntu CI job with services has not run on GitHub yet (there is no remote).

### Deviations from the plan / CLAUDE.md

- **New contract** `Alert` (§6 updated). §6 also lists `StateChange`, whose `ts` is now event time (ADR 0010).
- **Insert API:** `EventStore.insert(stream, rows)` is one generic method instead of one insert method per table.
- **Runner module name:** `storage/migrator.py` (not `migrations.py`), to avoid clashing with the `migrations/` SQL folder.
- **1 s history** is computed as 1 s buckets (equal to raw data at 1 Hz).
- **Shutdown:** `SourceManager.stop()` with storage publishes what the reorder buffer holds before draining the writer.
- **Refresh policy start:** a new policy first runs one schedule interval after it is created (`initial_start`). Running it at once collided with explicit refreshes in tests.
- **jobs view:** TimescaleDB 2.31 reports the continuous aggregate's view name in `timescaledb_information.jobs`; the job lookup accepts that or the materialization table name.
- **Benchmark method:** a paced rate ladder instead of a flat-out producer, which starved the writer of CPU and measured that instead of the store. "Sustained" means no drops and a bounded backlog at the end of the window.
- **Not mentioned in the plan:** new files `twin_backend/runtime.py`, `storage/rows.py`, `storage/metrics.py` and `storage/bench.py`.
- **Persistent refresh queue (correction before commit):** pending aggregate refreshes for old data used to be kept in process memory. They now live in `aggregate_refresh_queue` (migration 0003), so they work across processes and restarts.
- **Results are committed (correction before commit):** `experiments/results/` is no longer ignored as a whole. Reports, JSON summaries, plots and small CSVs are tracked; large or regenerable artefacts are ignored. `tests/test_results_policy.py` enforces the allowed types and a 5 MB per-file limit. `storage_bench.md` and the `sim_examples` plots are now in git. CLAUDE.md §7 states the rule.

## Phase 2: Adapters and ingestion (done, 2026-11-12)

### Fix: MQTT publisher buffer race (after CI on ubuntu-latest)

- **Symptom:** `test_commands_over_mqtt` failed with `IndexError: pop from an empty deque` in `MqttPublisher._flush`.
- **Cause:** on `stop()`, `_serve` flushed the buffer while the publish task was still waiting for a PUBACK, so two coroutines drained the same deque. Besides the crash, this sent readings twice and dropped others. Separately, the `deque(maxlen=...)` could evict the reading that was being sent, and then the wrong reading was removed.
- **Fix:**
  - The publish loop is now the only drainer, guarded by an `asyncio.Lock`. On `stop()` it drains what is left and returns; `_serve` waits for that instead of flushing itself.
  - The reading being sent is taken out of the buffer first and put back at the front if the publish fails or is cancelled. Delivery stays at least once (dedup removes duplicates), in order.
  - The buffer bound now drops the oldest buffered reading, never the one in flight.
  - The retained state is marked as published only after the broker accepted it.
- **Other senders checked:**
  - MqttAdapter has no buffer, only pending ack futures.
  - The mock server's WebSocket feed and the adapter queues are single-reader `asyncio.Queue`s.
  - OPC UA events are triggered sequentially by the runner.
- **Test:** `simulator/tests/test_sim_mqtt_publisher.py` uses a fake client with a slow PUBACK, so the interleaving is deterministic. On the old code it reproduces the exact CI error (duplicate seq 1, lost seq 8, `IndexError` at the same line) and the eviction bug. After the fix it passes, along with a test that a failed publish is resent first on the next connection.
- **Verified:**
  - `uv run poe lint` passes.
  - `uv run poe test` with services up and `LINETWIN_REQUIRE_SERVICES=1`: 419 passed.
  - The MQTT unit and integration tests passed 10 runs in a row; the race tests passed 20 runs in a row.

### What was built

- **Shared code (`twin_common`):**
  - `Reading` sensor fields are nullable (`null` = missing); `schemaVersion` stays 2. Helpers: `sensor_values()`, `missing_sensors()`.
  - New `StateChange` contract (bus topic `states`).
  - New enums: `Capability`, `AdapterStatus`, `QuarantineReason` (including `NO_SENSOR_DATA`).
  - `units.py`: unit-conversion table into the catalog units.
  - `opcua_names.py`: OPC UA browse names, event type and method table, shared by server and adapter.
  - `aio.py`: `run()` and `new_event_loop()` use a selector loop on Windows; `require_selector_loop()` gives a clear error otherwise ([ADR 0008](adr/0008-windows-selector-event-loop.md)).
- **`config/sources.yaml` / `SourcesConfig`:**
  - A field mapping per adapter (`fields`, `units`, `drop_unmapped`).
  - New required sections: `reconnect`, `ingest`, `bus`, `chaos`, plus `command_timeout_s`.
  - Per-adapter settings:
    - `mqtt.qos: 1` and `mqtt.session_expiry_s`;
    - `opcua.publishing_interval_ms`, `event_queue_size` and `watchdog_s`;
    - `file.format` accepts `csv | jsonl | json`, plus `start_offset_s`;
    - `mock.ws_path` is now `/stream`, plus `history_path`, `command_path`, `rest_poll_s` and `history_limit`.
  - The default replay file is now `data/sim_quick/run_0000.csv`; the old default never existed.
  - Validation rules: STALE before OFFLINE, `seq_reset_threshold <= dedup_window`, and a JSONL quarantine needs a path.
- **Backend (`twin_backend`):**
  - **`adapters/`:**
    - `base.py`: `SourceAdapter` protocol, `RawMessage`, `AdapterHealth`, `BaseAdapter` (I/O task, bounded queue, re-cancelling `stop()`) and `Backoff`.
    - Adapters: `file.py` (standard library only), `mock.py` (WebSocket, `/history` backfill, REST fallback), `mqtt.py` (aiomqtt, MQTT v5 persistent session, commands via `cmd`/`cmd/ack`) and `opcua.py` (ReadingEvent subscription, watchdog, method calls).
    - `chaos.py`: seeded drop, duplicate, delay, reorder and corrupt faults.
    - `registry.py` (lazy imports) and `manager.py` (`SourceManager`: active adapter, pump and ticker, hot `switch()`, `send_command`).
  - **`ingest/`:** normalizer, validator (missing values, ranges), quarantine (memory or JSONL), dedup, reorder buffer, staleness monitor, Prometheus metrics, and the pipeline that ties them together ([ADR 0007](adr/0007-adapters-and-ingest.md)).
  - **`bus/`:** `EventBus` protocol and `InMemoryEventBus` (bounded per-subscriber queues, drop-oldest) ([ADR 0006](adr/0006-event-bus-interface.md)).
  - **`clock.py`:** `SystemClock` and `FakeClock`.
- **Simulator publishers (`twin_sim.publishers`, never imported by `twin_sim` itself):**
  - `session.py`: `SimSession` wraps the RealtimeRunner, with history and listeners.
  - `mock_server.py`: FastAPI app with `/info`, `/state`, `/history`, `WS /stream` and `POST /command`.
  - `mqtt.py`: telemetry plus a retained state topic, commands and acks, a buffered publish loop and reconnect.
  - `opcua.py`: objects, variables and methods per machine, plus one `ReadingEventType` event per reading.
  - `http.py`: uvicorn started inside the running loop.
- **CLI and poe tasks:**
  - New subcommands `mock-server`, `mqtt` and `opcua`, which take the same `--line`, `--scenario`, `--seed`, `--run-index`, `--speed`, `--duration` and `--history-size` options as `run`. Their defaults come from `sources.yaml`.
  - New poe tasks: `sim-mock`, `sim-mqtt`, `sim-opcua`.
  - `run` now starts through `aio.run`.
  - `minimal-data-check` also runs `tests/test_phase2_minimal.py`.
- **Config:** the load catalog maximum was raised from 120 to 130 % in `line.yaml` and `line_single.yaml`, after the sensor-range check (approved). The line.yaml comment now describes the configurable out-of-range handling.
- **Docs:**
  - [api.md](api.md) is new: Reading, the mock server, MQTT topics and the OPC UA address space.
  - [architecture.md](architecture.md) has a new ingest section; [simulator.md](simulator.md) a new §11 on the publishers.
  - ADRs [0006](adr/0006-event-bus-interface.md), [0007](adr/0007-adapters-and-ingest.md) and [0008](adr/0008-windows-selector-event-loop.md).
  - CLAUDE.md §6 notes the nullable sensor fields.
- **Dependencies (approved):**
  - simulator: fastapi, uvicorn, websockets, aiomqtt, asyncua;
  - backend: asyncua, websockets, httpx, prometheus-client.

### How to run

```text
uv run poe sim-quick                      # replay file for the default file adapter
uv run poe minimal-data-check             # file -> adapter -> ingest -> bus, no services
uv run poe sim-mock                       # mock server on http://127.0.0.1:8100
uv run poe up; uv run poe sim-mqtt        # MQTT publisher (needs the broker)
uv run poe sim-opcua                      # OPC UA server on opc.tcp://127.0.0.1:4840/linetwin/
uv run poe sim-opcua --line config/line_single.yaml --scenario config/scenarios/quick_single.yaml
```

The backend entry point (API, `/metrics`, hot switch over HTTP) arrives in Phase 4. Until then, `SourceManager` is used from code and tests.

### Tests

pytest now runs 416 tests, up from 226. The most important ones:

- **Identical sequences** (`tests/test_phase2_integration.py`): one seeded 120 s run through File JSONL, File CSV, Mock, OPC UA and MQTT gives identical canonical Reading sequences. The MQTT case asserts QoS 1 and that the publisher dropped nothing.
- **Commands** through MQTT, OPC UA and Mock reach the simulator (SET_SPEED applied, EXECUTED) and return acks (an invalid RESET is REJECTED with the simulator's reason).
- **OPC UA:**
  - the adapter reconnects after a server restart;
  - the single-station line works;
  - a partially missing reading travels as NaN and arrives as `null` with SUSPECT.
- **Chaos:**
  - with 10 % duplicates and reordering, the output equals the simulated run exactly (no LATE readings, no gaps);
  - with 5 % corruption, every corrupted message is quarantined with a reason and nothing else is lost;
  - same seed gives the same output, and the fault rates match.
- **Mock adapter:**
  - backfill closes the startup gap;
  - a server restart mid-stream loses no seq;
  - REST fallback works while the WebSocket is unavailable;
  - commands, a server that is down, and the single-station line are covered.
- **Hot switch** from file to a mock server streaming the same run is seamless (overlap deduplicated). A failing switch keeps the current source.
- **Unit tests:**
  - normalizer: mapping and units;
  - validator: every quarantine reason, partially missing readings, suspect vs quarantine mode;
  - dedup and source reset; reorder (in order, gap filled, window timeout, LATE, initial hold);
  - staleness at exact fake-clock times; bus fan-out and drop-oldest; backoff; clock;
  - file replay: CSV/JSONL/JSON, pacing, speed, offset, loop, missing-file hint;
  - pipeline metrics.
- **Contracts:** round trips with null sensors; missing keys are accepted; non-sensor fields stay required.
- **Loop type:** on Windows, async tests and `aio.run()` run on a SelectorEventLoop, and a Proactor loop is rejected.
- **Sensor ranges** (`simulator/tests/test_sim_sensor_ranges.py`): analytic worst cases per machine kind, plus an empirical sweep (quick, sensor_faults, e5_lubrication_loss, cnc_stop, quick_single; 3 seeds × 1800 s) and a commanded 120 % overload. Everything lies inside the catalog.
- **Minimal mode** (`tests/test_phase2_minimal.py`): in a fresh interpreter, the `sim-quick` CSV and JSONL go through `SourceManager` with no broker, DB, HTTP or ML module imported.

### Verified here (Windows 11)

| Check | Result |
|---|---|
| `uv run poe lint` | passes |
| `uv run poe test`, services up, `LINETWIN_REQUIRE_SERVICES=1` | 416 passed |
| `uv run poe test`, after `uv run poe down` | 412 passed, 4 skipped (MQTT/DB) with the start hint |
| `uv run poe minimal-data-check` | passes |
| `mock-server` | smoke-run |
| `mqtt` | streamed through its adapter (full line) |
| `opcua` | streamed through its adapter (full and single-station line) |
| OPC UA test group | passed 8 runs in a row |
| Mock and source-manager tests | passed 5 runs in a row |

### Known limitations

- **Startup delay:** each machine's first readings wait up to `lateness_window_s` (2 s) after start or a detected source restart, so early out-of-order messages are not flagged LATE.
- **OPC UA outages lose data:** readings fired while the adapter is disconnected are lost and show up as a gap. Mock (history backfill) and MQTT (persistent session) recover short outages.
- **No MQTT broker-outage test:** stopping the shared Docker broker from a test is not done. Publisher buffering and the adapter's persistent session are covered by code review only.
- **Dropout rows in CSV:** the CSV export contains sensor-dropout rows with every sensor NaN; they are quarantined as `NO_SENSOR_DATA`. JSONL omits them.
- **Looped file replay** restarts `runTime` on every pass.
- **Load and CSV order:** the file adapter loads the whole file into memory and sorts CSV rows by (ts, line order). That is fine for simulator exports; very large files would need streaming.
- **Metrics not yet served:** they are collected per pipeline, but nothing serves `/metrics` before Phase 4.
- **OPC UA test speed:** the OPC UA tests take 3 to 8 s each, mostly asyncua connection setup and teardown.
- **CI:** the new jobs (Ubuntu with services, Windows and macOS without) have not run on GitHub yet; there is no remote.

### Deviations from the plan / CLAUDE.md

- **`Reading` contract:** sensor fields are nullable while `schemaVersion` stays 2 (requested in review). CLAUDE.md §6 was updated.
- **`QuarantineReason.NO_SENSOR_DATA`** was added. Sensor NaN/inf now means "missing", not `NON_FINITE`, which now applies only to `runTime`.
- **Load catalog max** raised from 120 to 130 % (approved).
- **New files outside the CLAUDE.md layout:**
  - `common/twin_common/aio.py`, `units.py` and `opcua_names.py`;
  - `backend/twin_backend/clock.py`;
  - `simulator/twin_sim/publishers/` (CLAUDE.md §5 lists publishers under the simulator without a folder).
- **Root `conftest.py`** now also sets the event loop type, through pytest-asyncio's `pytest_asyncio_loop_factories` hook; overriding `event_loop_policy` is deprecated in pytest-asyncio 1.4.
- **Two workarounds:**
  - A pytest filter ignores Starlette's "use httpx2" TestClient deprecation warning.
  - `OpcUaServer.stop()` closes asyncua's listening socket first. Otherwise a client reconnecting during shutdown keeps `wait_closed()` (Python 3.11.15) waiting forever.
- **Python 3.11 cancellation:** `BaseAdapter.stop()` re-cancels its task until it ends, because on Python 3.11 `asyncio.wait_for` (used inside asyncua) can swallow a cancellation. Both of these hangs were seen in testing and are fixed.

## Infrastructure: Docker Compose services (2026-11-12)

### What was built

- **`docker-compose.yml`** (project `linetwin`). Both services publish their ports on 127.0.0.1 only.
  - **`timescaledb`:** `timescale/timescaledb:2.31.0-pg16`, i.e. PostgreSQL 16 with TimescaleDB. 2.32.0 was published the same day, so I took the previous release. Named volume `timescale-data`, health check `pg_isready` over TCP.
  - **`mosquitto`:** `eclipse-mosquitto:2.1.2-alpine`, matching the native 2.1.2 (2.1.x is only published as alpine tags). Named volume `mosquitto-data`, health check `mosquitto_pub`.
- **`.env.example`** (committed; `.env` stays gitignored): `LINETWIN_PG_DB`, `LINETWIN_PG_USER`, `LINETWIN_PG_PASSWORD`, `LINETWIN_PG_PORT`. Docker Compose and `TwinSettings` both read it.
- **Mosquitto config split:**
  - `config/mosquitto.d/common.conf`: shared settings;
  - `config/mosquitto.docker.conf`: container entry, all interfaces inside the container;
  - `config/mosquitto.conf`: native entry, `listener 1883 127.0.0.1` plus `include_dir`.
- **poe tasks:** `up` (starts and waits until healthy), `down`, `reset` (also deletes volumes), `logs`, `ps`. All are plain `docker compose` commands.
- **`TwinSettings`:** new fields `pg_host`, `pg_port`, `pg_db`, `pg_user` and `pg_password` (SecretStr), and new properties `pg_dsn` and `database_url`. Reads `.env`; real environment variables take precedence.
- **Backend dependencies:** `asyncpg`, and `aiomqtt` (which brings `paho-mqtt`).
- **Test fixtures** in `twin_backend/testing.py` (plugin registered in the root `conftest.py`): `mqtt_broker` and `timescale_db`.
  - When a service can't be reached, they skip with "… not reachable at … - start services with: uv run poe up".
  - With `LINETWIN_REQUIRE_SERVICES=1` they fail instead.
  - The pytest marker `services` is registered.
- **Tests:**
  - `tests/test_compose_services.py`: an MQTT publish/subscribe round trip, and a PostgreSQL 16 check that the `timescaledb` extension is available and installed;
  - `common/tests/test_settings.py`: settings defaults, `.env` and environment overrides, URL escaping, and `.env.example` matching the defaults.
- **CI:** the Ubuntu job runs `uv run poe up` before the tests and sets `LINETWIN_REQUIRE_SERVICES=1`. Windows and macOS don't start services, and the service tests are skipped there.
- **Docs:**
  - **CLAUDE.md:** architecture diagram, §4 storage and messaging, §5 layout, and the §7 rule (services via Docker Compose, native as the fallback, minimal mode never requires Docker; the poe rule now allows `docker compose`).
  - **ADRs:** [ADR 0005](adr/0005-docker-compose-services.md) is new; [ADR 0002](adr/0002-native-services-no-docker.md) is marked superseded.
  - **Setup guide:** `docs/setup_windows.md` was renamed to [setup.md](setup.md). Docker Desktop (Windows, macOS) and Docker Engine (Linux) are the main path; native Mosquitto and PostgreSQL with TimescaleDB are the fallback.
  - **architecture.md:** updated.

### How to run

```text
uv run poe up      # start + wait until healthy (first run downloads the images)
uv run poe ps      # status
uv run poe test    # service tests run when the services are up, otherwise they are skipped
uv run poe down    # stop (data kept);  uv run poe reset  # stop and delete volumes
```

### Verified here (Windows 11, Docker 29.9, Compose v5.7)

- `uv run poe up`: both services healthy. The first run spent about 11 minutes downloading the images; later starts take seconds.
- **With services up:** `uv run poe lint` passes, and `uv run poe test` with `LINETWIN_REQUIRE_SERVICES=1` gives 226 passed (both service tests ran).
- **After `uv run poe down`:** `uv run poe lint` passes, and `uv run poe test` gives 224 passed and 2 skipped with the start hint.
- With services down, `LINETWIN_REQUIRE_SERVICES=1` makes the service tests error, as intended.
- `uv run poe reset` followed by `up`: on a fresh volume the extension is recreated and the service tests pass. `uv run poe logs` works.

### Known limitations

- **Port 5432 is taken on this machine** by a native `postgres` process (PID at the time: 6716, not a visible Windows service), so Docker could not bind it. A local `.env` (not committed) sets `LINETWIN_PG_PORT=5433`. If that native PostgreSQL is no longer needed, stop it and delete the `.env` line to go back to 5432.
- **Mosquitto 2.1 quirk:** a per-listener option such as `protocol` in a file without a listener makes mosquitto 2.1 create an extra default listener, which collides on 1883. `common.conf` therefore contains no per-listener options; a comment there explains why.
- **The native fallback configs were not run here.** `mosquitto -c config/mosquitto.conf` and a native PostgreSQL with TimescaleDB are documented but untested, because the native Mosquitto service is disabled.
- The Ubuntu CI job with services has not run on GitHub yet (there is no remote).

## Tooling: poethepoet task runner (2026-11-12)

### What changed

- **One OS-neutral task runner.** The `Makefile` and `scripts/dev.ps1` were removed. Every developer task is now a [poethepoet](https://poethepoet.natn.io/) task in the root `pyproject.toml` (`[tool.poe.tasks]`), run as `uv run poe <task>`. The same commands work on Windows, Linux and macOS. See [ADR 0004](adr/0004-poe-task-runner.md).
- **Tasks:**
  - kept from before: `install`, `lint`, `fmt`, `test`, `dev`, `minimal` (`dev` and `minimal` are still stubs that exit with code 1), `sim`, `sim-quick`, `sim-quick-single`, `sim-plots`;
  - new, moved out of CI: `minimal-data-check` and `build-frontend`;
  - the old `help` target is replaced by `uv run poe`, which lists every task with its help text.
- **No shell syntax.** Tasks run without a shell and call only tools from the uv environment, `uv`, and `npm --prefix frontend`. The environment variable for `minimal-data-check` is set through poe's `env` option.
- **CI:** one job with an OS matrix (`ubuntu-latest`, `windows-latest`, `macos-latest`, `fail-fast: false`). Every check step is `uv run --frozen poe <task>`: lint, test, minimal-data-check, build-frontend.
- **CLAUDE.md:** §4, §5 and §8 now refer to poe. §7 has a new rule: developer tasks are added only as poe tasks, using cross-platform commands only.
- **Docs:** `setup_windows.md`, `simulator.md` and `config/line_single.yaml` use `uv run poe …`. The CLI examples in `simulator.md` no longer use PowerShell line continuations.
- **Dependency:** `poethepoet` (0.49) added to the root dev group.

### How to run

```text
uv sync --all-packages    # fresh clone only (installs poe)
uv run poe install        # uv sync + npm ci
uv run poe lint
uv run poe test
uv run poe                # list all tasks
```

### Known limitations

- The new three-OS CI matrix has not run on GitHub yet (there is no remote). Only Windows was exercised locally.
- The Phase 0 and Phase 1 sections below are historical: wherever they mention `make` or `dev.ps1`, the equivalent is now `uv run poe <same target>`.

## Phase 1: Simulator (done, 2026-11-11)

### What was built

- **`twin_sim` package:**
  - `physics.py`: first-order thermal model (exact integration), load-dependent vibration, current and speed.
  - `faults` (in `physics.py` and `line.py`): bearing_wear, overheating, spindle_imbalance (CNC only), belt_jam, and the held-out lubrication_loss.
  - `thresholds.py`: per-run perturbed thresholds with hysteresis.
  - `feasibility.py`: reachability and minimum-lead-time validator.
  - `line.py`: `LineSimulator` with discrete parts, buffers, STARVED/BLOCKED with root cause, commands, auto-recovery, sensor faults, `step` / `snapshot` / `restore` / `fast_forward`.
  - `scenario.py`: `RunPlan`.
  - `recorder.py` and `labels.py`: Arrow table, time-to-fault with a censored flag.
  - `batch.py`: multiprocessing, Parquet/CSV/JSONL, `manifest.json`.
  - `runner.py`: asyncio `RealtimeRunner`.
  - `plots.py`: example plots.
  - `__main__.py`: CLI with `generate`, `run` and `plot-examples`.
  - `testing.py`: pytest fixtures.
- **Config:**
  - `config/simulator.yaml`: every physics, fault, threshold, noise and label parameter. Loading it runs the feasibility validator.
  - `config/scenarios/`: `default`, `quick`, `e5_lubrication_loss`, `sensor_faults`, `cnc_stop`.
- **Docs:** [simulator.md](simulator.md) covers the equations, parameter tables, threshold sources (ISO 10816 zones, IEC 60034-1 insulation classes, IEC 60947-4-1 overload), how lubrication_loss differs, the feasibility numbers, label definitions and measured performance. Design decisions are in [ADR 0003](adr/0003-simulator-design.md).
- **Tooling:**
  - `sim`, `sim-quick` and `sim-plots` targets. These were originally in `Makefile` and `dev.ps1`; they are now poe tasks.
  - CI step "minimal-mode data path": `generate --quick --export csv,jsonl`, then `tests/test_phase1_integration.py` on the output. This is now the poe task `minimal-data-check`.

### How to run

```text
uv run poe sim         # 200 runs x 3600 s, seed 42 -> data/sim/ (Parquet + manifest.json)
uv run poe sim-quick   # 5 x 1800 s fault-dense runs + CSV/JSONL -> data/sim_quick/
uv run poe sim-plots   # experiments/results/sim_examples/*.png
uv run python -m twin_sim run --speed 10 --duration 120    # live Readings as JSON lines on stdout
uv run python -m twin_sim generate --help
```

Measured: the full 200-run dataset (2.88 M rows, 132 MB) generated in 9.8 s with 17 workers. One worker does 10 runs in 2.7 s, which puts the full dataset at about 1 min on a single core.

### Tests

pytest now runs 221 tests: 76 in `simulator/tests` and 3 in `tests/test_phase1_integration.py`.

- **Determinism:** same seed gives an identical table, a different seed or run gives a different one, and output is identical with 1 or 2 workers.
- **Physics:**
  - higher load gives higher steady-state temperature and current (pure functions and a full simulation)
  - exact thermal integration
  - degradation is monotone in progress
  - the cooling-loss cap holds
- **Faults:**
  - healthy line never warns at the worst case
  - every gradual fault reaches FAULT at the best case within its progression time (several seeds)
  - every gradual fault has lead time ≥ 300 s, WARNING before FAULT and a rising primary precursor at the worst case (several seeds)
  - jam drops speed and spikes current
  - sensor faults change observations but not states
  - held-out fault excluded by default
  - imbalance occurs only on the CNC
- **Feasibility validator:** rejects the old overheating model, too-fast progression and a too-low healthy margin. Readable `ConfigError`s.
- **Propagation:**
  - stopping CNC-01 makes CNV-01 BLOCKED and INSP-01/CNV-02 STARVED, all with root cause CNC-01, then recovers
  - fault propagation follows the root-cause chain
  - SOURCE starvation
- **Commands:** every type, rejections, idempotency, maintenance resets wear, RESET keeps wear, auto-recovery.
- **Snapshot:** snapshot → JSON → restore is bit-identical (including mid-jam and mid-fault). fast_forward equals repeated steps.
- **Labels:** `time_to_fault` unit tests. On generated data, ttf counts down by 1 s to onset, censored rows have no fault ahead, and non-censored rows do.
- **Runner:** order, seq and timestamps, pacing by speed, async callback, commands, stop.
- **Batch and CLI:** files, schema, manifest, JSONL parses as `Reading`, `--quick`, config errors.
- **Integration:** JSONL replays into `Reading`s time-ordered per machine. CSV agrees with Parquet. Importing the simulator pulls in no DB, broker or ML module.

### Known limitations

- `dev` and `minimal` are still stubs; they need the backend file adapter and the frontend (Phase 2+). The CI minimal-mode step checks only the file-based data path for now.
- Parquet stores observed values only. With sensor faults on, the true values are not in the dataset; they are available in `StepFrame` (`true_vib/temp/current`).
- Ambient temperature is constant per run, and speed has process noise but no separate measurement noise.
- With the default scenario, few spindle_imbalance faults reach FAULT (17 of 35 planned in the 200-run dataset), because only CNC-01 is eligible. Late or low-load gradual faults are often censored, and onsets that hit an already-faulted machine are skipped (60 of 383). All of this is recorded in the manifest.
- At the time, only `dev.ps1` was run locally (no `make`). Both have since been replaced by poe tasks (see the Tooling section above). The CI steps have not run on GitHub yet.
- I checked the example plots visually (bearing_wear, lubrication_loss, line coupling). Please review all seven PNGs.

### Enhancement: CNC spindle speed changes between jobs (2026-11-12)

- **New scenario option, `speed_changes`.** Every U(300, 900) s it issues a SET_SPEED for CNC-01 to U(7500, 10000) RPM, through the same command path as the auto-recovery operator.
  - It is in `default`, and in `quick`, `sensor_faults` and `e5_lubrication_loss`, which mirror the default operating conditions. It is not in `cnc_stop`.
  - The setpoints come from a separate seeded stream (`speed_plan`), so fault plans are unchanged: the regenerated 200-run dataset has the same planned faults as before.
- **Validated speed range.** A new per-kind `speed_operating_rpm` in `simulator.yaml` (CNC 7500–10000). The validator runs the healthy, reachability and lead-time checks at both ends of the range.
  - No parameter change was needed.
  - The tightest case is bearing wear at 10000 RPM, with a worst-case lead time of 315 s.
  - spindle_imbalance needs 380 s at 10000 RPM.
- **Scenario checks.** `check_scenario_compat` now takes the `LineConfig`. It rejects speeds outside the validated range, and speeds whose CNC cycle time would not stay below the arrival interval (7500 RPM gives 6.4 s, against 6.5 s arrivals).
- **New plot:** `imbalance_vs_bearing_speed_steps.png`. Vibration divided by the speed ratio is smooth for bearing wear and steps with speed for imbalance.
- **Tests (213 in total):**
  - healthy run at both CNC speed ends
  - reachability and lead time at both speed ends
  - validator rejects a 12000 RPM range
  - scenario speed and cycle-time rules
  - speed changes planned, applied, and independent of the fault plan
  - imbalance ∝ speed² vs bearing ∝ speed
- **Regenerated:** `sim-quick`, the plots and `data/sim`. The 200 runs took 16.2 s this time (9.8 s before), within normal run-to-run variation on this machine.

### Enhancement: single-station mode (2026-11-12)

- **`config/line_single.yaml`:** CNV-01 only, no buffers. This is the "conveyor or CNC station" from the original synopsis. The simulator needed no code change: the source feeds the machine directly and parts leave to the sink.
- **`config/scenarios/quick_single.yaml`:** `quick` without the CNC-only parts. Full-line scenarios reference CNC-01 and are refused on the single line with a clear message (`... not on line 'line1_single'`).
- **CLI:** `--line` on `generate` and `run`, defaulting to `config/line.yaml`. It replaces the earlier global `--line-config` option, which had never been documented. `plot-examples` always uses the full line.
- **Targets:** `sim-quick-single` (now the poe task `uv run poe sim-quick-single`) writes to `data/sim_quick_single/`. Ran it: 5 runs, 9000 rows, 1.6 s.
- **Tests:** 8 new in `simulator/tests/test_sim_single_line.py`:
  - healthy station shows only RUNNING with null buffers
  - gradual fault lead time and root cause
  - SOURCE starvation, STOP
  - snapshot round trip
  - full-line scenario rejected
  - batch output
  - CLI `generate`/`run` with `--line`
- **Docs:** [simulator.md](simulator.md) has a new §5 "Single-station mode"; the later sections are renumbered.

### Deviations from the plan / CLAUDE.md

- **Minimum progression time:** `D_min` is 720 s instead of 600 s, so the worst-case bearing-wear lead time stays ≥ 300 s.
- **New current thresholds** (1.15 / 1.30 × rated). They are needed so lubrication_loss can trip on overcurrent.
- **Model details changed from the original plan:**
  - bearing-wear friction is 0.2·w (was 0.3·w), to keep current flat;
  - bearing vibration scales with the speed ratio;
  - lubrication_loss has no deterministic vibration term, only stick-slip bursts.
- **New files outside the CLAUDE.md layout:**
  - `config/simulator.yaml`
  - `twin_sim/feasibility.py`
  - `twin_sim/testing.py`, a fixture plugin registered by a new root `conftest.py`, because `conftest.py` names must be unique repo-wide (ADR 0001)
- **`twin_common.logging.configure_logging`** has a new optional `stream` argument, so the `run` CLI can keep stdout for Readings.
- **Dependencies:** `numpy` and `pyarrow` were added to `twin-sim`, and `matplotlib` to the root dev group (approved). A mypy override ignores missing pyarrow type information.
- **Targets:** `sim-quick` and `sim-plots` were added on top of `sim`.

## Phase 0: Foundation (done, 2026-11-11)

### What was built

- **CLAUDE.md updated for the no-Docker decision:**
  - native Mosquitto
  - native PostgreSQL (plain features only: BRIN indexes and materialized views, no TimescaleDB)
  - Docker Compose and Grafana removed
  - `config/mosquitto.conf`, `scripts/dev.ps1` and `setup_windows.md` added to the layout
  - new rule in §7: "Do not use Docker."

  See [ADR 0002](adr/0002-native-services-no-docker.md).
- **uv workspace** with members `common/`, `simulator/`, `backend/` and `ml/`, one `uv.lock`, and Python 3.11 pinned in `.python-version`. See [ADR 0001](adr/0001-monorepo.md). The empty subpackages from CLAUDE.md §5 exist in `twin_backend` and `twin_ml`.
- **`twin_common`:**
  - `enums.py`: MachineState, Quality, HealthClass, SensorName, CommandType, AckStatus, Reliability, MachineKind, AdapterKind. It also has `to_health_class()` implementing the §6 ML target mapping. OFFLINE and STALE raise `UnlabelableStateError`.
  - `contracts.py`: `Reading` (schemaVersion 2), `Prediction`, `Attribution`, `Command`, `CommandAck`.
    - JSON is camelCase, Python is snake_case.
    - Models are frozen, `extra="forbid"`, and use UTC-normalized aware timestamps.
    - Prediction probs must cover all three classes and sum to 1. `conformalSet` is unique and in canonical order, and `ttfInterval` must satisfy lo ≤ hi.
    - SET_SPEED / SET_LOAD require a numeric `speed` / `load` param.
  - `config/`: `LineConfig`, `SourcesConfig`, `TwinSettings` (env prefix `LINETWIN_`), and `load_line_config` / `load_sources_config`. The loaders raise a single `ConfigError` that names the file and each failing location.
  - `topics.py`: MQTT topic builders. `logging.py`: structlog JSON setup. `testing.py`: sample contracts for tests in every package.
- **Config:**
  - `config/line.yaml`: CNV-01 → CNC-01 → INSP-01 → CNV-02, buffers B1/B2/B3 (capacities 10/8/10), a sensor catalog with physical ranges, per-machine sensor→part mappings, and load ranges low 30–50, medium 50–75, high 75–95.
  - `config/sources.yaml`: `active: file`, the minimal-mode default.
  - `config/mosquitto.conf`.
- **Frontend:** Vite 8, React 19, TypeScript 6 (strict, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`), ESLint 10 with typescript-eslint strict-type-checked, and Vitest 5. It has empty `src/engine`, `src/dashboard` and `src/api` folders. An ESLint rule forbids React or dashboard imports inside `src/engine`.
- **Tooling:**
  - `Makefile` and `scripts/dev.ps1`, with targets help, install, lint, fmt, test, dev, minimal and sim
  - `.pre-commit-config.yaml` (ruff, ruff-format, mypy, eslint), with local hooks run through uv/npm so versions match the lockfiles
  - `.github/workflows/ci.yml`: Linux runs `make`, Windows runs `dev.ps1` under PowerShell 5.1
- **Docs:** `architecture.md` (Mermaid), `setup_windows.md`, ADRs 0001 and 0002.

### How to run

```powershell
.\scripts\dev.ps1 install
.\scripts\dev.ps1 lint      # ruff, ruff format --check, mypy (strict), eslint, tsc
.\scripts\dev.ps1 test      # pytest (143 tests), vitest
```

### Tests

- Round trips (model → JSON → model, and via a dict) for every contract, every MachineState, Quality, CommandType and AckStatus value, and optional-field variants. The wire keys are checked to be camelCase.
- Contract rejections: unknown field, naive timestamp, schemaVersion ≠ 2, negative sensor values, bad probs, unordered ttf interval, duplicate conformal classes, missing or non-numeric command params, empty ids.
- Config errors:
  - unknown sensor
  - catalog sensor that isn't a Reading field
  - buffer from/to a machine that doesn't exist
  - buffer self-link
  - negative and zero capacity
  - duplicate machine id
  - part not on the machine
  - sensor without a part
  - overlapping, gapped or out-of-bounds load ranges
  - unknown key
  - missing file, YAML syntax error, non-mapping top level
  - active adapter without its section, unknown adapter, bad port
- Integration (`tests/test_phase0_integration.py`): the shipped configs load through `TwinSettings`. Every configured sensor is a Reading field with a part mapped, and a Reading can be built and round-tripped for every machine.

### Known limitations

- `dev`, `minimal` and `sim` are stubs that print "available from Phase 1" and exit with code 1. There is no minimal-mode CI job yet because there is nothing to run. Add it in Phase 1.
- Field-level config errors (e.g. a negative capacity) are reported before the cross-reference checks run. Cross-reference errors are all reported together, but they only appear once the field-level errors are fixed.
- `make` is not installed on the development machine. The Makefile is exercised by the Linux CI job only, and only `dev.ps1` was run locally.
- The CI workflow has not run on GitHub yet, since there's no remote.
- `pre-commit run --all-files` only sees files tracked by git. Nothing is committed yet, so only `pre-commit validate-config` was run.
- `npm audit` reports a moderate minimatch ReDoS advisory. It comes from typescript-eslint and ESLint (dev-only lint tooling), and the suggested fix downgrades ESLint, so it was left as is for now.
- No frontend unit tests yet (`vitest --passWithNoTests`). The first ones arrive with the engine's state→visual mapping.

### Deviations from the plan / CLAUDE.md

- **React 19** instead of the React 18 named in the plan, because 19 is the current stable release.
- **TypeScript ~6.0** rather than 7.x, because typescript-eslint supports TS < 6.1.
- `to_health_class` raises for OFFLINE/STALE. §6 does not map them, and they describe the data link rather than machine health.
- `Prediction.ttfS` and `ttfInterval` are nullable, where `null` means no fault is expected within the horizon.
- Test directories have no `__init__.py`, and test file names are unique repo-wide (see ADR 0001).
- `help` and `install` targets were added on top of the requested dev/minimal/test/lint/fmt/sim.

## Phase 7 follow-ups (completed 2026-11-15)

Two rounds. Round 1 added learning curves, EMA early stopping and paired comparisons. Round 2 (follow-up 2, at the end of this section) replaced early stopping with a fixed 22-epoch budget; the current E1 report and all current results are from round 2.

Learning curves and robust early stopping for the sequence models, 3-seed reruns, paired bootstrap comparisons, interpretation notes, regenerated E1 (`experiments/results/E1_clean.md`, plus `E1_clean_learning_curves.png`).

### What was built

- **Learning curves:** per-epoch train and validation loss (total and per head), validation macro F1 per horizon, its EMA and the learning rate. They are stored in each run's `metrics.json` and logged to MLflow as `curve.*` step metrics.
- **Early stopping** (`twin_ml/train/early_stopping.py`, unit tested; used by `models/nn.py`): an EMA of the validation macro F1 with `min_epochs`, `patience`, `min_delta` and a `max_epochs` cap. The best checkpoint is chosen by raw validation macro F1. Reduce-on-plateau on the validation loss.
- **Resumable reruns:** every run is tagged `model_config_sha`; `train-all --skip-existing` skips a (model, seed) only if it finished with the same hash.
- **Paired bootstrap** (`evaluate/bootstrap.py: paired_bootstrap`, `experiments/paired.py`): both sides on the same resampled test runs. Every model vs the threshold baseline, and the best sequence model vs the best tree model per horizon.
- **Report:** interpretation notes; early warning first (primary), classification second; sections for paired comparisons, sequence-model training with learning curves, and changes since the previous configuration. `docs/experiments.md` has the same notes.

### Follow-up 1 settings (superseded by follow-up 2 below)

Reduce-on-plateau for all three; `max_epochs: 16`; `min_epochs: 10`; `patience: 3` over `min_delta: 0.002`; `ema_alpha: 0.4`. Config hashes: lstm `40059aaad95e`, gru `5d9055af6981`, transformer `dc9c17caa015`. Choices were based on seed-0 diagnostics (16 epochs, MLflow `E1_curves_none` / `E1_curves_plateau`, not part of E1): constant LR gave a noisy validation score that stopped old runs on unlucky dips; plateau reduced the noise and raised +10 s macro F1 for all three.

### Follow-up 1 reruns (superseded by follow-up 2 below)

| Run | Epochs | Best epoch | Train time | MLflow run |
|---|---|---|---|---|
| lstm 0 | 16 | 15 | 614.7 s | `f518d3cea1b9418487381d5c1861b3da` |
| lstm 1 | 15 | 15 | 318.9 s | `dbc96b9eda2447a695e3628750d4b368` |
| lstm 2 | 11 | 8 | 257.1 s | `eeb1d4c738ac42f18463d6b9d0c8c324` |
| gru 0 | 16 | 16 | 561.1 s | `d3e33dc9d4834cba9b4e957e95953b9c` |
| gru 1 | 11 | 8 | 607.1 s | `8b455effceae40df9745c318620504d1` |
| gru 2 | 10 | 6 | 341.2 s | `d1d7f578f2a448aab814bbc75527ac08` |
| transformer 0 | 10 | 4 | 734.9 s | `c6fc4d445aed4667ace11c5588816ce0` |
| transformer 1 | 16 | 15 | 1213.6 s | `bdd20370808142da8cfde21aaed924a7` |
| transformer 2 | 10 | 6 | 819.1 s | `e9103c658a124f12b8a70d8ba6598eca` |

Training time varies with machine load (the same model took 257 to 615 s for LSTM). LSTM seed 0 ran earlier; LSTM seed 1 had been stopped once by request and was rerun. The last `train-all` invocation (lstm seed 1 to transformer seed 2) took 5011 s.

### Follow-up 1 results (superseded by follow-up 2 below; kept for the record)

- **Early warning (primary):** all learned models detect every gradual episode (miss rate 0.000) with about 0 false alarms per hour. Mean lead at +30 s: rf 283 s, xgb 289, lstm 296, gru 293, transformer 294 (threshold 226, persistence 229).
- **Paired vs threshold (+30 s):** every learned model warns earlier than the threshold baseline, with the CI excluding zero in 3/3 seeds: lead gain 57 s (rf), 63 (xgb), 70 (lstm), 67 (gru), 68 (transformer). Persistence is not different from threshold (+3 s, CI [-11, 15]). Macro F1 vs threshold at +30 s: rf +0.042 and gru +0.046 (CI excludes zero), xgb +0.027, lstm +0.035, transformer +0.041 (CI includes zero).
- **Paired best sequence vs best tree:** +10 s transformer - xgb: lead +13 s [4, 22] (CI excludes zero in 2/3 seeds), macro F1 +0.005 (no). +30 s lstm - xgb: lead +7 s [-4, 17] (no), macro F1 +0.008 (no). Sequence models are not clearly better than the trees.
- **+10 s macro F1** is below the threshold baseline for every learned model (-0.04 for rf, about -0.10 for the others). Models warn earlier than the simulator's WARNING label, which the per-window metric counts as errors; FAULT recall at +10 s is low (0.03 to 0.14).

### Follow-up 1 learning curves (epoch 16: not clearly flat)

- The three runs that reached the cap (lstm 0, gru 0, transformer 1) had their best epoch at 15, 16 and 15, and their validation EMA was still rising (about +0.001 to +0.008 per epoch over the last epochs, after the LR drops). A larger cap might gain a little; this was not tested.
- Six of nine runs stopped on patience at 10 to 15 epochs, mostly right after `min_epochs`. Four of them (lstm 2, gru 2, transformer 0, transformer 2) stopped at 10 to 11 epochs with the learning rate still at its initial value (no plateau drop yet), so they did not get the annealing phase that the longer runs got. Their EMAs were noisy and flat rather than converged.
- Seed-to-seed spread is large (transformer +10 s macro F1 ±0.030, train time ±256 s), which is partly this difference in stopping point. Epoch-count variance is therefore a limitation of the sequence-model results.

### Follow-up 1 changes vs the plain early-stopping configuration

Epochs rose from 9.3 / 6.7 / 8.3 to 14.0 / 12.3 / 12.0 (lstm / gru / transformer) and train time from 228 / 328 / 674 s to 397 / 503 / 923 s. Macro F1 at +10 s rose (lstm 0.646 to 0.666, gru 0.636 to 0.666, transformer 0.652 to 0.662). At +30 s and in lead times the changes are within the seed spread (lead 294 to 296, 296 to 293, 296 to 294 s).

### Follow-up 1 verification

`uv run poe lint` clean (ruff, mypy, ESLint, tsc). `uv run poe test`: 703 passed, 50 skipped (services not running: TimescaleDB, MQTT); frontend 163 passed. `uv run poe minimal-data-check`: 14 passed.

### Known limitations / decisions (follow-up 1; the first one is addressed by follow-up 2)

- Early stopping interacts with `min_epochs: 10`: some seeds stop before the first LR drop (see above). Options for later: raise `min_epochs`, or compare at a fixed 16 epochs. Not done, to keep to the planned settings.
- `E1_clean.md` is regenerated from MLflow; it contains the older runs only in the changes table.
- Decision: results were not tuned further after seeing the test metrics.


### Follow-up 2 (2026-11-15): fixed 22-epoch budget for the sequence models

**Decision, made from validation curves only (ADR 0020, amendment 2026-11-15).** In follow-up 1, several runs were still improving at the 16-epoch cap, while others stopped at 10 to 11 epochs on patience before any learning-rate reduction. Seeds therefore had different schedules. Every LSTM, GRU and Transformer run now trains exactly 22 epochs with the same schedule, with no early stopping, and keeps the epoch with the best EMA of the validation macro F1. The 22-epoch figure and the schedule were not tuned on test results.

- **Learning-rate schedule:** cosine decay from `lr` to `min_lr` 1e-5 over the 22 epochs, instead of reduce-on-plateau. Reduce-on-plateau cannot give every run a comparable annealing phase, because its drops depend on each run's validation loss. This was my choice among the two options offered.
- **Code:** `EarlyStopping.patience` may be null (fixed budget), new `checkpoint_by: raw | ema`, new `lr_schedule.type: cosine` (`config.py`, `train/early_stopping.py`, `models/nn.py`). Two unit tests added (fixed budget never stops early; EMA checkpoint ignores a lucky spike). The old options remain available.
- **Config hashes changed**, so `train-all --skip-existing` reran all nine runs. Last invocation: 8768 s.

| Run | Best epoch | Train time | MLflow run |
|---|---|---|---|
| lstm 0 | 22 | 438.9 s | `40119be6067546d480bf2ea4220b239c` |
| lstm 1 | 22 | 498.0 s | `340bf53036264014aa52c6db90efb665` |
| lstm 2 | 22 | 497.9 s | `a628361bc6eb4083935a198e99448830` |
| gru 0 | 20 | 723.6 s | `f4feb13aa3be433dbdf43c0fa11db346` |
| gru 1 | 22 | 713.5 s | `88ebaaa9ff9d4f60a49c396442b75df5` |
| gru 2 | 22 | 723.4 s | `a0b3d4a2c5294acfafd77c49fe9aa9ef` |
| transformer 0 | 21 | 1662.7 s | `68f4408b910b49c698c7f2a99fd83f26` |
| transformer 1 | 18 | 1692.2 s | `700352634d51403396f2d68e8f847c1a` |
| transformer 2 | 22 | 1650.7 s | `a493b55342b34fafa2de95e7ae17209b` |

**Did the seed-to-seed spread shrink?** Yes for most metrics, clearly for lead and +10 s macro F1, barely for +30 s macro F1 (std over 3 seeds, follow-up 1 to follow-up 2):

| Metric | LSTM | GRU | Transformer |
|---|---|---|---|
| Lead mean +30 s (s) | 5 to 3 | 7 to 1 | 10 to 3 |
| Macro F1 +10 s | 0.016 to 0.005 | 0.026 to 0.007 | 0.030 to 0.002 |
| Macro F1 +30 s | 0.010 to 0.008 | 0.017 to 0.015 | 0.015 to 0.013 |
| Fault recall +30 s | 0.034 to 0.023 | 0.091 to 0.045 | 0.033 to 0.035 |
| Train time (s) | 191 to 34 | 142 to 6 | 256 to 21 |

The remaining +30 s spread is probably genuine seed variance (initialisation, batch order), no longer schedule variance. With 3 seeds these stds are rough.

**Have the curves flattened at 22 epochs?** Mostly, but not fully. The learning rate is annealed to about 1e-5 to 2e-5 at epoch 22 in every run, so late gains are small. The validation EMA still rose by 0.001 to 0.006 per epoch over the last five epochs in all nine runs, and six of nine runs (LSTM 0, 1, 2, GRU 1, 2, Transformer 2) kept their last epoch as the best. A longer cosine budget could gain a little more. This was not tested.

**Did any conclusion change?** (E1 report, 3 seeds; test numbers are reported, not used to choose the budget)

- **Unchanged:**
  - every learned model still detects all 26 gradual episodes, with about 0 false alarms per hour;
  - every learned model warns earlier than the threshold baseline, with the CI excluding zero in 3/3 seeds (+30 s lead: rf 57 s, xgb 63, lstm 58, gru 66, transformer 61);
  - persistence is not different from the threshold baseline;
  - at +30 s no sequence model is clearly better than the best tree (gru - xgb: lead +3 s [-10, 16], macro F1 +0.014 [-0.034, 0.049]).
- **Changed:**
  - LSTM and Transformer lose lead relative to follow-up 1 (+30 s lead 296 to 284 s and 294 to 286 s; versus threshold WARNING 65 to 53 s and 63 to 56 s), and fall slightly below xgb (289 s). GRU keeps its lead (292 s), and is now the best sequence model at both horizons;
  - fault recall at +30 s is lower for all three (LSTM 0.344 to 0.320, GRU 0.399 to 0.344, Transformer 0.406 to 0.338);
  - at +10 s, GRU - xgb is now positive: lead +7 s [0, 14] (CI excludes zero in 2/3 seeds), macro F1 +0.045 [0.006, 0.086] (2/3 seeds). In follow-up 1 the best sequence model (Transformer) was not significantly better on macro F1;
  - macro F1 at +10 s rose for LSTM and GRU (0.666 to 0.704 and 0.701), is still below the threshold baseline, and by less (about -0.055 instead of -0.093). Transformer stayed at 0.669.
- **Summary:** sequence models are comparable to the tree models for early warning, with GRU slightly ahead at +10 s. The fixed budget removed most of the schedule-driven noise but did not make the sequence models better than XGBoost or the Random Forest, and the longer, uniform training lowered LSTM and Transformer lead times a little.

**Cost:** training takes 478 s (LSTM), 720 s (GRU) and 1669 s (Transformer) per seed on this machine, versus 397 / 503 / 923 s before.

**Verification:** `uv run poe lint` clean; `uv run poe test` 705 passed, 50 skipped (services not running); frontend 163 passed.

Suggested commit message: `Phase 7 follow-ups: fixed 22-epoch budget with cosine schedule for sequence models, EMA checkpoint, learning curves, paired bootstrap comparisons, regenerated E1`

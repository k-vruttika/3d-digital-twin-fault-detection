# ADR 0023: E4 condition shift: run selection by load band, test sets and the shift effect

- **Status:** Accepted
- **Date:** 2026-11-15
- **Phase:** 8 (robustness, calibration and conformal prediction)

## Context

E4 asks how a model behaves when the machine runs under operating conditions it never saw in training. The simulator draws one load band per run (low 30–50 %, medium 50–75 %, high 75–95 % of rated load; `config/line.yaml`). It records that band in the manifest (`run_entries[].load_range`) and as a constant Parquet column, which is never a model input. The fixed E1 split (seed 2026) is not stratified by band. Its test split holds only 7 high-load runs.

## Decision

### Run selection (`config/models/dataset_e4.yaml`, `run_filter`)

- **Keep the fixed split.** The E1 split stays as it is, and the run filter is applied on top (`data.build.select_runs`):

| Split | Runs |
|---|---|
| train | the low + medium runs of the E1 train split |
| val | the low + medium runs of the E1 val split |
| test (`e4_high`) | every high-load run, from all three E1 splits; none is trained or validated on |
| test_in (`e4_in`) | the low + medium runs of the E1 test split, as the in-distribution reference |

- **Counts:** from the built dataset's `meta.json` (`run_filter`, `counts`; dataset key `0b2beacae27b45d3`). The report prints them too.

| Split | Runs | By band | Windows | Gradual episodes |
|---|---|---|---|---|
| train | 93 | low 53, medium 40 | 89,560 | 90 |
| val | 22 | low 7, medium 15 | 62,392 | 21 |
| test (`e4_high`) | 62 | high 62 (47 from the E1 train split, 8 from val, 7 from test) | 878,168 | 62 |
| test_in (`e4_in`) | 23 | low 12, medium 11 | 325,772 | 20 |
- **Retrained on the filtered data:** the normaliser, `vib_norm`'s reference speeds and the class weights are refitted on the filtered training runs. Data-driven preprocessing is part of training, and the shifted test data is never seen.
- **Settings unchanged:** every model and baseline is retrained with its E1 config, using the same seeds (0, 1, 2).
- **Calibration and conformal predictors** are fitted on the low + medium validation runs. Whether their coverage survives the shift is part of the result.

### Why all 62 high-load runs (decided with the user before any E4 result)

- **More episodes:** the 7 high-load test runs would give very few gradual-fault episodes and very wide CIs.
- **Never seen:** no high-load run is used by E4 training or validation, so all of them are valid test runs for the E4 models.

### What the report compares

| Comparison | Status | Method |
|---|---|---|
| `e4_high`: each E4 model vs the threshold baseline | primary (ADR 0021) | paired, same runs |
| Shift effect: the same E4 model on `e4_high` minus on `e4_in` | exploratory | Same model and seed on both sides, so the smaller training set (fewer runs than E1) does not confound the shift. The run sets differ, so each side is resampled independently (unpaired run bootstrap). |
| Secondary check: E4 model minus E1 model on the high-load runs of the original test split | exploratory | Paired, with its episode count. The E1 models trained on 140 runs including high load, so this difference mixes the shift with the training-set size. It is reported for context only. |

## Consequences

- **The shift is in load only.** Load changes temperature, current and vibration levels and the degradation speed. Ambient temperature and CNC speed changes still vary as in E1.
- **Small numbers:** E4 has fewer training runs than E1. The in-distribution reference `e4_in` has about 23 runs, so its CIs are wide.
- **New dataset key:** the filtered dataset has its own `dataset_key`. The E1 key is unchanged, because unset optional fields stay out of the fingerprint (tested).

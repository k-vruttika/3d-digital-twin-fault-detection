# ADR 0028: Channel schema: datasets with their own sensors, kinds and time step

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** 10 (validation on real public data)

## Context

Phases 7–9 built every model input from module constants in `twin_ml.features.inputs`: five sensors (speed, temp, vib, load, current), the engineered `vib_norm`, the three simulator machine kinds, and one row per second. Window, horizon, stride and sustain lengths were given in seconds and used as row counts. Phase 10 trains the same models on MetroPT-3 (15 signals, one compressor, a 1-minute grid) and Azure PdM (4 sensors, four machine models, an hourly grid). Copying the model code per dataset would split the code base. Changing the constants would change the Phase 7–9 dataset keys and study hashes, and every stored run would stop matching.

## Decision

- **`ChannelSchema`** (`twin_ml/features/schema.py`) describes one dataset:
  - `sensors`: the raw input columns, in order;
  - `vib_norm`: whether the engineered channel is appended (needs speed and vib);
  - `kinds`: machine kind names, in one-hot order;
  - `step_s`: seconds between rows.

  It derives the value, mask, kind and sequence channel names, and maps value channels to sensors.
- **`SIM_SCHEMA`** equals the Phase 7–9 constants. The constants stay, derived from it, so every existing import keeps working.
- **Where the schema lives:** a built dataset records its schema in `meta.json` (`schema`). A dataset without that key is a simulator dataset (`ChannelSchema.of_meta`). `Dataset.schema` and `SplitData.schema` / `SplitData.step_s` expose it.
- **Who reads it:** everything that sized itself from the constants now takes the split's (or an explicit) schema, with `SIM_SCHEMA` as the default:
  - `SplitData.gather` / `sequence` / `summary` (kind one-hot width);
  - `Normalizer` (the speed reference only when `vib_norm` is on);
  - `summary_features` / `summary_names`;
  - the sequence models' input width; the autoencoder input and per-sensor error map;
  - `AnomalyThresholds` (number of kinds);
  - the explanation maps (`aggregate`, the IG baseline, `explain()`);
  - `perturb.normal_std` / `perturb_series`.
- **Rows vs seconds:**
  - Real-data builds record window, horizons and strides in rows under `meta.json` `rows`, with the time-to-fault cap in seconds. `Dataset.window`, `horizons`, `stride()` and `ttf_cap_s` read it when present.
  - The early-warning code converts with `step_s`: `sustain_windows(sustain_s, stride × step_s)` and negative time = windows × stride × step_s.
  - `step_s` is 1 for simulator data, so the arithmetic is unchanged there.
- **Time precision:** real-data `t_s` is stored as float64 (seconds from the dataset origin, up to 3 × 10⁷ s). Simulator builds keep float32.
- **Phase 7–9 results stay valid:**
  - simulator `meta.json` and `Normalizer.to_dict()` output are byte-for-byte as before (no `schema` key);
  - no field was added to a hashed config class;
  - `ml/tests/test_ml_schema.py` pins the fingerprints of `dataset.yaml`, `dataset_e4.yaml` and `dataset_aug.yaml`;
  - after the change, `robustness --dry-run --skip-existing` (all seven studies) and `e5 --dry-run --skip-existing` (all four studies) still show 0 runs to do.

## Consequences

- New datasets need a loader that writes the common layout plus its schema. No model, detector or evaluation code changes.
- Code that indexes sensors by name (`vib_norm`, the simulator threshold baseline, the sim sensor-fault labels, `signal_scale`) remains simulator-only, as before.
- Phase 11 live inference must load the schema from the model's dataset meta, not assume the simulator one.

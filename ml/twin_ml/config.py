"""Pydantic models for the ML configs in ``config/models/*.yaml`` (no heavy imports)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from twin_common.config.line import ConfigModel
from twin_common.config.loader import ConfigError, format_validation_error, load_model, read_yaml
from twin_common.enums import MachineState

SPLITS = ("train", "val", "test")


class SplitConfig(ConfigModel):
    fractions: dict[Literal["train", "val", "test"], float]
    seed: int

    @model_validator(mode="after")
    def _fractions(self) -> SplitConfig:
        if set(self.fractions) != set(SPLITS):
            raise ValueError(f"fractions must name exactly {list(SPLITS)}")
        if any(f <= 0 for f in self.fractions.values()):
            raise ValueError("every split fraction must be > 0")
        if abs(sum(self.fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class StrideConfig(ConfigModel):
    train: int = Field(gt=0)
    val: int = Field(gt=0)
    test: int = Field(gt=0)

    def of(self, split: str) -> int:
        return int(getattr(self, split))


class FeatureConfig(ConfigModel):
    min_speed_ratio: float = Field(gt=0, le=1)


PerturbationKind = Literal["noise", "mcar", "dropout", "stuck"]
PERTURBATION_KINDS: tuple[PerturbationKind, ...] = ("noise", "mcar", "dropout", "stuck")


class PerturbationGrid(ConfigModel):
    """Levels of one perturbation type in percent.

    noise: Gaussian sigma as % of the sensor's normal-operation std; mcar: % of sensor values
    removed; dropout / stuck: % of sensor-time covered by segments of ``duration_s``.
    """

    levels_pct: list[int] = Field(min_length=1)
    duration_s: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _check(self) -> PerturbationGrid:
        if any(lv <= 0 or lv >= 100 for lv in self.levels_pct):
            raise ValueError("levels_pct must lie in (0, 100)")
        if self.levels_pct != sorted(set(self.levels_pct)):
            raise ValueError("levels_pct must be unique and ascending")
        if self.duration_s is not None and not 1 <= self.duration_s[0] <= self.duration_s[1]:
            raise ValueError("duration_s must be [lo, hi] with 1 <= lo <= hi")
        return self


class PerturbationsConfig(ConfigModel):
    noise: PerturbationGrid
    mcar: PerturbationGrid
    dropout: PerturbationGrid
    stuck: PerturbationGrid
    # rows used for the normal-operation std of each sensor and machine kind (training data,
    # no active fault)
    normal_states: list[MachineState] = Field(min_length=1)

    @model_validator(mode="after")
    def _segments(self) -> PerturbationsConfig:
        if self.dropout.duration_s is None or self.stuck.duration_s is None:
            raise ValueError("dropout and stuck need duration_s")
        return self

    def grid(self, kind: PerturbationKind) -> PerturbationGrid:
        out: PerturbationGrid = getattr(self, kind)
        return out


class LoadRangeFilter(ConfigModel):
    """Keep only runs whose simulator load band is listed (manifest ``load_range``)."""

    load_ranges: list[str] = Field(min_length=1)


class RunFilterConfig(ConfigModel):
    """Run selection on top of the fixed split (Phase 8, E4 condition shift; ADR 0023).

    ``train`` / ``val`` restrict their split. ``test`` replaces the test split by every run of
    the listed bands that is in neither the (filtered) train nor val split; ``test_in`` names
    the bands of the original test split kept as the in-distribution reference split.
    """

    train: LoadRangeFilter | None = None
    val: LoadRangeFilter | None = None
    test: LoadRangeFilter | None = None
    test_in: LoadRangeFilter | None = None


class AugmentationConfig(ConfigModel):
    """Training-series augmentation before windowing (Phase 8 ablation; ADR 0021).

    Each training series is replaced, with probability ``p``, by a perturbed copy. The type is
    drawn uniformly from ``perturbations``, the level uniformly from ``[min, max]`` of that
    type's test grid. The perturbation settings are a copy of ``robustness.yaml``'s (a test
    checks they match), so the dataset fingerprint covers them. Validation and test are never
    augmented.
    """

    p: float = Field(gt=0, le=1)
    seed: int
    perturbations: PerturbationsConfig


class DatasetConfig(ConfigModel):
    source_dir: Path
    out_dir: Path
    line_config: Path
    window_s: int = Field(ge=2)
    horizons_s: list[int] = Field(min_length=1)
    ttf_cap_s: float = Field(gt=0)
    strides_s: StrideConfig
    split: SplitConfig
    normal_keep_frac: float = Field(gt=0, le=1)
    subsample_seed: int
    features: FeatureConfig
    # Phase 8 study options; unset fields are left out of the fingerprint, so the E1 dataset
    # key does not change.
    run_filter: RunFilterConfig | None = None
    augmentation: AugmentationConfig | None = None

    @model_validator(mode="after")
    def _horizons(self) -> DatasetConfig:
        if any(h <= 0 for h in self.horizons_s) or self.horizons_s != sorted(set(self.horizons_s)):
            raise ValueError("horizons_s must be positive, unique and ascending")
        return self

    def fingerprint(self) -> str:
        """Hash of every setting that changes the built arrays (paths excluded)."""
        exclude: set[str] = {"source_dir", "out_dir", "line_config"}
        exclude |= {k for k in ("run_filter", "augmentation") if getattr(self, k) is None}
        data = self.model_dump(mode="json", exclude=exclude)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class SubsetConfig(ConfigModel):
    operating: list[MachineState] = Field(min_length=1)
    stopped: list[MachineState] = Field(min_length=1)


class EarlyWarningConfig(ConfigModel):
    sustain_s: int = Field(ge=1)


class LatencyConfig(ConfigModel):
    warmup_calls: int = Field(ge=0)
    timed_calls: int = Field(ge=10)
    num_threads: int = Field(ge=1)


class BootstrapConfig(ConfigModel):
    n_resamples: int = Field(ge=10)
    seed: int
    ci: float = Field(gt=0, lt=1)


class ReportConfig(ConfigModel):
    out_dir: Path
    name: str = Field(min_length=1)


class EvalConfig(ConfigModel):
    experiment: str = Field(min_length=1)
    tracking_dir: Path
    seeds: list[int] = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    subsets: SubsetConfig
    early_warning: EarlyWarningConfig
    latency: LatencyConfig
    bootstrap: BootstrapConfig
    report: ReportConfig


# --------------------------------------------------------------------------- models


class ClassWeightConfig(ConfigModel):
    power: float = Field(ge=0, le=1)  # 0 = no class weighting, 1 = full inverse frequency
    max: float = Field(ge=1)


class PersistenceConfig(ConfigModel):
    type: Literal["persistence"]


class ThresholdConfig(ConfigModel):
    type: Literal["threshold"]
    sim_config: Path


class RfParams(ConfigModel):
    n_estimators: int = Field(ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_leaf: int = Field(ge=1)
    max_features: float | Literal["sqrt", "log2"] | None = None


class RfConfig(ConfigModel):
    type: Literal["rf"]
    class_weight: ClassWeightConfig
    classifier: RfParams
    regressor: RfParams
    n_jobs: int = -1


class XgbParams(ConfigModel):
    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    min_child_weight: float = Field(ge=0)


class XgbConfig(ConfigModel):
    type: Literal["xgb"]
    class_weight: ClassWeightConfig
    classifier: XgbParams
    regressor: XgbParams
    early_stopping_rounds: int = Field(ge=1)
    n_jobs: int = -1


class NnArch(ConfigModel):
    hidden: int = Field(ge=4)  # LSTM/GRU hidden size, Transformer d_model
    layers: int = Field(ge=1, le=4)
    dropout: float = Field(ge=0, lt=1)
    heads: int = Field(default=4, ge=1)  # Transformer only
    ff_mult: int = Field(default=2, ge=1)  # Transformer feed-forward width = ff_mult * hidden

    @model_validator(mode="after")
    def _heads(self) -> NnArch:
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        return self


class LossWeights(ConfigModel):
    cls: float = Field(ge=0)
    ttf: float = Field(ge=0)
    fault_type: float = Field(ge=0)


class EarlyStopping(ConfigModel):
    """Training budget and checkpoint rule for the sequence models.

    The score is the validation macro F1 (operating windows, mean over horizons), smoothed
    with an exponential moving average (EMA). ``patience: null`` means a fixed budget: always
    train ``max_epochs`` epochs, never stop early. ``checkpoint_by`` picks the epoch whose
    weights are kept: the best raw score or the best EMA.
    """

    min_epochs: int = Field(default=1, ge=1)  # never stop on patience before this many epochs
    max_epochs: int = Field(ge=1)  # hard cap (the fixed budget when patience is null)
    patience: int | None = Field(default=None, ge=1)  # epochs without an EMA gain > min_delta
    ema_alpha: float = Field(gt=0, le=1)  # weight of the newest epoch; 1 = no smoothing
    min_delta: float = Field(default=0.0, ge=0)
    checkpoint_by: Literal["raw", "ema"] = "raw"

    @model_validator(mode="after")
    def _order(self) -> EarlyStopping:
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs must be <= max_epochs")
        return self


class LrSchedule(ConfigModel):
    """Learning-rate schedule. ``none``: constant; ``plateau``: reduce-on-plateau on the
    validation loss (the annealing phase differs per run); ``cosine``: cosine decay from
    ``lr`` to ``min_lr`` over ``max_epochs`` (identical for every run)."""

    type: Literal["none", "plateau", "cosine"]
    factor: float = Field(default=0.5, gt=0, lt=1)
    patience: int = Field(default=1, ge=0)
    min_lr: float = Field(default=1e-5, ge=0)


class NnTraining(ConfigModel):
    early_stopping: EarlyStopping
    lr_schedule: LrSchedule
    batch_size: int = Field(ge=1)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    focal_gamma: float = Field(ge=0)  # 0 = plain weighted cross-entropy
    huber_delta: float = Field(gt=0)  # on ttf / cap
    loss_weights: LossWeights
    eval_batch_size: int = Field(ge=1)
    num_threads: int | None = Field(default=None, ge=1)  # None = torch default


class NnConfig(ConfigModel):
    type: Literal["lstm", "gru", "transformer"]
    class_weight: ClassWeightConfig
    arch: NnArch
    training: NnTraining


ModelConfig = Annotated[
    PersistenceConfig | ThresholdConfig | RfConfig | XgbConfig | NnConfig,
    Field(discriminator="type"),
]
_MODEL_ADAPTER: TypeAdapter[ModelConfig] = TypeAdapter(ModelConfig)


# --------------------------------------------------------------------------- Phase 8


class SensorFaultSet(ConfigModel):
    """A simulator-driven test set: the test runs regenerated with sensor faults (same seed)."""

    source_dir: Path
    scenario: Path
    types: list[Literal["stuck", "drift", "dropout"]] = Field(min_length=1)


class CalibrationConfig(ConfigModel):
    tree_methods: list[Literal["isotonic", "platt"]] = Field(min_length=1)
    cv_folds: int = Field(ge=2)  # run-grouped folds of the validation split for the selection
    temperature_bounds: tuple[float, float]  # search range of T for the sequence models


class ConformalConfig(ConfigModel):
    alpha: float = Field(gt=0, lt=1)
    methods: list[Literal["lac", "aps", "mondrian_lac"]] = Field(min_length=1)
    aps_seed: int  # uniform draws of randomised APS


class PrimaryConfig(ConfigModel):
    """Pre-registered primary outcomes (ADR 0021); everything else is exploratory."""

    horizon: int
    reference: str  # early-warning outcomes are paired against this baseline
    early_warning: list[str] = Field(min_length=1)
    calibration: list[str] = Field(min_length=1)
    conformal: list[str] = Field(min_length=1)
    conditions: list[str] = Field(min_length=1)


class StudyConfig(ConfigModel):
    """One study: where its models come from and which test conditions it evaluates."""

    models_from: Literal["e1", "train"]
    dataset_config: Path
    experiment: str = Field(min_length=1)  # MLflow experiment of the study's training runs
    conditions: list[str] = Field(min_length=1)  # "perturbations" expands to the full grid
    # exploratory extension study added after the pre-registration: kept out of the other
    # studies' config hashes (their runs stay valid) and labelled exploratory in the report
    extension: bool = False


class SignalNoiseConfig(ConfigModel):
    """Exploratory signal-relative noise grid (Part C): Gaussian sigma per machine kind and
    sensor = p % of that sensor's typical rise before FAULT (clean training episodes)."""

    levels_pct: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _levels(self) -> SignalNoiseConfig:
        if any(lv <= 0 for lv in self.levels_pct) or self.levels_pct != sorted(
            set(self.levels_pct)
        ):
            raise ValueError("levels_pct must be positive, unique and ascending")
        return self


class StdNoiseConfig(ConfigModel):
    """Exploratory uniform noise grid (Part C follow-up): Gaussian sigma per machine kind and
    sensor = m x that sensor's normal-operation std, the same multiple for every sensor."""

    multiples: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _multiples(self) -> StdNoiseConfig:
        if any(m <= 0 for m in self.multiples) or self.multiples != sorted(set(self.multiples)):
            raise ValueError("multiples must be positive, unique and ascending")
        return self


class RobustnessConfig(ConfigModel):
    experiment: str = Field(min_length=1)  # MLflow experiment of the study runs
    phase_tag: str = Field(min_length=1)
    source_experiment: str = Field(min_length=1)  # E1 runs whose models are reused
    perturbations: PerturbationsConfig
    perturbation_seed: int
    sensor_fault_sets: dict[str, SensorFaultSet]
    ece_bins: int = Field(ge=2)
    eval_subset: Literal["operating", "all"]  # windows used for calibration and conformal
    calibration: CalibrationConfig
    conformal: ConformalConfig
    primary: PrimaryConfig
    studies: dict[str, StudyConfig]
    seen_in_augmentation: list[str]  # perturbation kinds / sensor fault types seen in training
    reliability_conditions: list[str] = Field(min_length=1)
    reliability_min_count: int = Field(ge=1)  # plots only; the ECE uses every bin
    eta_train_s: dict[str, float]  # Phase 7 measured train time per (model, seed), for the ETA
    eta_condition_s: dict[str, float]  # rough evaluation time per test condition
    progress_dir: Path
    report_name: str = Field(min_length=1)
    signal_noise: SignalNoiseConfig | None = None  # exploratory extension (Part C)
    std_noise: StdNoiseConfig | None = None  # exploratory extension (Part C follow-up)


# --------------------------------------------------------------------------- Phase 9


class IForestConfig(ConfigModel):
    type: Literal["iforest"]
    n_estimators: int = Field(ge=1)
    max_samples: int = Field(ge=2)
    max_features: float = Field(gt=0, le=1)
    n_jobs: int = -1


class AutoencoderConfig(ConfigModel):
    type: Literal["autoencoder"]
    hidden: int = Field(ge=2)
    latent: int = Field(ge=1)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    eval_batch_size: int = Field(ge=1)
    num_threads: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _bottleneck(self) -> AutoencoderConfig:
        if self.latent >= self.hidden:
            raise ValueError("latent must be smaller than hidden")
        return self


# Kept apart from ModelConfig: detectors are never trained by train-all / the E1 trainer.
AnomalyModelConfig = Annotated[IForestConfig | AutoencoderConfig, Field(discriminator="type")]
_ANOMALY_ADAPTER: TypeAdapter[AnomalyModelConfig] = TypeAdapter(AnomalyModelConfig)


class E5DataConfig(ConfigModel):
    source_dir: Path
    scenario: Path
    base_seed: int
    generate_runs: int = Field(ge=1)
    test_split_runs: bool
    extra_run_indices: tuple[int, int]  # half-open range of fresh run indices
    fault_type: str = Field(min_length=1)
    n_runs: int = Field(ge=1)

    @model_validator(mode="after")
    def _range(self) -> E5DataConfig:
        lo, hi = self.extra_run_indices
        if not 0 <= lo <= hi <= self.generate_runs:
            raise ValueError("extra_run_indices must lie inside 0..generate_runs")
        return self


class AnomalySettings(ConfigModel):
    normal_states: list[MachineState] = Field(min_length=1)
    threshold_percentiles: list[float] = Field(min_length=1)
    primary_percentile: float
    per_kind: bool
    models: dict[str, Path] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _percentiles(self) -> AnomalySettings:
        p = self.threshold_percentiles
        if p != sorted(set(p)) or not all(0 < x < 100 for x in p):
            raise ValueError("threshold_percentiles must be unique, ascending, in (0, 100)")
        if self.primary_percentile not in p:
            raise ValueError("primary_percentile must be one of threshold_percentiles")
        return self


class CombinedConfig(ConfigModel):
    supervised: list[str] = Field(min_length=1)
    detectors: list[str] = Field(min_length=1)
    percentiles: list[float] = Field(min_length=1)


class ConfidenceConfig(ConfigModel):
    high: float = Field(gt=0, lt=1)


class ExplainConfig(ConfigModel):
    horizon: int
    window_stride_s: int = Field(ge=1)
    max_windows_per_episode: int = Field(ge=1)
    ig_steps: int = Field(ge=2)
    deletion_seed: int
    primary_models: list[str] = Field(min_length=1)
    exploratory_models: list[str]
    expected_sensor: dict[str, str] = Field(min_length=1)


class E5PrimaryConfig(ConfigModel):
    """Pre-registered primary outcomes of E5 (ADR 0024); the test checks them against the ADR."""

    horizon: int
    reference: str
    early_warning: dict[str, str] = Field(min_length=1)  # outcome -> condition
    groups: list[str] = Field(min_length=1)
    anomaly_percentile: float
    conformal: list[str] = Field(min_length=1)
    conformal_conditions: list[str] = Field(min_length=2)
    explanation: list[str] = Field(min_length=1)
    explanation_models: list[str] = Field(min_length=1)


class E5StudyConfig(ConfigModel):
    kind: Literal["supervised", "anomaly", "mixed"]
    dataset_config: Path
    source_experiment: str | None = None  # where reused supervised models are found
    source_phase: str | None = None
    models: list[str] = Field(min_length=1)
    conditions: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _source(self) -> E5StudyConfig:
        if self.kind != "anomaly" and not (self.source_experiment and self.source_phase):
            raise ValueError("supervised studies need source_experiment and source_phase")
        return self


class E5Config(ConfigModel):
    experiment: str = Field(min_length=1)
    smoke_experiment: str = Field(min_length=1)
    phase_tag: str = Field(min_length=1)
    robustness_config: Path
    e5_data: E5DataConfig
    anomaly: AnomalySettings
    combined: CombinedConfig
    confidence: ConfidenceConfig
    explain: ExplainConfig
    primary: E5PrimaryConfig
    studies: dict[str, E5StudyConfig] = Field(min_length=1)
    eta_train_s: dict[str, float]
    eta_condition_s: dict[str, float]
    progress_dir: Path
    report_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> E5Config:
        if self.primary.anomaly_percentile != self.anomaly.primary_percentile:
            raise ValueError("primary.anomaly_percentile must equal anomaly.primary_percentile")
        if self.anomaly.primary_percentile not in self.combined.percentiles:
            raise ValueError("combined.percentiles must include the primary percentile")
        if set(self.combined.detectors) - set(self.anomaly.models):
            raise ValueError("combined.detectors must be anomaly models")
        return self


def load_e5_config(path: str | Path) -> E5Config:
    return load_model(path, E5Config)


class AeSharesConfig(ConfigModel):
    """Exploratory: the autoencoder's per-sensor reconstruction-error shares (ADR 0024)."""

    detector: Literal["autoencoder"]
    source_study: str = Field(min_length=1)  # the E5 study whose stored alarms must reproduce
    precursor_conditions: list[str] = Field(min_length=1)  # episodes, expected sensor per type
    drift_condition: str = Field(min_length=1)  # drift segments, expected = the drifting sensor


class IgSensitivityConfig(ConfigModel):
    """Exploratory: Integrated Gradients at another step count on the same windows."""

    models: list[str] = Field(min_length=1)
    ig_steps: int = Field(ge=2)


class E5FollowupConfig(ConfigModel):
    """Exploratory Phase 9 Part C follow-ups, added after the E5 results were seen; kept out of
    ``e5.yaml`` so the pre-registered settings and every earlier study hash stay unchanged."""

    ae_shares: AeSharesConfig
    ig_sensitivity: IgSensitivityConfig


def load_e5_followup_config(path: str | Path) -> E5FollowupConfig:
    return load_model(path, E5FollowupConfig)


def parse_anomaly_config(data: object, source: str = "<anomaly config>") -> AnomalyModelConfig:
    try:
        return _ANOMALY_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(source, exc)) from exc


def load_anomaly_config(path: str | Path) -> AnomalyModelConfig:
    p = Path(path)
    return parse_anomaly_config(read_yaml(p), str(p))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return load_model(path, DatasetConfig)


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_model(path, EvalConfig)


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    return load_model(path, RobustnessConfig)


def parse_model_config(data: object, source: str = "<model config>") -> ModelConfig:
    """Validate a model config given as plain data (e.g. from ``model_dump(mode="json")``)."""
    try:
        return _MODEL_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(source, exc)) from exc


def load_model_config(path: str | Path) -> ModelConfig:
    p = Path(path)
    return parse_model_config(read_yaml(p), str(p))

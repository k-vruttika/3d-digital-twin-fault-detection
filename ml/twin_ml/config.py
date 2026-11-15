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

    @model_validator(mode="after")
    def _horizons(self) -> DatasetConfig:
        if any(h <= 0 for h in self.horizons_s) or self.horizons_s != sorted(set(self.horizons_s)):
            raise ValueError("horizons_s must be positive, unique and ascending")
        return self

    def fingerprint(self) -> str:
        """Hash of every setting that changes the built arrays (paths excluded)."""
        data = self.model_dump(mode="json", exclude={"source_dir", "out_dir", "line_config"})
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


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return load_model(path, DatasetConfig)


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_model(path, EvalConfig)


def load_model_config(path: str | Path) -> ModelConfig:
    p = Path(path)
    data = read_yaml(p)
    try:
        return _MODEL_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(str(p), exc)) from exc

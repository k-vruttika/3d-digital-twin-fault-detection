"""Configs of the public datasets (Phase 10, ADR 0027): dataset facts and window builds.

* :class:`DatasetFacts` (``config/datasets/<name>.yaml``): files, sensors, failure list and its
  source. Facts about the data, never model settings.
* :class:`RealDatasetConfig` (``config/models/real_<name>.yaml``): grid, window, horizons,
  label bands and the time-ordered split, all in rows of the dataset's own grid.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.config.loader import load_model

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_time(text: str) -> datetime:
    """Naive timestamp ``YYYY-MM-DD HH:MM:SS`` (both datasets use one naive clock)."""
    return datetime.strptime(text, TIME_FORMAT)


class SensorFact(ConfigModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    column: str
    type: Literal["analog", "digital"]
    unit: str | None = None
    part: str
    description: str


class FailureFact(ConfigModel):
    id: int
    start: str
    end: str
    type: str
    severity: str | None = None
    report: str | None = None

    @model_validator(mode="after")
    def _order(self) -> FailureFact:
        if parse_time(self.end) < parse_time(self.start):
            raise ValueError(f"failure {self.id}: end before start")
        return self


class PdfCheck(ConfigModel):
    date: str
    method: str
    result: str
    inconsistencies: list[str] = Field(default_factory=list)


class DatasetFacts(ConfigModel):
    name: Literal["metropt3", "azure_pdm"]
    title: str
    provider: str
    licence: str
    origin: str
    download: str
    dir: Path
    files: dict[str, str] = Field(min_length=1)
    optional_files: dict[str, str] = Field(default_factory=dict)
    timestamp_column: str
    timestamp_format: str
    time_zone: str
    sensors: list[SensorFact] = Field(min_length=1)
    failure_source: str
    # MetroPT-3
    kind: str | None = None
    machine_id: str | None = None
    failures: list[FailureFact] = Field(default_factory=list)
    pdf_check: PdfCheck | None = None
    # Azure PdM
    machine_column: str | None = None
    kind_column: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sensors)

    @property
    def sensor_parts(self) -> dict[str, str]:
        return {s.name: s.part for s in self.sensors}

    def fingerprint(self) -> str:
        """Hash of the facts that change built arrays (failure list, sensors, columns)."""
        data = self.model_dump(
            mode="json",
            include={
                "name",
                "files",
                "timestamp_column",
                "timestamp_format",
                "sensors",
                "kind",
                "machine_id",
                "failures",
                "machine_column",
                "kind_column",
                "error_codes",
                "components",
            },
        )
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class LabelBands(ConfigModel):
    warning_band: int = Field(ge=1)  # rows before a failure start labelled WARNING
    fault_context: int = Field(ge=1)  # rows before a failure start with fault_type set
    post_failure: int = Field(ge=0)  # rows after a failure end labelled MAINTENANCE

    @model_validator(mode="after")
    def _context(self) -> LabelBands:
        if self.fault_context < self.warning_band:
            raise ValueError("fault_context must cover the warning band")
        return self


class TimeSplit(ConfigModel):
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]

    @model_validator(mode="after")
    def _ordered(self) -> TimeSplit:
        for name in ("train", "val", "test"):
            a, b = getattr(self, name)
            if parse_time(a) >= parse_time(b):
                raise ValueError(f"split {name}: start must be before end")
        if parse_time(self.val[0]) < parse_time(self.train[1]) or parse_time(
            self.test[0]
        ) < parse_time(self.val[1]):
            raise ValueError("splits must be in time order and must not overlap")
        return self

    def of(self, name: str) -> tuple[datetime, datetime]:
        a, b = getattr(self, name)
        return parse_time(a), parse_time(b)


class StrideRows(ConfigModel):
    train: int = Field(gt=0)
    val: int = Field(gt=0)
    test: int = Field(gt=0)


class MachineHoldout(ConfigModel):
    fraction: float = Field(gt=0, lt=1)
    seed: int


class RealDatasetConfig(ConfigModel):
    facts: Path
    out_dir: Path
    step_s: int = Field(gt=0)
    max_gap_rows: int = Field(ge=0)
    window: int = Field(ge=2)
    horizons: list[int] = Field(min_length=1)
    ttf_cap: int = Field(gt=0)
    strides: StrideRows
    labels: LabelBands
    split: TimeSplit
    normal_keep_frac: float = Field(gt=0, le=1)
    subsample_seed: int
    error_channels: bool = False
    machine_holdout: MachineHoldout | None = None

    @model_validator(mode="after")
    def _horizons(self) -> RealDatasetConfig:
        if any(h <= 0 for h in self.horizons) or self.horizons != sorted(set(self.horizons)):
            raise ValueError("horizons must be positive, unique and ascending")
        return self

    def fingerprint(self, facts: DatasetFacts) -> str:
        """Hash of every setting and fact that changes the built arrays (paths excluded)."""
        data = {
            "config": self.model_dump(mode="json", exclude={"facts", "out_dir"}),
            "facts": facts.fingerprint(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_facts(path: str | Path) -> DatasetFacts:
    return load_model(path, DatasetFacts)


def load_real_dataset_config(path: str | Path) -> RealDatasetConfig:
    return load_model(path, RealDatasetConfig)


# ---------------------------------------------------------------------- real.yaml (Phase 10)


class RealBootstrap(ConfigModel):
    n_resamples: int = Field(ge=10)
    seed: int
    ci: float = Field(gt=0, lt=1)


class RealDatasetStudy(ConfigModel):
    dataset_config: Path
    sustain: int = Field(ge=1)  # rows
    horizon: int | None = None  # primary supervised head (rows); None = no supervised model
    lookbacks: list[int] = Field(min_length=1)  # rows before the failure start
    primary_lookback: int
    evaluable_min_coverage: float = Field(gt=0, le=1)
    supervised: list[str] = Field(default_factory=list)
    detectors: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    zscore_trailing: int = Field(ge=2)  # rows

    @model_validator(mode="after")
    def _primary(self) -> RealDatasetStudy:
        if self.primary_lookback not in self.lookbacks:
            raise ValueError("primary_lookback must be one of lookbacks")
        if self.supervised and self.horizon is None:
            raise ValueError("supervised models need a horizon")
        return self


class RealAnomaly(ConfigModel):
    threshold_percentiles: list[float] = Field(min_length=1)
    primary_percentile: float
    per_kind: bool = True

    @model_validator(mode="after")
    def _primary(self) -> RealAnomaly:
        if self.primary_percentile not in self.threshold_percentiles:
            raise ValueError("primary_percentile must be one of threshold_percentiles")
        return self


class ThresholdRule(ConfigModel):
    low_pct: float = Field(ge=0, lt=50)
    high_pct: float = Field(gt=50, le=100)


class ZScoreRule(ConfigModel):
    z: float = Field(gt=0)
    min_std: float = Field(gt=0)


class RealBaselines(ConfigModel):
    threshold: ThresholdRule
    zscore: ZScoreRule
    persistence: Literal["observable_only"]


class SelectionRule(ConfigModel):
    metric: Literal["pr_auc_mean"]
    subset: Literal["operating"]
    horizon: int
    split: Literal["val"]


class CombinedRule(ConfigModel):
    primary: list[str] = Field(min_length=2, max_length=2)
    exploratory_detectors: list[str] = Field(min_length=1)


class ErrorRule(ConfigModel):
    codes: list[str] = Field(min_length=1)


class ErrorXgb(ConfigModel):
    dataset_config: Path
    model: Literal["xgb"]


class HoldoutVariant(ConfigModel):
    dataset_config: Path
    fraction: float = Field(gt=0, lt=1)
    seed: int
    models: list[str] = Field(min_length=1)


class RobustnessSubset(ConfigModel):
    conditions: list[str] = Field(min_length=1)
    models: list[str] = Field(min_length=1)


class RealExplain(ConfigModel):
    window_stride: int = Field(ge=1)
    metropt_window_stride: int = Field(ge=1)
    max_windows_per_episode: int = Field(ge=1)
    ig_steps: int = Field(ge=2)
    air_leak_plausible: list[str] = Field(min_length=1)


class RealExploratory(ConfigModel):
    error_rule: ErrorRule
    error_xgb: ErrorXgb
    holdout: HoldoutVariant
    robustness: RobustnessSubset
    fa_by_month: list[str]
    explain: RealExplain


class AzurePrimary(ConfigModel):
    horizon: int
    lookback: int
    outcomes: list[str]
    groups: list[str]
    reference: str
    combined_vs: str
    anomaly_percentile: float


class MetroPrimary(ConfigModel):
    lookback: int
    outcomes: list[str]
    groups: list[str]
    anomaly_percentile: float
    evaluable_min_coverage: float


class RealPrimary(ConfigModel):
    azure_pdm: AzurePrimary
    metropt3: MetroPrimary


class RealConfig(ConfigModel):
    experiment: str
    smoke_experiment: str
    phase_tag: str
    eval_config: Path
    robustness_config: Path
    seeds: list[int] = Field(min_length=1)
    bootstrap: RealBootstrap
    datasets: dict[Literal["metropt3", "azure_pdm"], RealDatasetStudy]
    models_dir: Path
    anomaly: RealAnomaly
    baselines: RealBaselines
    selection: SelectionRule
    combined: CombinedRule
    exploratory: RealExploratory
    primary: RealPrimary
    progress_dir: Path = Path("experiments/progress")  # runner progress logs (gitignored)

    @model_validator(mode="after")
    def _consistent(self) -> RealConfig:
        az, mp = self.datasets["azure_pdm"], self.datasets["metropt3"]
        p = self.primary
        if p.azure_pdm.horizon != az.horizon or p.azure_pdm.lookback != az.primary_lookback:
            raise ValueError("primary.azure_pdm must match datasets.azure_pdm")
        if p.metropt3.lookback != mp.primary_lookback:
            raise ValueError("primary.metropt3.lookback must match datasets.metropt3")
        if p.metropt3.evaluable_min_coverage != mp.evaluable_min_coverage:
            raise ValueError("primary.metropt3.evaluable_min_coverage must match the dataset")
        for pp in (p.azure_pdm.anomaly_percentile, p.metropt3.anomaly_percentile):
            if pp != self.anomaly.primary_percentile:
                raise ValueError("primary anomaly percentile must equal anomaly.primary_percentile")
        if self.selection.horizon != az.horizon:
            raise ValueError("selection.horizon must be the primary Azure horizon")
        return self


def load_real_config(path: str | Path) -> RealConfig:
    return load_model(path, RealConfig)

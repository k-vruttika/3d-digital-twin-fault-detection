"""Schema for ``config/inference.yaml``: live inference in the twin (Phase 11, ADR 0030).

The backend never imports ``twin_ml``: it starts inference workers
(``python -m twin_ml.serve worker``) as child processes. :meth:`InferenceConfig.worker_plans`
decides which processes run: a sequence model and the autoencoder share one torch process;
an XGBoost model gets its own process next to a torch monitor process (ADR 0020).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from twin_common.config.line import ConfigModel
from twin_common.config.loader import load_model
from twin_common.enums import MachineState


class InferenceModelEntry(ConfigModel):
    registered: str = Field(min_length=1, description="MLflow registered model name")
    family: Literal["nn", "xgb"] = Field(description="decides the worker processes (ADR 0020)")


class AnomalyEntry(ConfigModel):
    registered: str = Field(min_length=1)
    version: str = Field(min_length=1, description="a version number or 'latest'")


class GateSettings(ConfigModel):
    operating: list[MachineState] = Field(min_length=1)
    waiting: list[MachineState]
    max_gap_rows: int = Field(ge=0, description="missing readings filled before a reset")


class WorkerSettings(ConfigModel):
    start_timeout_s: float = Field(gt=0, description="bundle download + model load")
    request_timeout_s: float = Field(gt=0)
    restart_backoff_s: list[float] = Field(min_length=1)
    cache_dir: Path = Field(description="downloaded bundles (one folder per worker)")


class DriftConfig(ConfigModel):
    """Live drift detection on the operating-condition inputs (ADR 0031).

    The window, the bins and the PSI alert levels come from the monitor bundle (calibrated on
    clean validation windows); these are the live decision settings."""

    enabled: bool = Field(description="needs a monitor bundle with a drift reference")
    threshold_margin: float = Field(gt=0, description="multiplies the bundle's PSI alert level")
    clear_ratio: float = Field(gt=0, lt=1, description="clears below this x the alert level")
    sustain_s: float = Field(gt=0, description="PSI over the alert level this long raises drift")
    clear_s: float = Field(gt=0, description="PSI under the clear level this long clears it")
    adwin_delta: float = Field(gt=0, lt=1, description="ADWIN confidence (dates the change)")

    def message(self) -> dict[str, object]:
        return {
            "margin": self.threshold_margin,
            "clearRatio": self.clear_ratio,
            "sustainS": self.sustain_s,
            "clearS": self.clear_s,
            "adwinDelta": self.adwin_delta,
        }


class InferenceConfig(ConfigModel):
    enabled: bool = Field(description="off by default; `serve --inference` turns it on")
    model: str = Field(min_length=1, description="key of `models`")
    version: str = Field(min_length=1, description="a version number or 'latest'")
    anomaly: AnomalyEntry
    models: dict[str, InferenceModelEntry] = Field(min_length=1)
    tracking_dir: Path
    primary_horizon_s: int = Field(gt=0, description="alarm horizon (the evaluated +30 s)")
    batch_interval_s: float = Field(gt=0, description="readings are sent to the workers in batches")
    buffer_capacity_rows: int = Field(ge=1)
    explain_async: bool = Field(
        description="explain beside the prediction path (background thread in the worker)"
    )
    ig_steps: int = Field(ge=2, description="Integrated Gradients steps (e5.yaml explain)")
    prediction_stale_s: float = Field(gt=0, description="older predictions are not shown")
    reading_queue_size: int = Field(ge=1)
    gate: GateSettings
    drift: DriftConfig

    worker: WorkerSettings

    @model_validator(mode="after")
    def _model_known(self) -> InferenceConfig:
        if self.model not in self.models:
            raise ValueError(f"model {self.model!r} is not in models: {sorted(self.models)}")
        overlap = set(self.gate.operating) & set(self.gate.waiting)
        if overlap:
            raise ValueError(f"gate: {sorted(s.value for s in overlap)} both operating and waiting")
        return self

    def with_model(self, model: str | None, version: str | None = None) -> InferenceConfig:
        update: dict[str, object] = {}
        if model is not None:
            update["model"] = model
        if version is not None:
            update["version"] = version
        return self.model_validate(self.model_dump() | update) if update else self

    def worker_plans(self) -> list[WorkerPlan]:
        entry = self.models[self.model]
        sup = (entry.registered, self.version)
        ano = (self.anomaly.registered, self.anomaly.version)
        if entry.family == "nn":
            return [WorkerPlan("nn", ("supervised", "monitor"), sup, ano)]
        return [
            WorkerPlan("xgb", ("supervised",), sup, None),
            WorkerPlan("monitor", ("monitor",), None, ano),
        ]


@dataclass(frozen=True)
class WorkerPlan:
    name: str
    roles: tuple[str, ...]
    supervised: tuple[str, str] | None
    anomaly: tuple[str, str] | None

    def command(self, cfg: InferenceConfig, root: Path, python: str | None = None) -> list[str]:
        def abs_(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        cmd = [
            python or sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "twin_ml.serve",
            "worker",
            "--roles",
            ",".join(self.roles),
            "--tracking-dir",
            str(abs_(cfg.tracking_dir)),
            "--cache-dir",
            str(abs_(cfg.worker.cache_dir) / self.name),
            "--ig-steps",
            str(cfg.ig_steps),
        ]
        if self.supervised is not None:
            cmd += ["--supervised", self.supervised[0], "--supervised-version", self.supervised[1]]
        if self.anomaly is not None:
            cmd += ["--anomaly", self.anomaly[0], "--anomaly-version", self.anomaly[1]]
        return cmd


def load_inference_config(path: str | Path) -> InferenceConfig:
    return load_model(path, InferenceConfig)

"""Live predictions (Phase 11, ADR 0030): GET /predict, GET /predict/history.

``/predict`` describes the inference service (state, deployed bundles and their model card,
the honest notes about where the models come from) and, per machine, the prediction gate and
the newest prediction per horizon. ``/predict/history`` reads stored predictions.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from twin_backend.api.deps import Ctx, Store, Viewer, check_machines
from twin_backend.api.models import (
    BundleInfoModel,
    InferenceInfoModel,
    MachineGateModel,
    MachinePredictionModel,
    PredictionsResponse,
    PredictResponse,
)
from twin_backend.api.routes.history import From, Machines, To, _range
from twin_backend.storage.base import Stream

router = APIRouter(tags=["predictions"])


def bundle_model(info: dict[str, Any] | None) -> BundleInfoModel | None:
    if not info:
        return None
    return BundleInfoModel(
        registered=info["registered"],
        version=str(info["version"]),
        label=info["label"],
        kind=info["kind"],
        model_type=info["model"]["type"],
        seed=int(info["model"]["seed"]),
        dataset_key=info["datasetKey"],
        built_at=info["builtAt"],
        checks={k: str(v) for k, v in info["checks"].items()},
        card=info["card"],
        horizons=info.get("horizons"),
        ttf_cap_s=info.get("ttfCapS"),
        sustain_s=info.get("sustainS"),
        alpha=info.get("alpha"),
        percentile=info.get("percentile"),
    )


def inference_info(ctx: Any) -> InferenceInfoModel:
    svc = ctx.runtime.inference
    cfg = ctx.runtime.inference_cfg
    if svc is None:
        return InferenceInfoModel(
            enabled=False,
            state="off",
            model=None if cfg is None else cfg.model,
            reason="live inference is off (start the backend with --inference)",
        )
    info = svc.info
    return InferenceInfoModel(
        enabled=True,
        state=svc.status.state,
        reason=svc.status.reason,
        since=svc.status.since,
        model=svc.cfg.model,
        model_version=info.model_version or None,
        window_s=info.window or None,
        primary_horizon_s=svc.cfg.primary_horizon_s,
        prediction_stale_s=svc.cfg.prediction_stale_s,
        supervised=bundle_model(info.bundles.get("supervised")),
        anomaly=bundle_model(info.bundles.get("anomaly")),
    )


@router.get(
    "/predict",
    response_model=PredictResponse,
    summary="Live predictions: inference state, deployed models, per-machine prediction gate "
    "and newest prediction per horizon",
)
async def predict(ctx: Ctx, _: Viewer, machines: Machines = None) -> PredictResponse:
    ids = check_machines(ctx, machines) or ctx.runtime.engine.machine_ids
    svc = ctx.runtime.inference
    out: dict[str, MachinePredictionModel] = {}
    if svc is not None:
        state = ctx.runtime.engine.snapshot()
        for m in ids:
            out[m] = MachinePredictionModel(
                gate=MachineGateModel(**svc.machine_status(m)),
                predictions=dict(state.predictions.get(m, {})),
            )
    return PredictResponse(inference=inference_info(ctx), machines=out)


@router.get(
    "/predict/history",
    response_model=PredictionsResponse,
    summary="Stored predictions in a time range (every horizon and model version)",
)
async def predict_history(
    ctx: Ctx,
    store: Store,
    _: Viewer,
    t0: From,
    t1: To,
    machines: Machines = None,
    horizon_s: Annotated[float | None, Query(alias="horizonS", description="one horizon")] = None,
) -> PredictionsResponse:
    lo, hi = _range(t0, t1)
    ids = check_machines(ctx, machines)
    rows = await store.fetch(Stream.PREDICTIONS, lo, hi, machine_ids=ids)
    preds = [r.item for r in rows if horizon_s is None or r.item.horizon_s == horizon_s]
    return PredictionsResponse(predictions=preds[-ctx.runtime.api_cfg.max_history_rows :])

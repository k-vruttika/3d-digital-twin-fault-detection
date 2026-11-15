"""MLflow (local file store) helpers. mlflow is imported only inside functions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# mlflow 3.17 prints an assistant hint on import; keep CLI and test output clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# The local file store (CLAUDE.md §4) is in maintenance mode since mlflow 3.x and needs an
# explicit opt-in; it still works in the pinned version (ADR 0020).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PHASE_TAG = "E1"


def tracking_uri(tracking_dir: Path) -> str:
    """``file:`` URI of the local store (absolute, so it works from any working directory)."""
    return tracking_dir.resolve().as_uri()


def configure(tracking_dir: Path, experiment: str) -> str:
    """Point mlflow at the local store and select (or create) the experiment; returns its id."""
    import mlflow

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(tracking_dir))
    exp = mlflow.set_experiment(experiment)
    return str(exp.experiment_id)


def flatten(data: Any, prefix: str = "") -> dict[str, str]:
    """Nested config -> flat ``a.b.c`` params (values as strings)."""
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = str(data)
    return out


def finished_runs(tracking_dir: Path, experiment: str, dataset_key: str | None = None) -> list[Any]:
    """Finished E1 runs of the experiment (newest first), optionally for one dataset."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri(tracking_dir))
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    flt = f"tags.phase = '{PHASE_TAG}' and attributes.status = 'FINISHED'"
    if dataset_key:
        flt += f" and tags.dataset_key = '{dataset_key}'"
    runs: list[Any] = []
    token = None
    while True:
        page = client.search_runs(
            [exp.experiment_id],
            filter_string=flt,
            order_by=["attributes.start_time DESC"],
            max_results=500,
            page_token=token,
        )
        runs.extend(page)
        token = page.token
        if not token:
            return runs

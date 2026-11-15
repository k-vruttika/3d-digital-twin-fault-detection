"""Read simulator runs (Parquet + manifest) into per-machine series. Imports twin_sim."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from twin_common.config.line import LineConfig
from twin_common.enums import MachineKind
from twin_ml.features.inputs import ALLOWED_INPUT_COLUMNS, FloatArray, select_inputs
from twin_sim.batch import MANIFEST_NAME, file_sha256

# Columns read for labels and bookkeeping; never passed to the feature code.
LABEL_COLUMNS = ("machine_id", "t_s", "state", "fault_type")


@dataclass(frozen=True)
class MachineSeries:
    run_id: str
    machine_id: str
    kind: MachineKind
    t_s: npt.NDArray[np.float64]
    sensors: FloatArray  # (T, 5), NaN = missing; built by select_inputs only
    state: list[str]
    fault_type: list[str]


def read_manifest(source_dir: Path) -> dict[str, Any]:
    path = source_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no simulator manifest at {path}; generate the data first: uv run poe dataset"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def verify_files(source_dir: Path, manifest: dict[str, Any]) -> None:
    """Check every Parquet file against the sha256 recorded in the manifest."""
    for entry in manifest["run_entries"]:
        expected = entry.get("sha256", {}).get("parquet")
        if expected is None:
            raise ValueError(f"{entry['run_id']}: manifest has no file hash; regenerate the data")
        actual = file_sha256(source_dir / entry["files"]["parquet"])
        if actual != expected:
            raise ValueError(f"{entry['run_id']}: Parquet file does not match the manifest hash")


def iter_series(
    source_dir: Path, manifest: dict[str, Any], run_ids: Iterable[str], line: LineConfig
) -> Iterator[MachineSeries]:
    """Per-machine series of the given runs, each sorted by time, in line order."""
    wanted = set(run_ids)
    for entry in manifest["run_entries"]:
        run_id = entry["run_id"]
        if run_id not in wanted:
            continue
        table = pq.read_table(
            source_dir / entry["files"]["parquet"],
            columns=[*LABEL_COLUMNS, *ALLOWED_INPUT_COLUMNS],
        )
        cols = {
            name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names
        }
        machine = cols["machine_id"].astype(str)
        for m in line.machines:
            sel = np.flatnonzero(machine == m.id)
            if sel.size == 0:
                continue
            sel = sel[np.argsort(cols["t_s"][sel], kind="stable")]
            inputs = {c: cols[c][sel] for c in ALLOWED_INPUT_COLUMNS}
            yield MachineSeries(
                run_id=run_id,
                machine_id=m.id,
                kind=m.kind,
                t_s=cols["t_s"][sel].astype(np.float64),
                sensors=select_inputs(inputs),
                state=[str(s) for s in cols["state"][sel]],
                fault_type=[str(s) for s in cols["fault_type"][sel]],
            )

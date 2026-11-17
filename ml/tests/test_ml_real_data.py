"""Phase 10 loaders, labels, time split, build and audit on synthetic stand-in files (CI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from twin_common.enums import MachineState
from twin_ml.data.dataset import Dataset
from twin_ml.datasets import azure_pdm, metropt3
from twin_ml.datasets.build import build_real_dataset, holdout_machines, load, split_series
from twin_ml.datasets.common import (
    NO_FAULT,
    Event,
    MissingDatasetError,
    label_rows,
    require_files,
    resample_mean,
    segments,
    time_to_fault,
)
from twin_ml.datasets.config import LabelBands, load_facts, load_real_dataset_config
from twin_ml.testing_real import AZURE_FAILURES, real_tree


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Path]]:
    root = tmp_path_factory.mktemp("real")
    return root, real_tree(root)


def test_no_fault_label_matches_the_simulator() -> None:
    from twin_sim.config import NO_FAULT_LABEL

    assert NO_FAULT == NO_FAULT_LABEL


def test_missing_files_explain_what_to_download(tmp_path: Path) -> None:
    from twin_ml.testing import CONFIG_DIR

    for name, needle in (("metropt3", "UCI"), ("azure_pdm", "Kaggle")):
        facts = load_facts(CONFIG_DIR / "datasets" / f"{name}.yaml")
        with pytest.raises(MissingDatasetError) as exc:
            require_files(facts, tmp_path)
        msg = str(exc.value)
        assert needle in msg
        assert f"data/external/{name}/" in msg
        assert "never downloaded automatically" in msg


def test_resample_mean_and_segments() -> None:
    t = np.array([0, 10, 20, 70, 80, 300, 310])
    v = np.array([[1.0], [3.0], [np.nan], [5.0], [7.0], [9.0], [11.0]])
    bins, means, counts = resample_mean(t, v, 60, 0)
    assert bins.tolist() == [0, 1, 5]
    assert means[:, 0].tolist() == [2.0, 6.0, 10.0]
    assert counts.tolist() == [3, 2, 2]
    obs = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert segments(obs, 2) == [(0, 5), (8, 9)]
    assert segments(obs, 3) == [(0, 9)]


def test_label_priorities() -> None:
    t = np.arange(20, dtype=np.float64) * 60
    ev = [Event("1", "M", start_s=600.0, end_s=720.0, fault_type="leak")]
    state, ft = label_rows(t, ev, LabelBands(warning_band=3, fault_context=5, post_failure=2), 60)
    assert state[7:10] == [MachineState.WARNING.value] * 3
    assert state[10:13] == [MachineState.FAULT.value] * 3
    assert state[13:15] == [MachineState.MAINTENANCE.value] * 2
    assert state[15] == MachineState.RUNNING.value
    assert state[6] == MachineState.RUNNING.value
    assert ft[4] == NO_FAULT
    assert ft[5:13] == ["leak"] * 8
    assert ft[13] == NO_FAULT


def test_time_to_fault() -> None:
    t = np.arange(6, dtype=np.float64) * 10
    is_fault = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    ttf, no_ahead = time_to_fault(t, is_fault)
    assert ttf.tolist() == [20, 10, 0, 0, 10, 0]
    assert no_ahead.tolist() == [False, False, False, False, True, True]


def test_metropt_grid_events_and_audit(tree: tuple[Path, dict[str, Path]]) -> None:
    root, paths = tree
    cfg = load_real_dataset_config(root / paths["real_metropt3"])
    facts = load_facts(root / cfg.facts)
    raw = metropt3.read_raw(facts, root)
    a = metropt3.audit(facts, raw, step_s=60, max_gap_rows=10, lookbacks_rows=[120, 360, 1440])
    assert a["rows"] == raw.t.size
    assert a["duplicates"] == 0
    assert a["gaps"]["count"] == 2  # the 3-min and the 30-min gap
    assert a["grid"]["series"] == 2  # only the 30-min gap starts a new series
    (f,) = a["failures"]
    assert f["lookbacks"]["360"]["coverage"] == pytest.approx(1.0)
    assert 0 < f["lookbacks"]["1440"]["coverage"] <= 1.0


def test_time_split_is_ordered_and_disjoint(tree: tuple[Path, dict[str, Path]]) -> None:
    root, paths = tree
    for name in ("real_metropt3", "real_azure"):
        cfg = load_real_dataset_config(root / paths[name])
        data = load(cfg, root, sha=False)
        series = split_series(data, cfg)
        bounds = {}
        for split, ss in series.items():
            assert ss, split
            bounds[split] = (min(s.t_s[0] for s in ss), max(s.t_s[-1] for s in ss))
        assert bounds["train"][1] < bounds["val"][0] <= bounds["val"][1] < bounds["test"][0]


def test_build_metropt(tree: tuple[Path, dict[str, Path]]) -> None:
    root, paths = tree
    out = build_real_dataset(paths["real_metropt3"], root)
    ds = Dataset.open(out)
    assert ds.schema.step_s == 60
    assert len(ds.schema.sensors) == 15
    assert ds.window == 60
    assert ds.horizons == (1,)
    te = ds.split("test")
    assert te.sequence(np.arange(3)).shape == (3, 60, 31)
    assert len(ds.meta["events"]["test"]) == 1
    assert not ds.meta["events"]["train"]
    assert ds.meta["fault_types"] == ["air_leak", NO_FAULT]
    # windows never cross a series boundary
    assert np.all(te.series[te.ends] == te.series[te.ends - te.window + 1])


def test_build_azure_and_holdout(tree: tuple[Path, dict[str, Path]]) -> None:
    root, paths = tree
    ds = Dataset.open(build_real_dataset(paths["real_azure"], root))
    assert ds.schema.kinds == ("model1", "model2")
    assert ds.schema.step_s == 3600
    assert "multi" in ds.fault_types
    n_events = sum(len(v) for v in ds.meta["events"].values())
    assert n_events == len({(m, t) for m, t, _ in AZURE_FAILURES})
    tr = ds.split("train")
    assert tr.weight.max() == pytest.approx(10.0)  # far-normal subsample keep 0.1
    err = Dataset.open(build_real_dataset(paths["real_azure_err"], root))
    assert err.schema.sensors[-5:] == ("error1", "error2", "error3", "error4", "error5")
    assert err.split("test").raw[:, -5:].sum() > 0
    ho = Dataset.open(build_real_dataset(paths["real_azure_holdout"], root))
    held = ho.meta["holdout_machines"]
    assert held == holdout_machines(["M001", "M002", "M003", "M004"], 0.25, 2027)
    assert ho.meta["splits"]["test"] == held
    assert not set(held) & (set(ho.meta["splits"]["train"]) | set(ho.meta["splits"]["val"]))


def test_azure_audit(tree: tuple[Path, dict[str, Path]]) -> None:
    root, _paths = tree
    facts = load_facts(root / "config/datasets/azure_pdm.yaml")
    raw = azure_pdm.read_raw(facts, root)
    a = azure_pdm.audit(facts, raw, step_s=3600)
    assert a["telemetry"]["machines"] == 4
    assert a["telemetry"]["gaps"] == 0
    assert a["failures"]["rows"] == len(AZURE_FAILURES)
    assert a["failures"]["components_per_event"]["2"] == 1
    assert a["errors"]["events_with_error_in_prev_24h"] == pytest.approx(1.0)


def test_audit_logged_and_report(tree: tuple[Path, dict[str, Path]]) -> None:
    from twin_ml.datasets.audit import run_audit
    from twin_ml.datasets.config import load_real_config
    from twin_ml.experiments.report_real import build_report

    root, paths = tree
    for name in ("metropt3", "azure_pdm"):
        run_audit(name, paths["real"], root, smoke=True)
    real = load_real_config(root / paths["real"])
    out = build_report(real, root, experiment=real.smoke_experiment)
    md = (root / "results" / "real_data.md").read_text(encoding="utf-8")
    js = json.loads((root / "results" / "real_data.json").read_text(encoding="utf-8"))
    assert {p.name for p in out} >= {"real_data.md", "real_data.json"}
    assert "### MetroPT-3" in md
    assert "### Azure PdM" in md
    m = js["audits"]["metropt3"]["audit"]
    assert f"{m['rows']:,} rows" in md
    test = js["audits"]["azure_pdm"]["audit"]["mapping"]["splits"]["test"]
    # the failure at 2015-02-15 12:00 has its 48 h lookback partly in val: not evaluable
    assert test["evaluable_primary"] == test["events"] - 1

"""Phase 10 checks on the real files in data/external/ (local only; skipped when absent).

Nothing here asserts a number taken from outside the code: the checks compare the loaders with
each other and with the configured failure list.
"""

from __future__ import annotations

import pytest

from twin_ml.datasets import azure_pdm, metropt3
from twin_ml.datasets.common import MissingDatasetError, require_files
from twin_ml.datasets.config import load_facts, load_real_config
from twin_ml.testing import CONFIG_DIR, REPO_ROOT

REAL = load_real_config(CONFIG_DIR / "models" / "real.yaml")


def _facts(name: str):  # type: ignore[no-untyped-def]
    facts = load_facts(CONFIG_DIR / "datasets" / f"{name}.yaml")
    try:
        require_files(facts, REPO_ROOT)
    except MissingDatasetError:
        pytest.skip(f"{name} files not in data/external/ (local check only)")
    return facts


def test_metropt3_real_file_consistent() -> None:
    facts = _facts("metropt3")
    raw = metropt3.read_raw(facts, REPO_ROOT, sha=False)
    assert raw.values.shape == (raw.t.size, len(facts.sensors))
    a = metropt3.audit(facts, raw, step_s=60, max_gap_rows=10, lookbacks_rows=[360])
    assert a["failure_count"] == len(facts.failures) == 4
    for f in a["failures"]:  # every reported failure lies inside the recording
        assert a["first"] < f["start"] < a["last"]
        assert f["raw_samples_inside"] > 0


def test_azure_real_files_consistent() -> None:
    facts = _facts("azure_pdm")
    raw = azure_pdm.read_raw(facts, REPO_ROOT, sha=False)
    a = azure_pdm.audit(facts, raw, step_s=3600)
    assert a["failures"]["events"] == len(azure_pdm.events(raw, 0))
    assert set(a["failures"]["by_component"]) <= set(facts.components)
    assert set(a["errors"]["by_code"]) <= set(facts.error_codes)

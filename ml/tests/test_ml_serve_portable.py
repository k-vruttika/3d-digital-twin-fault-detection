"""Portable deploy bundles (twin_ml.serve.portable): export to one zip, import into another
MLflow store, and the imported bundles reproduce the original predictions.

Uses the quick GRU + autoencoder bundles of ``serve_setup`` (torch only; the XGBoost bundle
goes through the same code, and the worker processes keep torch and XGBoost apart).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twin_ml.data.dataset import SplitData
from twin_ml.serve.portable import (
    MANIFEST,
    ArchiveError,
    GateSpec,
    absolute_paths,
    export_bundles,
    import_bundles,
    read_archive,
)
from twin_ml.testing import run_live

GATE = GateSpec(["RUNNING", "WARNING"], ["STARVED", "BLOCKED"], 5)
NAMES = ("test-gru", "test-autoencoder")


@pytest.fixture(scope="module")
def exported(serve_setup: Any, tmp_path_factory: Any) -> Path:
    tracking, _, _, _ = serve_setup
    out: Path = tmp_path_factory.mktemp("portable") / "bundles.zip"
    res = export_bundles([(n, "latest") for n in NAMES], tracking, out, GATE)
    assert res.size_bytes == out.stat().st_size > 0
    assert [b["registered"] for b in res.bundles] == list(NAMES)
    assert all(b["reference_windows"] > 0 for b in res.bundles)
    return out


def _rewrite(src: Path, dst: Path, change: Any) -> None:
    """Copy a zip, passing every (name, bytes) through ``change`` (None drops the entry)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            got = change(info.filename, zin.read(info))
            if got is not None:
                zout.writestr(got[0], got[1])


def test_archive_holds_only_the_bundles(exported: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(exported) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read(MANIFEST))
        for n in names:
            text = zf.read(n)
            if n.endswith(".json"):
                assert absolute_paths(json.loads(text)) == [], n
    assert {n.split("/")[0] for n in names} == {MANIFEST, "bundles", "reference"}
    assert not any("mlruns" in n or "meta.yaml" in n for n in names)
    assert "bundles/test-gru/model/model.pt" in names
    assert manifest["bundles"][1]["kind"] == "anomaly"
    read_archive(exported, tmp_path / "x")  # the untouched archive verifies
    assert (tmp_path / "x" / "bundles" / "test-autoencoder" / "model" / "ae.pt").is_file()


def test_export_is_reproducible(serve_setup: Any, exported: Path, tmp_path: Path) -> None:
    tracking, _, _, _ = serve_setup
    again = export_bundles([(n, "latest") for n in NAMES], tracking, tmp_path / "b.zip", GATE)
    assert again.path.read_bytes() == exported.read_bytes()


def test_round_trip_reproduces_predictions(
    serve_setup: Any, exported: Path, tmp_path: Path
) -> None:
    from twin_ml.serve.worker import load_worker

    tracking, _, ds, _ = serve_setup
    target = tmp_path / "mlruns"
    res = import_bundles(exported, target, "P11_deploy", "P11")
    assert [(r.registered, r.version, r.reused) for r in res] == [
        ("test-gru", "1", False),
        ("test-autoencoder", "1", False),
    ]
    for r in res:  # verified twice: checksums after registering, then the live worker
        assert r.checks[0] == "checksums"
        assert r.reference_windows > 0
    again = import_bundles(exported, target, "P11_deploy", "P11", check_load=False)
    assert [(r.version, r.reused) for r in again] == [("1", True), ("1", True)]

    test: SplitData = ds.split("test")
    sid = int(test.arrays["series"][test.arrays["ends"][-1]])
    workers = [
        load_worker(
            ["supervised", "monitor"],
            ("test-gru", "latest"),
            ("test-autoencoder", "latest"),
            store,
            tmp_path / f"cache_{i}",
            8,
        )
        for i, store in enumerate((tracking, target))
    ]
    original, imported = (run_live(w, test, sid, batch=9) for w in workers)
    predicted = [k for k, e in original.items() if "out" in e]
    assert len(predicted) > 20
    for k in predicted:
        a, b = original[k]["out"], imported[k]["out"]
        for h in ds.horizons:
            np.testing.assert_array_equal(a["probs"][str(h)], b["probs"][str(h)])
        assert a["ttf"] == b["ttf"]
        assert a["set"] == b["set"]
        assert a["anomalyScore"] == b["anomalyScore"]


def test_import_rejects_a_corrupted_file(exported: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"

    def flip(name: str, data: bytes) -> tuple[str, bytes]:
        if name.endswith("model.pt"):
            data = data[:-1] + bytes([data[-1] ^ 0xFF])  # same size, different content
        return name, data

    _rewrite(exported, bad, flip)
    with pytest.raises(ArchiveError, match="SHA-256"):
        import_bundles(bad, tmp_path / "mlruns", "P11_deploy", "P11")
    assert not (tmp_path / "mlruns").exists()  # nothing registered


@pytest.mark.parametrize("member", ["../evil.json", "/abs.json", "bundles/test-gru/extra.bin"])
def test_import_rejects_unsafe_or_unlisted_entries(
    exported: Path, tmp_path: Path, member: str
) -> None:
    bad = tmp_path / "bad.zip"
    _rewrite(exported, bad, lambda n, d: (n, d))
    with zipfile.ZipFile(bad, "a") as zf:
        zf.writestr(member, b"{}")
    with pytest.raises(ArchiveError, match=r"unsafe path|does not match its manifest"):
        read_archive(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*evil*"))


def test_import_needs_every_listed_file(exported: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    _rewrite(exported, bad, lambda n, d: None if n.endswith("ae.pt") else (n, d))
    with pytest.raises(ArchiveError, match="missing"):
        read_archive(bad, tmp_path / "out")


def test_missing_archive_says_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="GitHub Releases"):
        read_archive(tmp_path / "nope.zip", tmp_path / "out")


def test_absolute_paths_are_found() -> None:
    found = absolute_paths(
        {
            "a": "C:\\Users\\x\\mlruns",
            "b": ["/home/x/data", "relative/path", "file:///tmp/x"],
            "c": {"d:/data": 1, "note": "a / b", "unit": "/"},
        }
    )
    assert sorted(found) == sorted(
        ["C:\\Users\\x\\mlruns", "/home/x/data", "file:///tmp/x", "d:/data"]
    )

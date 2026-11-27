"""Portable deploy bundles: one zip to move the registered live-inference bundles to another
machine without retraining (poe tasks ``export-bundles`` / ``import-bundles``).

**Export** resolves the registered bundle versions the live twin uses (``config/inference.yaml``:
``gru``, ``xgb`` and the autoencoder with its drift reference; ``latest`` unless given) and
writes a zip with:

* ``manifest.json``: archive format, and per bundle its registered name, kind, the version it
  had on the exporting machine, and every file with its SHA-256 and size;
* ``bundles/<registered>/bundle.json`` + ``model/...``: the bundle directory exactly as it was
  registered (model card and metadata are in ``bundle.json``). No MLflow run history, no
  absolute paths (checked: a ``bundle.json`` holding one is refused);
* ``reference/<registered>.json``: a few seeded synthetic readings and the outputs the exporting
  machine's inference worker produced for them.

Zip entries have a fixed timestamp, so the same bundles give the same zip bytes.

**Import** refuses unsafe or unlisted entries and size mismatches, verifies every checksum,
registers each bundle as a new version in the target MLflow store (a bundle already registered
there with the same content is reused, so importing twice is harmless), downloads it back
through the registry and verifies the checksums again. It then starts the real inference
worker for each bundle (the same code path as the backend) and checks it reproduces the
reference outputs. Torch and XGBoost bundles run in separate worker processes (ADR 0020).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from twin_ml.serve.bundle import BUNDLE_FILE, BUNDLE_FORMAT, register_bundle_dir, resolve

ARCHIVE_FORMAT = 1
ARCHIVE_KIND = "linetwin-bundles"
MANIFEST = "manifest.json"
BUNDLES_DIR = "bundles"
REFERENCE_DIR = "reference"
DEFAULT_ARCHIVE = Path("dist") / f"linetwin-bundles-v{ARCHIVE_FORMAT}.zip"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed entry timestamps: reproducible zip bytes
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DIGEST_TAG = "bundle_sha256"  # model version tag: content digest of an imported bundle
REFERENCE_SEED = 20261127
REFERENCE_EXTRA_ROWS = 6  # predicted windows per machine kind beyond the first full window
# Reference outputs on the importing machine vs the exporting one (another CPU / BLAS build
# may change float sums slightly; the live-vs-offline parity tests use the same tolerances).
REF_RTOL = 1e-4
REF_ATOL = 1e-4


class ArchiveError(RuntimeError):
    """The archive is malformed, unsafe, or its contents do not match the manifest."""


# ---------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_digest(files: Sequence[Mapping[str, Any]]) -> str:
    """One digest of a bundle's content: SHA-256 over its sorted (path, sha256) pairs."""
    text = "\n".join(f"{f['path']}\t{f['sha256']}" for f in sorted(files, key=lambda f: f["path"]))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_list(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "size": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def _is_absolute(s: str) -> bool:
    return (
        s.startswith(("/", "\\", "file:"))
        or PureWindowsPath(s).is_absolute()
        or bool(re.match(r"^[A-Za-z]:[\\/]", s))
    )


def absolute_paths(data: Any) -> list[str]:
    """Every string in ``data`` (nested JSON) that looks like an absolute path or file URI."""
    found: list[str] = []
    if isinstance(data, str):
        if len(data) > 1 and _is_absolute(data):
            found.append(data)
    elif isinstance(data, Mapping):
        for k, v in data.items():
            found += absolute_paths(k) + absolute_paths(v)
    elif isinstance(data, list | tuple):
        for v in data:
            found += absolute_paths(v)
    return found


def _safe_member(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or p.is_absolute()
        or PureWindowsPath(name).drive
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ArchiveError(f"unsafe path in archive: {name!r}")
    return p


def _zip_write(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data, compresslevel=9)


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------- reference


@dataclass(frozen=True)
class GateSpec:
    operating: list[str]
    waiting: list[str]
    max_gap_rows: int


def reference_inputs(meta: Mapping[str, Any], seed: int = REFERENCE_SEED) -> dict[str, Any]:
    """Seeded synthetic operating readings, one machine per machine kind of the bundle's
    normaliser: a full window plus a few rows, values around the training means with a slow
    rise of temperature and vibration (so the outputs are not all trivially NORMAL)."""
    from twin_ml.features.normalize import Normalizer

    norm = Normalizer.from_dict(meta["dataset"]["normalizer"])
    sensors = list(norm.schema.sensors)
    window = int(meta["dataset"]["window"])
    n = window + REFERENCE_EXTRA_ROWS
    rng = np.random.default_rng(seed)
    machines: dict[str, str] = {}
    rows: dict[str, list[list[Any]]] = {}
    for key in sorted(norm.speed_ref, key=str):
        kind, speed = str(key), float(norm.speed_ref[key])
        mid = f"REF-{kind}"
        machines[mid] = kind
        rise = np.linspace(0.0, 2.0, n)
        out: list[list[Any]] = []
        for i in range(n):
            vals = []
            for c, s in enumerate(sensors):
                mean, std = float(norm.mean[c]), float(norm.std[c])
                if s == "speed":
                    v = speed * (1.0 + 0.01 * rng.standard_normal())
                else:
                    v = mean + std * 0.3 * rng.standard_normal()
                    if s in ("temp", "vib"):
                        v += std * rise[i]
                vals.append(round(float(v), 6))
            out.append([i, 1_000_000.0 + i * float(norm.schema.step_s), "RUNNING", *vals])
        rows[mid] = out
    return {"seed": seed, "machines": machines, "rows": rows}


def worker_command(
    registered: str, version: str, kind: str, tracking_dir: Path, cache_dir: Path
) -> list[str]:
    """The backend's inference worker for one bundle (supervised or monitor role)."""
    role, flag = (
        ("supervised", "--supervised") if kind == "supervised" else ("monitor", "--anomaly")
    )
    return [
        sys.executable, "-m", "twin_ml.serve", "worker", "--roles", role,
        flag, registered, f"{flag}-version", version,
        "--tracking-dir", str(tracking_dir), "--cache-dir", str(cache_dir),
        "--ig-steps", "2",
    ]  # fmt: skip


def run_reference(
    cmd: Sequence[str], inputs: Mapping[str, Any], gate: GateSpec, timeout_s: float
) -> dict[str, Any]:
    """Feed the reference readings through a worker; returns its ``hello`` and the model
    outputs of every predicted window (``[[machine, seq, out], ...]``)."""
    from twin_common.jsonl_process import JsonLinesProcess

    proc = JsonLinesProcess(cmd, name="bundle-check")
    proc.start()
    try:
        hello = proc.request({"op": "hello"}, timeout_s)
        rows = inputs["rows"]
        proc.request(
            {
                "op": "configure",
                "machines": {
                    m: {"kind": k, "sensorParts": {}} for m, k in inputs["machines"].items()
                },
                "gate": {
                    "operating": gate.operating,
                    "waiting": gate.waiting,
                    "maxGapRows": gate.max_gap_rows,
                },
                "explain": False,
                "explainAsync": False,
                "capacity": max(len(r) for r in rows.values()),
            },
            timeout_s,
        )
        res = proc.request({"op": "step", "rows": rows}, timeout_s)
    finally:
        proc.close()
    outputs = [
        [mid, int(e["seq"]), e["out"]]
        for mid in sorted(res["rows"])
        for e in res["rows"][mid]
        if "out" in e
    ]
    return {"hello": hello, "outputs": outputs}


def compare_outputs(expected: Any, got: Any, where: str = "") -> list[str]:
    """Differences between two nested outputs (floats within REF_RTOL / REF_ATOL)."""
    if isinstance(expected, bool) or isinstance(got, bool):
        return [] if expected == got else [f"{where}: {got!r} != {expected!r}"]
    if isinstance(expected, int | float) and isinstance(got, int | float):
        if math.isnan(expected) and math.isnan(got):
            return []
        ok = math.isclose(float(expected), float(got), rel_tol=REF_RTOL, abs_tol=REF_ATOL)
        return [] if ok else [f"{where}: {got!r} != {expected!r}"]
    if isinstance(expected, Mapping) and isinstance(got, Mapping):
        if set(expected) != set(got):
            return [f"{where}: keys {sorted(got)} != {sorted(expected)}"]
        return [d for k in expected for d in compare_outputs(expected[k], got[k], f"{where}.{k}")]
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            return [f"{where}: length {len(got)} != {len(expected)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(expected, got, strict=True))
            for d in compare_outputs(a, b, f"{where}[{i}]")
        ]
    return [] if expected == got else [f"{where}: {got!r} != {expected!r}"]


# ---------------------------------------------------------------------------- export


@dataclass
class ExportResult:
    path: Path
    size_bytes: int
    bundles: list[dict[str, Any]]


def export_bundles(
    entries: Sequence[tuple[str, str]],
    tracking_dir: Path,
    out: Path,
    gate: GateSpec,
    timeout_s: float = 300.0,
) -> ExportResult:
    """Write the registered bundles ``[(registered, version), ...]`` to the zip ``out``."""
    bundles: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="twin_export_") as tmp_s:
        tmp = Path(tmp_s)
        staged: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
        for registered, version in entries:
            if not NAME_RE.match(registered):
                raise ValueError(f"unexpected registered model name {registered!r}")
            rb = resolve(registered, version, tracking_dir, tmp / "resolved" / registered)
            bad = absolute_paths(rb.meta)
            if bad:
                raise ArchiveError(
                    f"{registered} v{rb.version}: bundle.json has absolute paths {bad}"
                )
            files = _file_list(rb.path)
            kind = str(rb.meta["kind"])
            inputs = reference_inputs(rb.meta)
            ref = run_reference(
                worker_command(
                    registered, rb.version, kind, tracking_dir, tmp / "worker" / registered
                ),
                inputs,
                gate,
                timeout_s,
            )
            if not ref["outputs"]:
                raise ArchiveError(f"{registered}: the reference readings produced no prediction")
            entry = {
                "registered": registered,
                "name": rb.meta["name"],
                "kind": kind,
                "label": rb.label,
                "source_version": rb.version,
                "bundle_format": rb.meta["format"],
                "built_at": rb.meta["built_at"],
                "digest": bundle_digest(files),
                "files": files,
                "reference": f"{REFERENCE_DIR}/{registered}.json",
                "reference_windows": len(ref["outputs"]),
            }
            bundles.append(entry)
            staged.append((entry, rb.path, {"inputs": inputs, "outputs": ref["outputs"]}))
        manifest = {
            "kind": ARCHIVE_KIND,
            "format": ARCHIVE_FORMAT,
            "gate": {
                "operating": gate.operating,
                "waiting": gate.waiting,
                "max_gap_rows": gate.max_gap_rows,
            },
            "bundles": bundles,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        with zipfile.ZipFile(part, "w") as zf:
            _zip_write(zf, MANIFEST, _json_bytes(manifest))
            for entry, path, ref_doc in staged:
                for f in entry["files"]:
                    data = (path / f["path"]).read_bytes()
                    _zip_write(zf, f"{BUNDLES_DIR}/{entry['registered']}/{f['path']}", data)
                _zip_write(zf, entry["reference"], _json_bytes(ref_doc))
        part.replace(out)
    return ExportResult(out, out.stat().st_size, bundles)


# ---------------------------------------------------------------------------- import


@dataclass
class ImportedBundle:
    registered: str
    kind: str
    version: str
    run_id: str
    digest: str
    reused: bool
    label: str = ""
    reference_windows: int = 0
    checks: list[str] = field(default_factory=list)


def read_archive(archive: Path, dest: Path) -> dict[str, Any]:
    """Validate the archive and write its files under ``dest``; returns the manifest.

    Checks: only safe relative paths, no entry outside the manifest, every listed file present
    with the listed size and SHA-256, a supported archive and bundle format."""
    if not archive.is_file():
        raise FileNotFoundError(
            f"{archive} not found; download linetwin-bundles-v{ARCHIVE_FORMAT}.zip from the "
            "GitHub Releases page"
        )
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"{archive} is not a zip file") from exc
    with zf:
        infos = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        for name in infos:
            _safe_member(name)
        if MANIFEST not in infos:
            raise ArchiveError(f"{archive} has no {MANIFEST}")
        manifest: dict[str, Any] = json.loads(zf.read(MANIFEST).decode("utf-8"))
        if manifest.get("kind") != ARCHIVE_KIND or manifest.get("format") != ARCHIVE_FORMAT:
            raise ArchiveError(
                f"unsupported archive {manifest.get('kind')!r} format {manifest.get('format')!r} "
                f"(this code reads {ARCHIVE_KIND!r} format {ARCHIVE_FORMAT})"
            )
        expected: dict[str, tuple[int, str]] = {}
        for b in manifest["bundles"]:
            if not NAME_RE.match(str(b["registered"])):
                raise ArchiveError(f"unexpected registered model name {b['registered']!r}")
            if b.get("bundle_format") != BUNDLE_FORMAT:
                raise ArchiveError(f"{b['registered']}: bundle format {b.get('bundle_format')}")
            for f in b["files"]:
                rel = _safe_member(str(f["path"]))
                expected[f"{BUNDLES_DIR}/{b['registered']}/{rel}"] = (int(f["size"]), f["sha256"])
            ref = str(b["reference"])
            _safe_member(ref)
            expected[ref] = (infos[ref].file_size if ref in infos else -1, "")
            if bundle_digest(b["files"]) != b["digest"]:
                raise ArchiveError(f"{b['registered']}: digest does not match its file list")
        missing = sorted(set(expected) - set(infos))
        extra = sorted(set(infos) - set(expected) - {MANIFEST})
        if missing or extra:
            raise ArchiveError(
                f"archive does not match its manifest: missing {missing}, extra {extra}"
            )
        for name, (size, digest) in expected.items():
            if infos[name].file_size != size:
                raise ArchiveError(f"{name}: size {infos[name].file_size} != {size}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(infos[name]) as src, target.open("wb") as dst:
                h = hashlib.sha256()
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    h.update(chunk)
                    dst.write(chunk)
            if digest and h.hexdigest() != digest:
                raise ArchiveError(
                    f"{name}: SHA-256 {h.hexdigest()} != {digest} (corrupt download?)"
                )
    for b in manifest["bundles"]:
        meta = json.loads((dest / BUNDLES_DIR / b["registered"] / BUNDLE_FILE).read_text("utf-8"))
        if meta.get("format") != BUNDLE_FORMAT or meta.get("kind") != b["kind"]:
            raise ArchiveError(f"{b['registered']}: bundle.json does not match the manifest")
    return manifest


def _existing_version(tracking_dir: Path, registered: str, digest: str) -> tuple[str, str] | None:
    """(version, run_id) of a version of ``registered`` imported with the same content."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    from twin_ml.train import tracking

    if not tracking_dir.exists():
        return None
    uri = tracking.tracking_uri(tracking_dir)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    try:
        versions = client.search_model_versions(f"name='{registered}'")
    except MlflowException:
        return None
    same = [v for v in versions if (v.tags or {}).get(DIGEST_TAG) == digest]
    if not same:
        return None
    v = max(same, key=lambda v: int(v.version))
    return str(v.version), str(v.run_id)


def import_bundles(
    archive: Path,
    tracking_dir: Path,
    experiment: str,
    phase_tag: str,
    check_load: bool = True,
    timeout_s: float = 300.0,
) -> list[ImportedBundle]:
    """Verify ``archive`` and register its bundles in ``tracking_dir`` (see module doc)."""
    results: list[ImportedBundle] = []
    with tempfile.TemporaryDirectory(prefix="twin_import_") as tmp_s:
        tmp = Path(tmp_s)
        manifest = read_archive(archive, tmp / "archive")
        g = manifest["gate"]
        gate = GateSpec(list(g["operating"]), list(g["waiting"]), int(g["max_gap_rows"]))
        for b in manifest["bundles"]:
            registered, digest = str(b["registered"]), str(b["digest"])
            src = tmp / "archive" / BUNDLES_DIR / registered
            found = _existing_version(tracking_dir, registered, digest)
            if found is not None:
                version, run_id = found
                reused = True
            else:
                run_id, version = register_bundle_dir(
                    src,
                    registered,
                    tracking_dir,
                    experiment,
                    {"phase": phase_tag, "imported": "true", "source_version": b["source_version"]},
                    {DIGEST_TAG: digest, "imported": "true", "source_version": b["source_version"]},
                )
                reused = False
            res = ImportedBundle(registered, str(b["kind"]), version, run_id, digest, reused)
            rb = resolve(registered, version, tracking_dir, tmp / "check" / registered)
            if bundle_digest(_file_list(rb.path)) != digest:
                raise ArchiveError(
                    f"{registered} v{version}: registered files differ from the archive"
                )
            res.label = rb.label
            res.checks.append("checksums")
            if check_load:
                ref = json.loads((tmp / "archive" / b["reference"]).read_text("utf-8"))
                cmd = worker_command(
                    registered, version, res.kind, tracking_dir, tmp / "worker" / registered
                )
                got = run_reference(cmd, ref["inputs"], gate, timeout_s)
                diffs = compare_outputs(ref["outputs"], got["outputs"], registered)
                if diffs:
                    raise ArchiveError(
                        f"{registered} v{version} loads but does not reproduce the exporting "
                        f"machine's reference outputs: {diffs[:5]}"
                    )
                res.reference_windows = len(got["outputs"])
                res.checks.append("loads in the inference worker")
                res.checks.append(f"reproduces {res.reference_windows} reference predictions")
            results.append(res)
    return results

"""The inference worker process (ADR 0030): JSON lines on stdin, JSON lines on stdout.

Started by the backend (``python -m twin_ml.serve worker ...``) with one or both roles:

* ``supervised``: a supervised bundle (XGBoost, or a sequence model);
* ``monitor``: the autoencoder bundle (torch) — later also drift detection.

The backend runs ``gru`` as one process with both roles; with ``xgb`` it runs a supervised
XGBoost worker and a separate torch monitor worker (ADR 0020). Every worker keeps its own
buffers of the same readings, so both compute the same gate.

Requests (one JSON object per line, answered in order, each answer echoes ``id``):

* ``{"op": "hello"}`` -> bundle info, loaded libraries;
* ``{"op": "configure", "machines": {id: {"kind", "sensorParts"}}, "gate": {...},
  "explainAsync": bool, "explain": bool (default true; false = no attributions, used by
  what-if), "drift": {...settings} or null, "capacity": rows}``;
* ``{"op": "step", "rows": {machine: [[seq, tsEpochS, state, speed, temp, vib, load,
  current], ...]}}`` -> for every new reading its gate status, and the model outputs of every
  predicted window, plus finished explanations and per-stage timings;
* ``{"op": "shutdown"}``.

**Explanations beside the prediction path.** The newest window of a machine whose
uncalibrated +30 s argmax is WARNING or FAULT is explained (``SupervisedLive.explain``, the
evaluated ``explain()``). Integrated Gradients on a sequence model takes far longer than a
prediction on one CPU thread (measured in ``live_inference.md``), so by default a background
thread explains while predictions continue: it keeps only the newest request per machine and
the finished attributions come back with a later ``step`` reply, tagged with the window they
explain (``seq`` / ``ts``). ``explainAsync: false`` explains synchronously (tests).

**Drift** (monitor role, ADR 0031): with a ``drift`` setting and a bundle that carries a training
reference, every operating row of a machine updates its PSI / ADWIN monitor and the entry of
that row gets a ``drift`` report (``drifted``, per-channel ``psi``, ``adwinTs``, ``since``).

stdout carries only protocol lines; logging goes to stderr.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np

from twin_common.enums import MachineKind
from twin_ml.serve.buffers import PREDICT, GateConfig, MachineBuffer, NewRow
from twin_ml.serve.drift import DriftSettings, MachineDrift, drift_public
from twin_ml.serve.live import AnomalyLive, SupervisedLive, Timer

ROLES = ("supervised", "monitor")
ML_LIBRARIES = ("torch", "xgboost", "sklearn")


def process_info() -> dict[str, Any]:
    """ML libraries and OpenMP runtimes loaded here; torch + XGBoost never (ADR 0020)."""
    from twin_ml.train.isolated import openmp_runtimes

    loaded = [m for m in ML_LIBRARIES if m in sys.modules]
    if "torch" in loaded and "xgboost" in loaded:
        raise RuntimeError("torch and xgboost are both loaded in this inference worker")
    return {"libraries": loaded, "openmp_runtimes": openmp_runtimes()}


class Explainer:
    """Background explanations: newest request per machine, batched, results collected."""

    def __init__(self, model: SupervisedLive) -> None:
        self.model = model
        self._cond = threading.Condition()
        self._pending: dict[str, dict[str, Any]] = {}
        self._done: list[dict[str, Any]] = []
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="explainer", daemon=True)
        self._thread.start()

    def submit(self, machine: str, request: dict[str, Any]) -> None:
        with self._cond:
            request["requested"] = time.perf_counter()
            self._pending[machine] = request  # an older pending request is dropped
            self._cond.notify_all()

    def collect(self) -> list[dict[str, Any]]:
        with self._cond:
            done, self._done = self._done, []
        return done

    def wait_idle(self, timeout_s: float) -> None:
        """Tests and measurements: wait until nothing is pending or being explained."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._pending or self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("explanations still pending")
                self._cond.wait(left)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                batch, self._pending = self._pending, {}
                self._busy = True
            reqs = list(batch.items())
            t0 = time.perf_counter()
            error: str | None = None
            try:
                att = self.model.explain(
                    np.stack([r["values"] for _, r in reqs]),
                    np.stack([r["mask"] for _, r in reqs]),
                    [r["kind"] for _, r in reqs],
                    [r["parts"] for _, r in reqs],
                )
            except Exception as exc:  # reported with the results, never kills the worker
                att = [[] for _ in reqs]
                error = f"{type(exc).__name__}: {exc}"
            now = time.perf_counter()
            with self._cond:
                for (machine, r), a in zip(reqs, att, strict=True):
                    self._done.append(
                        {
                            "machine": machine,
                            "seq": r["seq"],
                            "ts": r["ts"],
                            "attributions": a,
                            "computeMs": (now - t0) * 1000.0 / len(reqs),
                            "ageMs": (now - r["requested"]) * 1000.0,
                            "batch": len(reqs),
                            "error": error,
                        }
                    )
                self._busy = False
                self._cond.notify_all()


class Worker:
    def __init__(
        self,
        supervised: SupervisedLive | None,
        anomaly: AnomalyLive | None,
        labels: dict[str, str],
        bundles: dict[str, dict[str, Any]],
    ) -> None:
        if supervised is None and anomaly is None:
            raise ValueError("a worker needs at least one role")
        windows = {b.window for b in (supervised, anomaly) if b is not None}
        if len(windows) != 1:
            raise ValueError(f"bundles disagree on the window length: {sorted(windows)}")
        self.window = windows.pop()
        self.supervised = supervised
        self.anomaly = anomaly
        self.labels = labels
        self.bundles = bundles
        self.buffers: dict[str, MachineBuffer] = {}
        self.parts: dict[str, Mapping[str, str]] = {}
        self.explain = True
        self.explain_async = True
        self.explainer: Explainer | None = None
        self.drift: dict[str, MachineDrift] = {}
        self.drift_states: frozenset[str] = frozenset()
        self.drift_channels: dict[str, int] = {}  # channel -> column in a raw row

    # ------------------------------------------------------------------ ops
    def hello(self) -> dict[str, Any]:
        return {
            "roles": [
                r
                for r, b in (("supervised", self.supervised), ("monitor", self.anomaly))
                if b is not None
            ],
            "window": self.window,
            "labels": self.labels,
            "bundles": self.bundles,
            "process": process_info(),
        }

    def configure(self, msg: dict[str, Any]) -> dict[str, Any]:
        g = msg["gate"]
        cfg = GateConfig.of(self.window, g["operating"], g["waiting"], int(g["maxGapRows"]))
        capacity = int(msg["capacity"])
        self.explain = bool(msg.get("explain", True))
        self.explain_async = bool(msg.get("explainAsync", True)) and self.explain
        if self.explainer is not None:
            self.explainer.stop()
            self.explainer = None
        if self.explain_async and self.supervised is not None:
            self.explainer = Explainer(self.supervised)
        self.buffers = {}
        self.parts = {}
        self.drift = {}
        ref = self.anomaly.drift_reference if self.anomaly is not None else None
        settings = msg.get("drift")
        if ref is not None and settings is not None:
            ds = DriftSettings.of(settings, ref["window_s"])
            self.drift_states = frozenset(str(s) for s in ref["states"])
            sensors = list(self.anomaly.normalizer.schema.sensors) if self.anomaly else []
            self.drift_channels = {str(c): sensors.index(str(c)) for c in ref["channels"]}
        for mid, m in msg["machines"].items():
            self.buffers[mid] = MachineBuffer(mid, MachineKind(m["kind"]), cfg, capacity)
            self.parts[mid] = dict(m.get("sensorParts", {}))
            if ref is not None and settings is not None:
                md = MachineDrift(str(m["kind"]), ref["kinds"], ds)
                if md.available:
                    self.drift[mid] = md
        return {"machines": sorted(self.buffers), "drift": sorted(self.drift)}

    def step(self, rows: dict[str, list[list[Any]]]) -> dict[str, Any]:
        started = time.perf_counter()
        timer = Timer()
        t = time.perf_counter()
        report: dict[str, list[dict[str, Any]]] = {}
        todo: list[tuple[str, NewRow, dict[str, Any]]] = []
        for mid, batch in rows.items():
            buf = self.buffers.get(mid)
            if buf is None:
                continue
            entries = []
            for n in buf.extend(batch):
                entry: dict[str, Any] = {
                    "seq": n.seq,
                    "ts": n.ts,
                    "status": n.status,
                    "runLen": n.run_len,
                }
                entries.append(entry)
                self._drift(mid, buf, n, entry)
                if n.status == PREDICT and buf.has_window(n.row):
                    todo.append((mid, n, entry))
            report[mid] = entries
        explanations: list[dict[str, Any]] = []
        if todo:
            kinds = [self.buffers[mid].kind for mid, _, _ in todo]
            pairs = [(mid, n) for mid, n, _ in todo]
            outs: list[dict[str, Any]] = [{} for _ in todo]
            sup_windows: tuple[Any, Any] | None = None
            for bundle in (self.supervised, self.anomaly):
                if bundle is None:
                    continue
                values, mask = self._windows(bundle.normalizer, pairs)
                timer.add("features", t)
                res = bundle.run(values, mask, kinds, timer)
                if isinstance(bundle, SupervisedLive):
                    sup_windows = (values, mask)
                t = time.perf_counter()
                for i, r in enumerate(res):
                    outs[i] |= r
            for (_, _, entry), out in zip(todo, outs, strict=True):
                entry["out"] = out
            if sup_windows is not None and self.explain:
                explanations += self._explain(todo, outs, sup_windows, timer)
        if self.explainer is not None:
            explanations += self.explainer.collect()
        total = time.perf_counter() - started
        return {
            "rows": report,
            "windows": len(todo),
            "explanations": explanations,
            "timings_ms": {k: v * 1000.0 for k, v in timer.seconds.items()}
            | {"total": total * 1000.0},
        }

    def _drift(self, mid: str, buf: MachineBuffer, n: NewRow, entry: dict[str, Any]) -> None:
        """Operating-condition drift: only operating rows of machines with a reference count."""
        md = self.drift.get(mid)
        if md is None or n.state not in self.drift_states:
            return
        i = buf.index(n.row)
        if i is None:
            return
        raw = buf.raw[i]
        values = {c: float(raw[j]) for c, j in self.drift_channels.items()}
        entry["drift"] = md.update(values, n.ts)

    def _explain(
        self,
        todo: list[tuple[str, NewRow, dict[str, Any]]],
        outs: list[dict[str, Any]],
        windows: tuple[Any, Any],
        timer: Timer,
    ) -> list[dict[str, Any]]:
        """Request (async) or compute (sync) the explanation of the newest alarmed window of
        every machine in this batch; returns the synchronous results."""
        assert self.supervised is not None
        newest: dict[str, int] = {}
        for i, ((mid, _, _), out) in enumerate(zip(todo, outs, strict=True)):
            if out.get("alarmClass"):
                newest[mid] = i
        if not newest:
            return []
        values, mask = windows
        if self.explainer is not None:
            for mid, i in newest.items():
                n = todo[i][1]
                self.explainer.submit(
                    mid,
                    {
                        "seq": n.seq,
                        "ts": n.ts,
                        "values": values[i],
                        "mask": mask[i],
                        "kind": self.buffers[mid].kind,
                        "parts": self.parts[mid],
                    },
                )
            return []
        t = time.perf_counter()
        idx = list(newest.values())
        att = self.supervised.explain(
            values[idx],
            mask[idx],
            [self.buffers[todo[i][0]].kind for i in idx],
            [self.parts[todo[i][0]] for i in idx],
        )
        ms = (time.perf_counter() - t) * 1000.0
        timer.add("explain", t)
        return [
            {
                "machine": todo[i][0],
                "seq": todo[i][1].seq,
                "ts": todo[i][1].ts,
                "attributions": a,
                "computeMs": ms / len(idx),
                "ageMs": ms,
                "batch": len(idx),
                "error": None,
            }
            for i, a in zip(idx, att, strict=True)
        ]

    def _windows(self, norm: Any, todo: list[tuple[str, NewRow]]) -> tuple[Any, Any]:
        by_machine: dict[str, list[int]] = {}
        for mid, n in todo:
            by_machine.setdefault(mid, []).append(n.row)
        parts_v: dict[str, Any] = {}
        parts_m: dict[str, Any] = {}
        for mid, rws in by_machine.items():
            parts_v[mid], parts_m[mid] = self.buffers[mid].windows(norm, rws)
        cursor = dict.fromkeys(by_machine, 0)
        vs, ms = [], []
        for mid, _ in todo:
            k = cursor[mid]
            vs.append(parts_v[mid][k])
            ms.append(parts_m[mid][k])
            cursor[mid] = k + 1
        return np.stack(vs), np.stack(ms)


# ---------------------------------------------------------------------------- process


def load_worker(
    roles: list[str],
    supervised: tuple[str, str] | None,
    anomaly: tuple[str, str] | None,
    tracking_dir: Path,
    cache_dir: Path,
    ig_steps: int,
) -> Worker:
    from twin_ml.serve.bundle import resolve

    sup = ano = None
    labels: dict[str, str] = {}
    bundles: dict[str, dict[str, Any]] = {}
    if "supervised" in roles:
        if supervised is None:
            raise ValueError("the supervised role needs --supervised")
        rb = resolve(supervised[0], supervised[1], tracking_dir, cache_dir / "supervised")
        sup = SupervisedLive(rb.meta, rb.path, ig_steps)
        labels["supervised"] = rb.label
        bundles["supervised"] = _public(rb)
    if "monitor" in roles:
        if anomaly is None:
            raise ValueError("the monitor role needs --anomaly")
        rb = resolve(anomaly[0], anomaly[1], tracking_dir, cache_dir / "anomaly")
        ano = AnomalyLive(rb.meta, rb.path)
        labels["anomaly"] = rb.label
        bundles["anomaly"] = _public(rb)
    return Worker(sup, ano, labels, bundles)


def _public(rb: Any) -> dict[str, Any]:
    """What the backend and the UI may show about a bundle."""
    m = rb.meta
    out: dict[str, Any] = {
        "registered": rb.name,
        "version": rb.version,
        "runId": rb.run_id,
        "label": rb.label,
        "kind": m["kind"],
        "model": {k: m["model"][k] for k in ("type", "family", "seed")}
        | {"sourceRunId": m["model"].get("source_run_id")},
        "datasetKey": m["dataset"]["key"],
        "window": m["dataset"]["window"],
        "checks": {
            k: v.get("status", "ok") if isinstance(v, dict) else v for k, v in m["checks"].items()
        },
        "builtAt": m["built_at"],
        "card": m["card"],
    }
    if m["kind"] == "supervised":
        d = m["dataset"]
        out |= {
            "horizons": d["horizons"],
            "ttfCapS": d["ttf_cap_s"],
            "sustainS": m["sustain_s"],
            "operatingSubset": m["operating_subset"],
            "alpha": m["calibration"]["alpha"],
            "seedSelection": m["seed_selection"],
        }
    else:
        a = m["anomaly"]
        out |= {
            "percentile": a["primary_percentile"],
            "thresholds": a["thresholds"],
            "drift": drift_public(m.get("drift")),
        }
    return out


def serve(worker_factory: Any, stdin: IO[str], stdout: IO[str]) -> int:
    """The request loop. The worker is built lazily so that load errors are reported."""
    worker: Worker | None = None
    error: str | None = None
    try:
        worker = worker_factory()
    except Exception:  # reported to the backend on every request
        error = traceback.format_exc(limit=8)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        msg: dict[str, Any] = {}
        try:
            msg = json.loads(line)
            op = msg.get("op")
            if op == "shutdown":
                _send(stdout, {"id": msg.get("id"), "ok": True})
                return 0
            if worker is None:
                raise RuntimeError(f"the worker failed to load its bundles:\n{error}")
            if op == "hello":
                result = worker.hello()
            elif op == "configure":
                result = worker.configure(msg)
            elif op == "step":
                result = worker.step(msg["rows"])
            else:
                raise ValueError(f"unknown op {op!r}")
            _send(stdout, {"id": msg.get("id"), "ok": True, "result": result})
        except Exception as exc:
            _send(
                stdout,
                {
                    "id": msg.get("id"),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=8),
                },
            )
    return 0


def _send(stdout: IO[str], obj: dict[str, Any]) -> None:
    stdout.write(json.dumps(obj, default=_json_default) + "\n")
    stdout.flush()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")

"""Run one (model, seed) training in its own Python process (ADR 0020).

PyTorch and XGBoost each bring an OpenMP runtime. On macOS both are LLVM ``libomp.dylib``:
torch ships its own copy, XGBoost loads Homebrew's. Two copies of that runtime in one process
can crash it (segmentation fault). Running every (model, seed) in a fresh worker process means
a tree model never shares a process with torch and a sequence model never shares one with
XGBoost, on every OS. ``KMP_DUPLICATE_LIB_OK`` is deliberately not used: it only silences
Intel's duplicate-runtime check and leaves two runtimes fighting over the same threads.

Parent: :func:`run_isolated` writes a JSON request, starts
``python -X faulthandler -m twin_ml.train.isolated <request> <result>`` and reads back the
:class:`~twin_ml.train.trainer.RunResult`. The worker calls
:func:`~twin_ml.train.trainer.run_one` unchanged, so metrics, MLflow runs and tags are
identical to an in-process run. The worker inherits stdout and stderr, so its logs and, on a
crash, faulthandler's traceback of the crashing frame appear in the parent's output.

This module imports no heavy library; the worker imports torch, scikit-learn or XGBoost only
through the predictor of the model it trains.
"""

from __future__ import annotations

import faulthandler
import json
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from twin_ml.config import EvalConfig, ModelConfig
from twin_ml.train.trainer import RunResult

WORKER_MODULE = "twin_ml.train.isolated"


class TrainingProcessError(RuntimeError):
    """The worker process of one (model, seed) exited without a result."""

    def __init__(self, model: str, seed: int, returncode: int) -> None:
        self.model = model
        self.seed = seed
        self.returncode = returncode
        super().__init__(
            f"training {model} seed {seed} failed in its worker process "
            f"({describe_exit(returncode)}); its output, including faulthandler's traceback "
            "on a crash, is printed above"
        )


def describe_exit(returncode: int) -> str:
    """``exit code 1`` or, for a POSIX process killed by a signal, ``signal SIGSEGV``."""
    if returncode < 0:
        try:
            return f"signal {signal.Signals(-returncode).name}"
        except ValueError:
            return f"signal {-returncode}"
    return f"exit code {returncode}"


def build_request(
    model: str,
    cfg: ModelConfig,
    seed: int,
    dataset_dir: Path,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
) -> dict[str, Any]:
    """JSON-serialisable arguments of :func:`~twin_ml.train.trainer.run_one`.

    Relative paths (``root``, the tracking dir, a threshold baseline's ``sim_config``) stay
    relative: the worker runs in the parent's working directory.
    """
    return {
        "model": model,
        "config": cfg.model_dump(mode="json"),
        "seed": seed,
        "dataset_dir": str(dataset_dir.resolve()),
        "eval_config": eval_cfg.model_dump(mode="json"),
        "root": str(root),
        "tracking_dir": None if tracking_dir is None else str(tracking_dir),
    }


def run_isolated(
    model: str,
    cfg: ModelConfig,
    seed: int,
    dataset_dir: Path,
    eval_cfg: EvalConfig,
    root: Path = Path(),
    tracking_dir: Path | None = None,
) -> RunResult:
    """:func:`~twin_ml.train.trainer.run_one` in a fresh worker process."""
    request = build_request(model, cfg, seed, dataset_dir, eval_cfg, root, tracking_dir)
    with tempfile.TemporaryDirectory(prefix="twin_train_") as tmp:
        req_path, res_path = Path(tmp) / "request.json", Path(tmp) / "result.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        cmd = [sys.executable, "-X", "faulthandler", "-m", WORKER_MODULE, str(req_path)]
        proc = subprocess.run([*cmd, str(res_path)], check=False)
        if proc.returncode != 0 or not res_path.is_file():
            raise TrainingProcessError(model, seed, proc.returncode)
        data = json.loads(res_path.read_text(encoding="utf-8"))
    return RunResult(**data)


def openmp_runtimes() -> list[str]:
    """Paths of the OpenMP runtimes loaded in this process (empty without threadpoolctl)."""
    try:
        from threadpoolctl import threadpool_info  # scikit-learn dependency
    except ImportError:  # pragma: no cover
        return []
    return sorted(
        {str(i["filepath"]) for i in threadpool_info() if i.get("internal_api") == "openmp"}
    )


def main(argv: list[str] | None = None) -> int:
    """Worker: ``python -m twin_ml.train.isolated <request.json> <result.json>``."""
    faulthandler.enable(all_threads=True)  # also on without -X faulthandler
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(f"usage: python -m {WORKER_MODULE} <request.json> <result.json>", file=sys.stderr)
        return 2
    from twin_common.logging import configure_logging, get_logger
    from twin_ml.config import parse_model_config
    from twin_ml.data.dataset import Dataset
    from twin_ml.train.trainer import run_one

    configure_logging()
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    tracking_dir = request["tracking_dir"]
    res = run_one(
        request["model"],
        parse_model_config(request["config"]),
        int(request["seed"]),
        Dataset.open(Path(request["dataset_dir"])),
        EvalConfig.model_validate(request["eval_config"]),
        Path(request["root"]),
        None if tracking_dir is None else Path(tracking_dir),
    )
    # Which OpenMP runtime the model loaded; at most one per worker (ADR 0020).
    get_logger(__name__).info(
        "train.openmp", model=res.model, seed=res.seed, runtimes=openmp_runtimes()
    )
    Path(args[1]).write_text(json.dumps(asdict(res)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

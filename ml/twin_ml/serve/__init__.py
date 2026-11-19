"""Live inference for the twin (Phase 11, ADR 0029 / 0030).

* :mod:`twin_ml.serve.bundle`: build a deploy bundle (model + calibration + conformal +
  autoencoder) from the evaluated MLflow runs, register it, and load it back;
* :mod:`twin_ml.serve.buffers`: per-machine 1 Hz buffers and the prediction gate;
* :mod:`twin_ml.serve.live`: one bundle's predictions for a batch of windows;
* :mod:`twin_ml.serve.worker`: the inference worker process (JSON lines on stdin / stdout),
  started by the backend. One process never loads both torch and XGBoost (ADR 0020).

Importing this package loads no ML library; ``python -m twin_ml.serve`` is the CLI.
"""

"""Online inference, windows, drift, model registry (Phase 11, ADR 0030).

``service.InferenceService`` (event bus -> worker processes -> Prediction events and alert
conditions), ``workers.WorkerPool`` (the ``twin_ml.serve`` child processes) and
``alarms`` (the evaluated alarm rule). Importing this package loads no ML library.
"""

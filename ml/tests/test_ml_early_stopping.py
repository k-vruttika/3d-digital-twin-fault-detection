"""Early-stopping rule: EMA smoothing, min_epochs, patience, max cap, best checkpoint."""

from __future__ import annotations

import pytest

from twin_ml.config import EarlyStopping
from twin_ml.train.early_stopping import EarlyStopper


def cfg(**kw: float | str | None) -> EarlyStopping:
    base: dict[str, float | str | None] = {
        "min_epochs": 1,
        "max_epochs": 50,
        "patience": 2,
        "ema_alpha": 1.0,
        "min_delta": 0.0,
    }
    base.update(kw)
    return EarlyStopping.model_validate(base)


def run(stopper: EarlyStopper, scores: list[float]) -> int:
    for s in scores:
        _, stop = stopper.update(s)
        if stop:
            break
    return stopper.epoch


def test_without_smoothing_patience_counts_epochs_without_improvement() -> None:
    st = EarlyStopper(cfg(patience=2))
    assert run(st, [0.5, 0.6, 0.55, 0.58, 0.9]) == 4
    assert st.stop_reason == "patience"
    assert (st.best_epoch, st.best_score) == (2, 0.6)


def test_ema_ignores_a_single_noisy_dip() -> None:
    noisy = [0.60, 0.65, 0.55, 0.68, 0.70, 0.62, 0.72, 0.73]
    raw = EarlyStopper(cfg(patience=1))
    smooth = EarlyStopper(cfg(patience=2, ema_alpha=0.5))
    assert run(raw, noisy) == 3  # the dip at epoch 3 stops raw patience-1 at once
    assert run(smooth, noisy) == len(noisy)  # the smoothed signal keeps improving
    assert smooth.emas[1] == pytest.approx(0.625)
    assert smooth.best_epoch == 8  # checkpoint by raw score


def test_min_epochs_and_max_epochs() -> None:
    st = EarlyStopper(cfg(min_epochs=5, patience=1))
    assert run(st, [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) == 5
    assert st.best_epoch == 1
    capped = EarlyStopper(cfg(max_epochs=3))
    assert run(capped, [0.1, 0.2, 0.3, 0.4]) == 3
    assert capped.stop_reason == "max_epochs"


def test_min_delta_requires_a_real_improvement() -> None:
    st = EarlyStopper(cfg(patience=2, min_delta=0.05))
    assert run(st, [0.50, 0.52, 0.53, 0.54]) == 3
    with pytest.raises(ValueError, match="min_epochs"):
        cfg(min_epochs=10, max_epochs=5)


def test_resume_skips_only_runs_with_the_same_config_hash() -> None:
    from twin_ml.experiments.e1 import pending

    done = {("lstm", 0, "new"), ("lstm", 1, "old"), ("threshold", 0, "t")}
    todo = pending(
        ["threshold", "lstm", "gru"], [0, 1, 2], {"threshold": "t", "lstm": "new", "gru": "g"}, done
    )
    assert todo == [("lstm", 1), ("lstm", 2), ("gru", 0), ("gru", 1), ("gru", 2)]


def test_fixed_budget_never_stops_early() -> None:
    st = EarlyStopper(cfg(patience=None, max_epochs=8))
    assert run(st, [0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]) == 8
    assert st.stop_reason == "max_epochs"


def test_checkpoint_by_ema_prefers_smoothed_peak() -> None:
    scores = [0.5, 0.6, 0.85, 0.55, 0.7, 0.72]  # one lucky spike at epoch 3
    raw = EarlyStopper(cfg(patience=None, max_epochs=6))
    ema = EarlyStopper(cfg(patience=None, max_epochs=6, ema_alpha=0.4, checkpoint_by="ema"))
    run(raw, scores)
    run(ema, scores)
    assert raw.best_epoch == 3
    assert ema.best_epoch == 6
    assert ema.best_score == pytest.approx(0.72)  # raw score at the selected epoch

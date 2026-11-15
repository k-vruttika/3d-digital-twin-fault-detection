"""Early-stopping rule for the sequence models (pure Python, unit tested).

The stopping signal is an exponential moving average (EMA) of the validation score, so a
single noisy epoch neither stops training nor resets the patience. The checkpoint to keep is
chosen separately: by the best raw score or by the best EMA (``checkpoint_by``). With
``patience`` null the rule never stops early (fixed training budget of ``max_epochs``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twin_ml.config import EarlyStopping


@dataclass
class EarlyStopper:
    cfg: EarlyStopping
    epoch: int = 0
    ema: float | None = None
    best_score: float = float("-inf")
    best_epoch: int = 0
    best_selected: float = float("-inf")
    best_ema: float = float("-inf")
    since_ema: int = 0
    stop_reason: str = "max_epochs"
    emas: list[float] = field(default_factory=list)

    def update(self, score: float) -> tuple[bool, bool]:
        """Record one epoch's validation score.

        Returns (is_best, stop): ``is_best`` when this epoch beats every earlier one on the
        checkpoint signal (keep this checkpoint; ``best_score`` is its raw score), ``stop`` when
        training should end after this epoch.
        """
        self.epoch += 1
        a = self.cfg.ema_alpha
        self.ema = score if self.ema is None else a * score + (1 - a) * self.ema
        self.emas.append(self.ema)
        selected = self.ema if self.cfg.checkpoint_by == "ema" else score
        is_best = selected > self.best_selected
        if is_best:
            self.best_selected, self.best_epoch, self.best_score = selected, self.epoch, score
        if self.ema > self.best_ema + self.cfg.min_delta:
            self.best_ema, self.since_ema = self.ema, 0
        else:
            self.since_ema += 1
        stop = False
        if self.epoch >= self.cfg.max_epochs:
            stop, self.stop_reason = True, "max_epochs"
        elif (
            self.cfg.patience is not None
            and self.epoch >= self.cfg.min_epochs
            and self.since_ema >= self.cfg.patience
        ):
            stop, self.stop_reason = True, "patience"
        return is_best, stop

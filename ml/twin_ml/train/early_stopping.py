"""Early-stopping rule for the sequence models (pure Python, unit tested).

The stopping signal is an exponential moving average (EMA) of the validation score, so a
single noisy epoch neither stops training nor resets the patience. The checkpoint to keep is
chosen separately, by the best *raw* score.
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
    best_ema: float = float("-inf")
    since_ema: int = 0
    stop_reason: str = "max_epochs"
    emas: list[float] = field(default_factory=list)

    def update(self, score: float) -> tuple[bool, bool]:
        """Record one epoch's validation score.

        Returns (is_best, stop): ``is_best`` when the raw score beats every earlier epoch (keep
        this checkpoint), ``stop`` when training should end after this epoch.
        """
        self.epoch += 1
        a = self.cfg.ema_alpha
        self.ema = score if self.ema is None else a * score + (1 - a) * self.ema
        self.emas.append(self.ema)
        is_best = score > self.best_score
        if is_best:
            self.best_score, self.best_epoch = score, self.epoch
        if self.ema > self.best_ema + self.cfg.min_delta:
            self.best_ema, self.since_ema = self.ema, 0
        else:
            self.since_ema += 1
        stop = False
        if self.epoch >= self.cfg.max_epochs:
            stop, self.stop_reason = True, "max_epochs"
        elif self.epoch >= self.cfg.min_epochs and self.since_ema >= self.cfg.patience:
            stop, self.stop_reason = True, "patience"
        return is_best, stop

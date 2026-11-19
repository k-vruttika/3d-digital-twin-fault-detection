"""Random Forest and XGBoost on window summary features (lazy heavy imports).

Per model: one classifier per horizon, a time-to-fault regressor trained on rows with a known
target only (censored rows excluded, ADR 0019), and a fault-type classifier. Every head uses
sample weight = subsampling weight x class weight. XGBoost early-stops each head on its
validation loss; the Random Forest has no early stopping.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from twin_ml.config import RfConfig, XgbConfig
from twin_ml.data.dataset import SplitData
from twin_ml.data.labels import HEALTH
from twin_ml.evaluate.evaluator import Predictions
from twin_ml.features.summary import summary_features
from twin_ml.models.base import class_weights, effective_weights


class _Classifier:
    """Wraps a fitted classifier trained on the classes present; expands to all K columns."""

    def __init__(self, model: Any, present: npt.NDArray[np.int64], k: int) -> None:
        self.model, self.present, self.k = model, present, k

    def proba(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        out = np.zeros((x.shape[0], self.k), dtype=np.float32)
        if self.model is None:  # a single class in training: constant prediction
            out[:, self.present[0]] = 1.0
        else:
            out[:, self.present] = self.model.predict_proba(x)
        return out


class TreeModel:
    def __init__(self, cfg: RfConfig | XgbConfig, n_fault_types: int, ttf_cap_s: float) -> None:
        self.cfg = cfg
        self.name: str = cfg.type
        self.n_fault_types = n_fault_types
        self.ttf_cap_s = ttf_cap_s
        self.heads: dict[str, _Classifier] = {}
        self.regressor: Any = None
        self.horizons: tuple[int, ...] = ()

    # ------------------------------------------------------------------ factories
    def _classifier(self, seed: int) -> Any:
        if isinstance(self.cfg, RfConfig):
            from sklearn.ensemble import RandomForestClassifier

            p = self.cfg.classifier
            return RandomForestClassifier(
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                min_samples_leaf=p.min_samples_leaf,
                max_features=p.max_features,
                n_jobs=self.cfg.n_jobs,
                random_state=seed,
            )
        from xgboost import XGBClassifier

        q = self.cfg.classifier
        return XGBClassifier(
            n_estimators=q.n_estimators,
            max_depth=q.max_depth,
            learning_rate=q.learning_rate,
            subsample=q.subsample,
            colsample_bytree=q.colsample_bytree,
            min_child_weight=q.min_child_weight,
            tree_method="hist",
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            n_jobs=self.cfg.n_jobs,
            random_state=seed,
        )

    def _regressor(self, seed: int) -> Any:
        if isinstance(self.cfg, RfConfig):
            from sklearn.ensemble import RandomForestRegressor

            p = self.cfg.regressor
            return RandomForestRegressor(
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                min_samples_leaf=p.min_samples_leaf,
                max_features=p.max_features,
                n_jobs=self.cfg.n_jobs,
                random_state=seed,
            )
        from xgboost import XGBRegressor

        q = self.cfg.regressor
        return XGBRegressor(
            n_estimators=q.n_estimators,
            max_depth=q.max_depth,
            learning_rate=q.learning_rate,
            subsample=q.subsample,
            colsample_bytree=q.colsample_bytree,
            min_child_weight=q.min_child_weight,
            objective="reg:absoluteerror",
            tree_method="hist",
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            n_jobs=self.cfg.n_jobs,
            random_state=seed,
        )

    # ------------------------------------------------------------------ training
    def _fit_head(
        self,
        seed: int,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[Any],
        w: npt.NDArray[np.float32],
        k: int,
        val: tuple[npt.NDArray[np.float32], npt.NDArray[Any]],
    ) -> tuple[_Classifier, dict[str, Any]]:
        y = y.astype(np.int64)
        cw = class_weights(y, w, k, self.cfg.class_weight)
        present = np.flatnonzero(np.bincount(y, minlength=k) > 0)
        info: dict[str, Any] = {"class_weights": cw.tolist()}
        if present.size < 2:
            return _Classifier(None, present, k), info
        enc = np.full(k, -1, dtype=np.int64)
        enc[present] = np.arange(present.size)
        model = self._classifier(seed)
        sw = effective_weights(y, w, cw)
        if isinstance(self.cfg, XgbConfig):
            xv, yv = val
            keep = np.isin(yv.astype(np.int64), present)
            yv_enc = enc[yv.astype(np.int64)[keep]]
            if not keep.any() or np.unique(yv_enc).size < 1:
                model.set_params(early_stopping_rounds=None)
                model.fit(x, enc[y], sample_weight=sw, verbose=False)
                return _Classifier(model, present, k), info
            model.fit(
                x,
                enc[y],
                sample_weight=sw,
                eval_set=[(xv[keep], yv_enc)],
                sample_weight_eval_set=[cw[yv.astype(np.int64)[keep]]],
                verbose=False,
            )
            info["best_iteration"] = int(model.best_iteration)
        else:
            model.fit(x, enc[y], sample_weight=sw)
        return _Classifier(model, present, k), info

    def fit(self, train: SplitData, val: SplitData, seed: int) -> dict[str, Any]:
        self.horizons = train.horizons
        t0 = time.perf_counter()
        x, xv = train.summary(), val.summary()
        w = train.weight
        info: dict[str, Any] = {"n_train": int(x.shape[0]), "n_features": int(x.shape[1])}
        for h in train.horizons:
            self.heads[f"h{h}"], info[f"h{h}"] = self._fit_head(
                seed, x, train.target(h), w, len(HEALTH), (xv, val.target(h))
            )
        self.heads["fault_type"], info["fault_type"] = self._fit_head(
            seed,
            x,
            train.at_end("fault_type"),
            w,
            self.n_fault_types,
            (xv, val.at_end("fault_type")),
        )
        known = ~train.at_end("ttf_censored")
        known_v = ~val.at_end("ttf_censored")
        reg = self._regressor(seed)
        if isinstance(self.cfg, XgbConfig) and not known_v.any():
            reg.set_params(early_stopping_rounds=None)
            reg.fit(x[known], train.at_end("ttf")[known], sample_weight=w[known], verbose=False)
        elif isinstance(self.cfg, XgbConfig):
            reg.fit(
                x[known],
                train.at_end("ttf")[known],
                sample_weight=w[known],
                eval_set=[(xv[known_v], val.at_end("ttf")[known_v])],
                verbose=False,
            )
            info["ttf_best_iteration"] = int(reg.best_iteration)
        else:
            reg.fit(x[known], train.at_end("ttf")[known], sample_weight=w[known])
        self.regressor = reg
        info["ttf_train_rows"] = int(known.sum())
        info["fit_s"] = round(time.perf_counter() - t0, 2)
        return info

    # ------------------------------------------------------------------ inference
    def _predict_x(self, x: npt.NDArray[np.float32]) -> Predictions:
        ttf = np.clip(self.regressor.predict(x), 0.0, self.ttf_cap_s).astype(np.float32)
        return Predictions(
            probs={h: self.heads[f"h{h}"].proba(x) for h in self.horizons},
            ttf=ttf,
            fault_type=self.heads["fault_type"].proba(x),
        )

    def predict(self, split: SplitData) -> Predictions:
        return self._predict_x(split.summary())

    def predict_one(self, split: SplitData, i: int) -> None:
        self._predict_x(summary_features(*split.gather(np.array([i])), schema=split.schema))

    def set_threads(self, n: int) -> None:
        models = [c.model for c in self.heads.values() if c.model is not None] + [self.regressor]
        for m in models:
            if isinstance(self.cfg, XgbConfig):
                m.set_params(n_jobs=n)
            else:
                m.n_jobs = n

    def load(self, model_dir: Path, train: SplitData) -> None:
        """Restore saved XGBoost heads (Phase 8). Random Forests are never saved.

        The classes each head was trained on are not stored with the model; they are
        recomputed from the training split exactly as :meth:`fit` saw them. A head missing
        from ``model_dir`` was trained on a single class (constant prediction).
        """
        if not isinstance(self.cfg, XgbConfig):
            raise TypeError("only XGBoost models are saved; retrain Random Forests from the seed")
        from xgboost import XGBClassifier, XGBRegressor

        self.horizons = train.horizons
        targets: dict[str, tuple[npt.NDArray[Any], int]] = {
            f"h{h}": (train.target(h), len(HEALTH)) for h in train.horizons
        }
        targets["fault_type"] = (train.at_end("fault_type"), self.n_fault_types)
        for name, (y, k) in targets.items():
            present = np.flatnonzero(np.bincount(y.astype(np.int64), minlength=k) > 0)
            path = model_dir / f"{name}.json"
            if present.size < 2:
                self.heads[name] = _Classifier(None, present, k)
                continue
            if not path.is_file():
                raise FileNotFoundError(f"saved head {path} is missing")
            model = XGBClassifier()
            model.load_model(path)
            self.heads[name] = _Classifier(model, present, k)
        reg = XGBRegressor()
        reg.load_model(model_dir / "ttf.json")
        self.regressor = reg

    def save(self, out_dir: Path) -> list[Path]:
        """XGBoost heads as JSON; Random Forests are not saved (hundreds of MB, ADR 0020)."""
        if not isinstance(self.cfg, XgbConfig):
            return []
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, c in self.heads.items():
            if c.model is not None:
                p = out_dir / f"{name}.json"
                c.model.save_model(p)
                paths.append(p)
        p = out_dir / "ttf.json"
        self.regressor.save_model(p)
        paths.append(p)
        return paths

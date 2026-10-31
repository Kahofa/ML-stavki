"""Gradient boosting (LightGBM) на признаках матча -> вероятности 1X2."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURES: list[str] = [
    "elo_diff", "clubelo_diff", "dc_lam", "dc_mu", "dc_ph", "dc_pa",
    *[f"{side}_{s}_{n}" for side in ("h", "a") for s in ("xgf", "xga", "gf", "ga", "pts") for n in (5, 10)],
    *[f"h_{s}_venue5" for s in ("xgf", "xga", "pts")], *[f"a_{s}_venue5" for s in ("xgf", "xga", "pts")],
    "h_rest_days", "a_rest_days", "h_games_14d", "a_games_14d",
    "pos_h", "pos_a", "releg_gap_h", "releg_gap_a", "top_gap_h", "top_gap_a", "progress",
    "releg_battle_h", "releg_battle_a", "title_race_h", "title_race_a", "derby",
]

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=15, min_data_in_leaf=80,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
              num_threads=4, seed=42)


@dataclass
class GBMModel:
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    booster: lgb.Booster | None = None
    best_iter: int = 0

    def fit(self, train: pd.DataFrame, y: np.ndarray, valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None, num_rounds: int = 1500) -> "GBMModel":
        dtr = lgb.Dataset(train[self.features].astype(float), label=y)
        if valid is not None and len(valid):
            dva = lgb.Dataset(valid[self.features].astype(float), label=y_valid, reference=dtr)
            self.booster = lgb.train(PARAMS, dtr, num_rounds, valid_sets=[dva],
                                     callbacks=[lgb.early_stopping(100, verbose=False)])
            self.best_iter = self.booster.best_iteration or num_rounds
            # переобучаем на train+valid с найденным числом итераций
            full = pd.concat([train, valid])
            yy = np.concatenate([y, y_valid])
            self.booster = lgb.train(PARAMS, lgb.Dataset(full[self.features].astype(float), label=yy),
                                     max(50, int(self.best_iter * 1.1)))
        else:
            self.booster = lgb.train(PARAMS, dtr, 400)
            self.best_iter = 400
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.booster is not None, "Модель не обучена"
        return self.booster.predict(X[self.features].astype(float))

    def importance(self, top: int = 10) -> list[tuple[str, float]]:
        assert self.booster is not None
        imp = self.booster.feature_importance("gain")
        order = np.argsort(imp)[::-1][:top]
        tot = imp.sum() or 1.0
        return [(self.features[i], float(imp[i] / tot)) for i in order]

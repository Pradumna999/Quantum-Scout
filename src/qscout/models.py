"""Forecasters. All predict the change in log value from the origin to cycle k+h.

A prediction of 0 is therefore exactly the persistence ("last value carried forward") forecast.
"""

from __future__ import annotations

from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .features import FEATURE_COLS


class Persistence:
    name = "persist"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Persistence:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class Ewma:
    """Forecast = EWMA(span 7) of the log series, i.e. delta = -(y0 - ewma)."""

    name = "ewma"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Ewma:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -X["dev_ewma7"].fillna(0.0).to_numpy()


class RollingMedian:
    """Forecast = rolling median of the last 60 cycles."""

    name = "rollmed"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RollingMedian:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -X["dev_med60"].fillna(0.0).to_numpy()


class LGBMQuantile:
    """One LightGBM model per quantile, trained on the delta target with a pinball loss."""

    name = "lgbm"

    def __init__(self, cfg: Config, quantiles: Sequence[float] | None = None):
        self.cfg = cfg
        self.quantiles = tuple(quantiles if quantiles is not None else cfg.quantiles)
        self.models_: list = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LGBMQuantile:
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            raise ValueError("too few labelled rows to fit a model")
        Xf = X.loc[ok, FEATURE_COLS]
        self.models_ = []
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                random_state=self.cfg.seed,
                verbose=-1,
                n_jobs=1,
                **self.cfg.lgbm,
            )
            m.fit(Xf, y[ok])
            self.models_.append(m)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """(n, n_quantiles), sorted per row so quantiles never cross."""
        Xf = X[FEATURE_COLS]
        out = np.column_stack([m.predict(Xf) for m in self.models_])
        return np.sort(out, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        q = self.predict_quantiles(X)
        return q[:, self.quantiles.index(0.5)]

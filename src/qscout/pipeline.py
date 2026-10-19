"""One fitted forecasting + uncertainty + trust stack per horizon.

Research code (`experiments.py`) and production code (`serve.py`) both go through
`fit_horizon` / `HorizonPipeline.predict`, so what is evaluated is what is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .conformal import GroupedCQR, GroupedSplitConformal
from .features import target_col
from .models import Ewma, LGBMQuantile, RollingMedian
from .schema import SERIES_KEYS
from .trust import TrustScorer

_TRUST_INPUT_COLS = ["ts", "absd_mean10", "vol_ratio", "tsl", "ph_stat", "xs_jump_frac"]


def series_thresholds(train: pd.DataFrame, h: int, pct: float, min_rows: int = 30):
    """Per-series `pct` quantile of |persistence error| on training rows, plus metric-level
    fallback. This defines what counts as a "large" error for a given series."""
    d = train[SERIES_KEYS].copy()
    d["e"] = train[target_col(h)].abs().to_numpy()
    g = d.groupby(SERIES_KEYS)["e"]
    per = g.quantile(pct)[g.size() >= min_rows].rename("thr").reset_index()
    metric = d.groupby("metric")["e"].quantile(pct).to_dict()
    overall = float(d["e"].quantile(pct))
    return per, metric, overall


@dataclass
class HorizonPipeline:
    h: int
    cfg: Config
    lgbm: LGBMQuantile
    cqr: GroupedCQR
    pconf_norm: GroupedSplitConformal
    pconf_abs: GroupedSplitConformal
    lconf_norm: GroupedSplitConformal
    thr_series: pd.DataFrame
    thr_metric: dict
    thr_overall: float
    scale_fb: dict
    ewma: Ewma = field(default_factory=Ewma)
    rollmed: RollingMedian = field(default_factory=RollingMedian)
    trust: TrustScorer | None = None

    # -- helpers --------------------------------------------------------------------------
    def scale(self, df: pd.DataFrame) -> np.ndarray:
        s = df["absd_mean10"].to_numpy(dtype=float)
        fb = df["metric"].map(self.scale_fb).to_numpy(dtype=float)
        s = np.where(np.isfinite(s), s, fb)
        return np.where(np.isfinite(s), s, 0.0) + self.cfg.scale_floor

    def large_error_threshold(self, df: pd.DataFrame) -> np.ndarray:
        m = df[SERIES_KEYS].merge(self.thr_series, how="left", on=SERIES_KEYS)["thr"].to_numpy(float)
        fb = df["metric"].map(self.thr_metric).to_numpy(dtype=float)
        out = np.where(np.isfinite(m), m, fb)
        return np.where(np.isfinite(out), out, self.thr_overall)

    def trust_frame(self, df: pd.DataFrame, width: np.ndarray, disagreement: np.ndarray) -> pd.DataFrame:
        tf = df[SERIES_KEYS + _TRUST_INPUT_COLS].reset_index(drop=True).copy()
        tf["width"] = np.asarray(width, dtype=float)
        tf["disagreement"] = np.asarray(disagreement, dtype=float)
        return tf

    def _disagreement(self, df: pd.DataFrame, med: np.ndarray) -> np.ndarray:
        stack = np.column_stack(
            [np.zeros(len(df)), self.ewma.predict(df), self.rollmed.predict(df), med]
        )
        return stack.std(axis=1) / self.scale(df)

    # -- inference ------------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        q = self.lgbm.predict_quantiles(df)
        med = q[:, 1]
        groups = df["metric"].to_numpy()
        scale = self.scale(df)
        zeros = np.zeros(len(df))
        out = pd.DataFrame(index=df.index)
        out["p_persist"] = zeros
        out["p_ewma"] = self.ewma.predict(df)
        out["p_rollmed"] = self.rollmed.predict(df)
        out["p_lgbm"] = med
        out["q_lo"], out["q_hi"] = q[:, 0], q[:, 2]
        out["lo_cqr"], out["hi_cqr"] = self.cqr.interval(q[:, 0], q[:, 2], groups)
        out["lo_pconf"], out["hi_pconf"] = self.pconf_norm.interval(zeros, groups, scale=scale)
        out["lo_pconf_abs"], out["hi_pconf_abs"] = self.pconf_abs.interval(zeros, groups)
        out["lo_lconf"], out["hi_lconf"] = self.lconf_norm.interval(med, groups, scale=scale)
        out["width_cqr"] = out["hi_cqr"] - out["lo_cqr"]
        out["disagreement"] = self._disagreement(df, med)
        tf = self.trust_frame(df, out["width_cqr"].to_numpy(), out["disagreement"].to_numpy())
        score = self.trust.predict_proba(tf)
        out["trust"] = score
        out["tier"] = self.trust.tier(score)
        out["thr_log"] = self.large_error_threshold(df)
        return out


def fit_horizon(train: pd.DataFrame, calib: pd.DataFrame, h: int, cfg: Config) -> HorizonPipeline:
    """Fit the point/quantile model on `train`, then calibrate intervals and the trust scorer on
    `calib`. The two row sets must not overlap in time (see evaluation.splits)."""
    tcol = target_col(h)
    lgbm = LGBMQuantile(cfg).fit(train, train[tcol].to_numpy())
    thr_series, thr_metric, thr_overall = series_thresholds(train, h, cfg.large_error_pct)
    scale_fb = calib.groupby("metric")["absd_mean10"].median().to_dict()
    pipe = HorizonPipeline(
        h=h,
        cfg=cfg,
        lgbm=lgbm,
        cqr=GroupedCQR(),
        pconf_norm=GroupedSplitConformal(),
        pconf_abs=GroupedSplitConformal(),
        lconf_norm=GroupedSplitConformal(),
        thr_series=thr_series,
        thr_metric=thr_metric,
        thr_overall=thr_overall,
        scale_fb=scale_fb,
    )
    q = lgbm.predict_quantiles(calib)
    med = q[:, 1]
    y = calib[tcol].to_numpy(dtype=float)
    groups = calib["metric"].to_numpy()
    scale = pipe.scale(calib)
    zeros = np.zeros(len(calib))
    pipe.cqr.fit(q[:, 0], q[:, 2], y, groups, cfg.alpha, cfg.min_group)
    pipe.pconf_norm.fit(zeros, y, groups, cfg.alpha, cfg.min_group, scale=scale)
    pipe.pconf_abs.fit(zeros, y, groups, cfg.alpha, cfg.min_group)
    pipe.lconf_norm.fit(med, y, groups, cfg.alpha, cfg.min_group, scale=scale)

    lo, hi = pipe.cqr.interval(q[:, 0], q[:, 2], groups)
    tf = pipe.trust_frame(calib, hi - lo, pipe._disagreement(calib, med))
    label = np.abs(y - med) > pipe.large_error_threshold(calib)
    pipe.trust = TrustScorer(cfg.trust_fprs, seed=cfg.seed, scale_floor=cfg.scale_floor).fit(
        tf, label, calib["ts"].reset_index(drop=True)
    )
    return pipe

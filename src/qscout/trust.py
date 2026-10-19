"""Trust scoring: estimate P(large forecast error) from drift/uncertainty signals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .schema import SERIES_KEYS

TRUST_FEATURES: list[str] = [
    "width_ratio",
    "scale_ratio",
    "vol_ratio",
    "tsl",
    "ph_stat",
    "xs_jump_frac",
    "disagreement",
]
TIERS = ("ok", "caution", "unreliable")
_REF_COLS = ["width_ref", "scale_ref"]


class TrustScorer:
    """Logistic model on standardized signals. Thresholds for the caution / unreliable tiers are
    chosen so that the share of *non-large-error* rows flagged equals the configured false-alarm
    rates. The scorer is fit on the earlier 60% (by time) of the calibration rows and the
    thresholds are set on the later 40%, so thresholds are not tuned on the fitting data. With
    fewer than `min_positives` large-error rows to learn from, the scorer falls back to a constant
    (no signal) instead of fitting noise. Rows are strongly cross-correlated in time (qubits
    share device events), so the *effective* sample is far smaller than the row count: use
    calibration windows of many cycles."""

    def __init__(
        self,
        fprs: tuple[float, float] = (0.25, 0.10),
        features: Sequence[str] = TRUST_FEATURES,
        seed: int = 0,
        scale_floor: float = 0.005,
        min_positives: int = 30,
        C: float = 0.1,
    ):
        self.min_positives = min_positives
        self.C = C
        self.fprs = fprs
        self.features = list(features)
        self.seed = seed
        self.scale_floor = scale_floor

    # -- references -----------------------------------------------------------------------
    def _fit_refs(self, df: pd.DataFrame) -> None:
        g = df.groupby(SERIES_KEYS)
        self.ref_ = pd.DataFrame(
            {"width_ref": g["width"].median(), "scale_ref": g["absd_mean10"].median()}
        ).reset_index()
        gm = df.groupby("metric")
        self.ref_metric_ = pd.DataFrame(
            {"width_ref": gm["width"].median(), "scale_ref": gm["absd_mean10"].median()}
        )

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        m = df[SERIES_KEYS].merge(self.ref_, how="left", on=SERIES_KEYS)
        for c in _REF_COLS:
            fb = df["metric"].map(self.ref_metric_[c]).to_numpy()
            m[c] = m[c].to_numpy(dtype=float)
            m[c] = np.where(np.isfinite(m[c]), m[c], fb)
        floor = self.scale_floor
        w = np.maximum(df["width"].to_numpy(dtype=float), 1e-9)
        X = pd.DataFrame(index=range(len(df)))
        X["width_ratio"] = np.log(w / np.maximum(m["width_ref"].to_numpy(), 1e-9))
        X["scale_ratio"] = np.log(
            (df["absd_mean10"].to_numpy(dtype=float) + floor) / (m["scale_ref"].to_numpy() + floor)
        )
        X["vol_ratio"] = df["vol_ratio"].to_numpy(dtype=float).clip(0, 10)
        X["tsl"] = np.log1p(df["tsl"].to_numpy(dtype=float))
        X["ph_stat"] = df["ph_stat"].to_numpy(dtype=float).clip(0, 50)
        X["xs_jump_frac"] = df["xs_jump_frac"].to_numpy(dtype=float)
        X["disagreement"] = df["disagreement"].to_numpy(dtype=float).clip(0, 10)
        X = X.replace([np.inf, -np.inf], np.nan)
        neutral = {"vol_ratio": 1.0}
        for c in X.columns:
            X[c] = X[c].fillna(neutral.get(c, 0.0))
        return X[self.features]

    # -- fit / predict --------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, label: np.ndarray, ts: pd.Series) -> TrustScorer:
        label = np.asarray(label).astype(int)
        self._fit_refs(df)
        X = self._design(df).to_numpy()
        t = pd.to_datetime(ts).to_numpy()
        uniq = np.sort(np.unique(t))
        cut = uniq[max(1, int(0.6 * len(uniq))) - 1]
        fit_m, thr_m = t <= cut, t > cut
        self.constant_: float | None = None
        if len(np.unique(label[fit_m])) < 2 or label[fit_m].sum() < self.min_positives:
            self.constant_ = float(label[fit_m].mean()) if fit_m.any() else 0.0
            self.model_ = None
        else:
            self.model_ = make_pipeline(
                StandardScaler(), LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
            ).fit(X[fit_m], label[fit_m])
        s = self._score(X[thr_m]) if thr_m.any() else self._score(X)
        y_thr = label[thr_m] if thr_m.any() else label
        self.val_auroc_ = auroc(y_thr, s)  # skill on the held-back calibration slice
        neg = s[y_thr == 0] if (y_thr == 0).any() else s
        self.thr_caution_ = float(np.quantile(neg, 1 - self.fprs[0]))
        self.thr_unreliable_ = float(np.quantile(neg, 1 - self.fprs[1]))
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            return np.full(len(X), self.constant_)
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._score(self._design(df).to_numpy())

    def tier(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score)
        return np.where(
            score >= self.thr_unreliable_,
            "unreliable",
            np.where(score >= self.thr_caution_, "caution", "ok"),
        )


# -- evaluation helpers -------------------------------------------------------------------


def auroc(label: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC of `score` for the binary `label`; rows with a non-finite score are dropped."""
    label = np.asarray(label).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    label, score = label[ok], score[ok]
    if len(np.unique(label)) < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def risk_coverage(score: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: keep the most trusted rows first (lowest score)."""
    order = np.argsort(score, kind="stable")
    e = np.asarray(err, dtype=float)[order]
    n = len(e)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    return coverage, risk


def aurc(score: np.ndarray, err: np.ndarray) -> float:
    cov, risk = risk_coverage(score, err)
    return float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)

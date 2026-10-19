"""Split conformal, CQR and Adaptive Conformal Inference for panel forecasts.

Calibration is Mondrian: scores are pooled within a group (here: the metric), because error
scales differ by orders of magnitude between, say, T1 and a two-qubit gate error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def conformal_quantile(scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of calibration scores."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("inf")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def level_quantile(sorted_scores: np.ndarray, alpha_t: float, alpha: float | None = None) -> float:
    """Quantile used by ACI at miscoverage level alpha_t.

    For 0 < alpha_t the usual conformal quantile (clipped at the largest score). Textbook ACI
    returns an infinite interval once alpha_t <= 0; to keep intervals finite and usable we instead
    extrapolate beyond the largest calibration score by one "tail spread" (max - median) per
    `alpha` of negative alpha_t. The two definitions agree at alpha_t = 0, and the interval can
    still grow without bound, which is what lets alpha_t return to its target after a large shift.
    """
    n = len(sorted_scores)
    level = float(np.clip(np.ceil((n + 1) * (1 - alpha_t)) / n, 0.0, 1.0))
    q = float(np.quantile(sorted_scores, level, method="higher"))
    if alpha_t < 0 and alpha:
        spread = max(float(sorted_scores[-1] - np.median(sorted_scores)), 1e-9)
        q += spread * (-alpha_t) / alpha
    return q


class _Grouped:
    alpha: float

    def _fit_scores(self, scores: np.ndarray, groups: np.ndarray, alpha: float, min_group: int):
        self.alpha = alpha
        scores = np.asarray(scores, dtype=float)
        groups = np.asarray(groups).astype(str)
        fin = np.isfinite(scores)
        self._global = np.sort(scores[fin])
        if len(self._global) == 0:
            raise ValueError("no finite calibration scores")
        self._by_group: dict[str, np.ndarray] = {}
        for g in np.unique(groups):
            s = scores[(groups == g) & fin]
            if len(s) >= min_group:
                self._by_group[g] = np.sort(s)

    def scores_for(self, group: str) -> np.ndarray:
        return self._by_group.get(str(group), self._global)

    def _q(self, groups: np.ndarray, alpha: float | None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        groups = np.asarray(groups).astype(str)
        out = np.empty(len(groups))
        for g in np.unique(groups):
            out[groups == g] = conformal_quantile(self.scores_for(g), a)
        return out


class GroupedSplitConformal(_Grouped):
    """Intervals point +/- q * scale with q the conformal quantile of |y - point| / scale."""

    def fit(self, point, y, groups, alpha: float, min_group: int = 50, scale=None):
        point, y = np.asarray(point, float), np.asarray(y, float)
        scale = np.ones_like(point) if scale is None else np.asarray(scale, float)
        self._fit_scores(np.abs(y - point) / scale, groups, alpha, min_group)
        return self

    def interval(self, point, groups, alpha: float | None = None, scale=None):
        point = np.asarray(point, float)
        scale = np.ones_like(point) if scale is None else np.asarray(scale, float)
        q = self._q(groups, alpha) * scale
        return point - q, point + q


class GroupedCQR(_Grouped):
    """Conformalized quantile regression: widen (or shrink) [lo, hi] by the conformal quantile of
    max(lo - y, y - hi)."""

    def fit(self, lo, hi, y, groups, alpha: float, min_group: int = 50):
        lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
        self._fit_scores(np.maximum(lo - y, y - hi), groups, alpha, min_group)
        return self

    def interval(self, lo, hi, groups, alpha: float | None = None):
        q = self._q(groups, alpha)
        return np.asarray(lo, float) - q, np.asarray(hi, float) + q


def aci_cqr(
    cqr: GroupedCQR,
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    origin_ns: np.ndarray,
    target_ns: np.ndarray,
    stream: np.ndarray,
    group: np.ndarray,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive Conformal Inference (Gibbs & Candes 2021) on top of CQR, with delayed feedback.

    One alpha_t state is kept per `stream` (e.g. backend|metric). All rows issued at the same
    origin time update it jointly, using their mean miscoverage (a panel approximation of the
    single-stream algorithm). The outcome of a forecast issued at origin o becomes available only
    at its target time, so alpha_t at origin t uses feedback solely from origins whose target
    time is <= t. Returns (lo_aci, hi_aci, alpha_t used for each row). See `level_quantile` for how
    intervals are kept finite when alpha_t goes negative.
    """
    lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
    origin_ns = np.asarray(origin_ns, dtype="int64")
    target_ns = np.asarray(target_ns, dtype="int64")
    stream = np.asarray(stream).astype(str)
    group = np.asarray(group).astype(str)
    out_lo, out_hi = lo.copy(), hi.copy()
    alpha_used = np.full(len(lo), alpha)

    for s in pd.unique(stream):
        idx = np.flatnonzero(stream == s)
        scores = cqr.scores_for(group[idx[0]])
        order = np.argsort(origin_ns[idx], kind="stable")
        idx = idx[order]
        times, starts = np.unique(origin_ns[idx], return_index=True)
        blocks = np.split(idx, starts[1:])
        tgt = np.array([target_ns[b[0]] for b in blocks])
        miss = np.full(len(blocks), np.nan)
        a, p = alpha, 0
        for i, rows in enumerate(blocks):
            while p < i and tgt[p] <= times[i]:
                if np.isfinite(miss[p]):
                    a += gamma * (alpha - miss[p])
                p += 1
            q = level_quantile(scores, a, alpha)
            out_lo[rows] = lo[rows] - q
            out_hi[rows] = hi[rows] + q
            alpha_used[rows] = a
            yy = y[rows]
            ok = np.isfinite(yy)
            if ok.any():
                miss[i] = float(np.mean((yy[ok] < out_lo[rows][ok]) | (yy[ok] > out_hi[rows][ok])))
    return out_lo, out_hi, alpha_used

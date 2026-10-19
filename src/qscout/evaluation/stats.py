"""Diebold-Mariano test, Holm correction and moving-block bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps


def diebold_mariano(d: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test on a loss-differential series d_t = L(model B) - L(model A) (positive: A better).

    HAC variance with a Bartlett kernel and lag h-1, with the Harvey-Leybourne-Newbold
    small-sample correction; p-value from a Student-t with T-1 d.o.f. (two-sided).
    Returns (statistic, p_value).
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 8:
        return float("nan"), float("nan")
    mean = d.mean()
    dc = d - mean
    lag = max(h - 1, 0)
    gamma0 = float(np.dot(dc, dc) / T)
    var = gamma0
    for k in range(1, lag + 1):
        gk = float(np.dot(dc[k:], dc[:-k]) / T)
        var += 2.0 * (1.0 - k / (lag + 1)) * gk
    var = max(var, 1e-18)
    dm = mean / np.sqrt(var / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = float(dm * hln)
    p = float(2.0 * sps.t.sf(abs(stat), df=T - 1))
    return stat, p


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are passed through and excluded from the family)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[ok][i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def block_bootstrap_ci(
    df: pd.DataFrame,
    time_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    block_len: int,
    n_boot: int,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Moving-block bootstrap over *time*: whole time slices (all rows sharing a timestamp) are
    resampled in blocks of consecutive timestamps, preserving both cross-sectional dependence
    (qubits move together) and serial dependence. Returns (point, lo, hi)."""
    point = float(stat_fn(df))
    times = np.sort(df[time_col].unique())
    T = len(times)
    if T < 2 or n_boot <= 0:
        return point, float("nan"), float("nan")
    groups = {t: idx for t, idx in df.groupby(time_col).indices.items()}
    keys = list(times)
    row_sets = [groups[k] for k in keys]
    rng = np.random.default_rng(seed)
    bl = max(1, min(block_len, T))
    n_blocks = int(np.ceil(T / bl))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - bl + 1, n_blocks)
        pick = np.concatenate([np.arange(s, s + bl) for s in starts])[:T]
        rows = np.concatenate([row_sets[i] for i in pick])
        v = stat_fn(df.iloc[rows])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    a = (1 - level) / 2
    return point, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))

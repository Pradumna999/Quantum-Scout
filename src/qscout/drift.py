"""Causal drift primitives: robust jump flags, volatility z-scores and Page-Hinkley.

Every function here uses only data up to and including time t when producing the value at t,
which is what makes the feature layer leakage-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826


def robust_diffs(
    logy: pd.DataFrame, window: int, floor: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(d1, |d1|, scale). `scale` is a robust sd of past |d1| (strictly before t), floored."""
    d1 = logy.diff()
    absd = d1.abs()
    scale = (
        absd.rolling(window, min_periods=max(5, window // 4)).median().shift(1) * MAD_TO_SD
    ).clip(lower=floor)
    return d1, absd, scale


def jump_flags(absd: pd.DataFrame, scale: pd.DataFrame, k: float) -> pd.DataFrame:
    """True where the one-step change exceeds k robust standard deviations."""
    return absd > (k * scale)


def volatility_z(absd: pd.DataFrame, floor: float, span: int = 120, cap: float = 5.0) -> pd.DataFrame:
    """|d1| relative to its own slow EWMA (strictly past). Mean ~1 in a stationary regime."""
    ref = absd.ewm(span=span, min_periods=10, adjust=False).mean().shift(1).clip(lower=floor)
    return (absd / ref).clip(upper=cap)


def page_hinkley(
    x: np.ndarray, delta: float, lam: float, min_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Page-Hinkley test for an increase in the mean of a 1-D stream.

    Returns (statistic, alarm). NaN inputs are skipped (state is carried). After an alarm the
    detector resets; the statistic reported at the alarm step is the value that crossed lam.
    """
    n_t = len(x)
    stat = np.zeros(n_t)
    alarm = np.zeros(n_t, dtype=bool)
    n = 0
    mean = 0.0
    m = 0.0
    m_min = 0.0
    s = 0.0
    for t in range(n_t):
        v = x[t]
        if v == v:  # not NaN
            n += 1
            mean += (v - mean) / n
            m += v - mean - delta
            m_min = min(m_min, m)
            s = m - m_min
            if n >= min_n and s > lam:
                alarm[t] = True
                stat[t] = s
                n, mean, m, m_min, s = 0, 0.0, 0.0, 0.0, 0.0
                continue
        stat[t] = s
    return stat, alarm


def page_hinkley_frame(
    z: pd.DataFrame, delta: float, lam: float, min_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arr = z.to_numpy(dtype=float)
    stat = np.zeros_like(arr)
    alarm = np.zeros(arr.shape, dtype=bool)
    for j in range(arr.shape[1]):
        stat[:, j], alarm[:, j] = page_hinkley(arr[:, j], delta, lam, min_n)
    return (
        pd.DataFrame(stat, index=z.index, columns=z.columns),
        pd.DataFrame(alarm, index=z.index, columns=z.columns),
    )


def pelt_changepoints(x: np.ndarray, penalty: float = 3.0, min_size: int = 5) -> list[int]:
    """Offline change-points (uses the whole series, so it sees the future). Needs `ruptures`.

    Used only to build post-hoc event annotations, never as a model feature.
    """
    import ruptures as rpt

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 * min_size:
        return []
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(x.reshape(-1, 1))
    sigma = float(np.median(np.abs(np.diff(x)))) * MAD_TO_SD or 1.0
    bkps = algo.predict(pen=penalty * sigma**2 * np.log(len(x)))
    return [b for b in bkps if b < len(x)]

"""Point-in-time-correct feature construction.

All rolling operations are trailing (window ends at t, inclusive) so the feature row at origin
cycle t depends only on observations with timestamp <= t. `as_of` truncates the input first, and
the test-suite asserts that features computed on a truncated panel equal the prefix of features
computed on the full panel, and that perturbing the future leaves them unchanged.

Targets (`target_h{h}`, `ts_target_h{h}`) are the only columns that look forward and are never
included in FEATURE_COLS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import Config
from .drift import jump_flags, page_hinkley_frame, robust_diffs, volatility_z
from .schema import SERIES_KEYS
from .transforms import ENTITY_TYPE, METRIC_ID, METRICS, to_log

FEATURE_COLS: list[str] = [
    "y0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "dev_mean3",
    "dev_mean7",
    "dev_mean14",
    "std7",
    "std30",
    "dev_ewma7",
    "dev_med60",
    "vol_ratio",
    "absd_mean10",
    "tsl",
    "ph_stat",
    "xs_med_d1",
    "xs_jump_frac",
    "hours_since_prev",
    "metric_id",
]


def target_col(h: int) -> str:
    return f"target_h{h}"


def target_ts_col(h: int) -> str:
    return f"ts_target_h{h}"


def _broadcast(vec: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.repeat(vec.to_numpy(dtype=float)[:, None], like.shape[1], axis=1),
        index=like.index,
        columns=like.columns,
    )


def _block_features(
    logy: pd.DataFrame, ts_all: pd.DatetimeIndex, cfg: Config
) -> dict[str, pd.DataFrame]:
    d1, absd, scale = robust_diffs(logy, cfg.jump_window, cfg.scale_floor)
    jump = jump_flags(absd, scale, cfg.jump_k)
    T = len(logy)

    f: dict[str, pd.DataFrame] = {"y0": logy, "d1": d1}
    for i in range(2, 6):
        f[f"d{i}"] = d1.shift(i - 1)
    for w in (3, 7, 14):
        f[f"dev_mean{w}"] = logy - logy.rolling(w, min_periods=2).mean()
    f["std7"] = logy.rolling(7, min_periods=3).std()
    f["std30"] = logy.rolling(30, min_periods=5).std()
    f["dev_ewma7"] = logy - logy.ewm(span=7, min_periods=2, adjust=False).mean()
    f["dev_med60"] = logy - logy.rolling(60, min_periods=10).median()

    vol_s = d1.rolling(7, min_periods=4).std()
    vol_l = d1.rolling(60, min_periods=15).std()
    f["vol_ratio"] = vol_s / vol_l.where(vol_l > 0)
    f["absd_mean10"] = absd.rolling(10, min_periods=4).mean()

    J = jump.to_numpy()
    idx = np.where(J, np.arange(T)[:, None], np.nan)
    last = pd.DataFrame(idx, index=logy.index, columns=logy.columns).ffill().to_numpy()
    tsl = np.where(np.isnan(last), cfg.tsl_cap, np.arange(T)[:, None] - last)
    f["tsl"] = pd.DataFrame(np.minimum(tsl, cfg.tsl_cap), index=logy.index, columns=logy.columns)

    z = volatility_z(absd, cfg.scale_floor)
    f["ph_stat"], _ = page_hinkley_frame(z, cfg.ph_delta, cfg.ph_lambda, cfg.ph_min_n)

    valid = absd.notna().sum(axis=1).replace(0, np.nan)
    f["xs_med_d1"] = _broadcast(d1.median(axis=1), logy)
    f["xs_jump_frac"] = _broadcast(jump.sum(axis=1) / valid, logy)

    gap_h = pd.Series(ts_all).diff().dt.total_seconds().to_numpy() / 3600.0
    f["hours_since_prev"] = pd.DataFrame(
        np.repeat(gap_h[:, None], logy.shape[1], axis=1), index=logy.index, columns=logy.columns
    )
    return f


def build_features(
    cal: pd.DataFrame,
    horizons: Sequence[int],
    cfg: Config,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per (series, origin cycle) with features, plus forward targets.

    Columns: SERIES_KEYS, ts, cycle, FEATURE_COLS, and for each h: target_h{h} (change in log
    value from origin to the h-th next calibration cycle) and ts_target_h{h}.
    """
    if as_of is not None:
        cal = cal[cal["snapshot_ts_utc"] <= pd.Timestamp(as_of)]
    parts: list[pd.DataFrame] = []
    for backend, g in cal.groupby("backend", sort=True):
        ts_all = pd.DatetimeIndex(sorted(g["snapshot_ts_utc"].unique()))
        T = len(ts_all)
        for metric in METRICS:
            gm = g[g["metric"] == metric]
            if gm.empty:
                continue
            wide = gm.pivot(index="snapshot_ts_utc", columns="entity_id", values="value")
            wide = wide.reindex(ts_all).sort_index(axis=1)
            logy = to_log(wide).astype(float)
            feats = _block_features(logy, ts_all, cfg)
            N = logy.shape[1]
            cols: dict[str, np.ndarray] = {
                name: frame.to_numpy(dtype=float).reshape(-1) for name, frame in feats.items()
            }
            y = logy.to_numpy(dtype=float)
            for h in horizons:
                fut = np.full_like(y, np.nan)
                if T > h:
                    fut[:-h] = y[h:] - y[:-h]
                cols[target_col(h)] = fut.reshape(-1)
            out = pd.DataFrame(cols)
            out["metric_id"] = float(METRIC_ID[metric])
            out["backend"] = backend
            out["metric"] = metric
            out["entity_type"] = ENTITY_TYPE[metric]
            out["entity_id"] = np.tile(logy.columns.to_numpy().astype(str), T)
            out["ts"] = ts_all.repeat(N)
            out["cycle"] = np.repeat(np.arange(T), N)
            ts_series = pd.Series(ts_all)
            for h in horizons:
                tt = ts_series.shift(-h)
                out[target_ts_col(h)] = tt.iloc[np.repeat(np.arange(T), N)].reset_index(drop=True)
            parts.append(out)
    if not parts:
        return pd.DataFrame(columns=SERIES_KEYS + ["ts", "cycle"] + FEATURE_COLS)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["y0"].notna() & (df["cycle"] >= cfg.min_history)]
    lead = SERIES_KEYS + ["ts", "cycle"]
    tail = [c for h in horizons for c in (target_col(h), target_ts_col(h))]
    return df[lead + FEATURE_COLS + tail].sort_values(["backend", "metric", "entity_id", "ts"]).reset_index(drop=True)


def latest_origin_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Rows at each backend's most recent calibration cycle (the nightly forecast origins)."""
    latest = feats.groupby("backend")["ts"].transform("max")
    return feats[feats["ts"] == latest].reset_index(drop=True)

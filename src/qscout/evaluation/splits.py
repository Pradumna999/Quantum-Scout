"""Leakage-controlled temporal splits with label purging.

Per backend, origin cycles are cut at fractions of its own timeline:

    |---- train ----|---- val (calibration [+ dev evaluation]) ----|---- test ----|
                    t_train_end                                    t_val_end

A row (origin ts, target ts_{k+h}) belongs to
  * train  iff its target is realized before t_train_end         (so no label touches val/test)
  * calib  iff origin >= t_train_end and its target is realized before the calibration end
  * eval   iff origin >= the calibration end

"dev" mode calibrates on the first half of val and evaluates on the second half; "final" mode
calibrates on all of val and evaluates on test. Holdout backends are never used for train/calib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..features import target_col, target_ts_col


@dataclass(frozen=True)
class Bounds:
    t_train_end: pd.Timestamp
    t_calib_mid: pd.Timestamp  # end of calibration window in dev mode
    t_val_end: pd.Timestamp


def backend_bounds(feats: pd.DataFrame, cfg: Config) -> dict[str, Bounds]:
    out: dict[str, Bounds] = {}
    tr, va = cfg.split_fracs
    for b, g in feats.groupby("backend"):
        ts = pd.DatetimeIndex(sorted(g["ts"].unique()))
        T = len(ts)
        i1 = int(T * tr)
        i2 = int(T * (tr + va))
        out[b] = Bounds(ts[i1], ts[(i1 + i2) // 2], ts[min(i2, T - 1)])
    return out


def _per_row(feats: pd.DataFrame, bounds: dict[str, Bounds], attr: str) -> pd.Series:
    mapping = {b: getattr(v, attr) for b, v in bounds.items()}
    return feats["backend"].map(mapping)


def window_masks(
    feats: pd.DataFrame,
    h: int,
    bounds: dict[str, Bounds],
    cfg: Config,
    mode: str = "dev",
) -> dict[str, np.ndarray]:
    if mode not in ("dev", "final"):
        raise ValueError("mode must be 'dev' or 'final'")
    tts = feats[target_ts_col(h)]
    has_t = feats[target_col(h)].notna() & tts.notna()
    ts = feats["ts"]
    t_train_end = _per_row(feats, bounds, "t_train_end")
    t_val_end = _per_row(feats, bounds, "t_val_end")
    t_calib_end = _per_row(feats, bounds, "t_calib_mid") if mode == "dev" else t_val_end
    holdout = feats["backend"].isin(cfg.holdout_backends)

    train = has_t & (tts < t_train_end) & ~holdout
    calib = has_t & (ts >= t_train_end) & (tts < t_calib_end) & ~holdout
    if mode == "dev":
        ev = has_t & (ts >= t_calib_end) & (tts < t_val_end)
    else:
        ev = has_t & (ts >= t_val_end)
    return {
        "train": train.to_numpy(),
        "calib": calib.to_numpy(),
        "eval": ev.to_numpy(),
        "holdout": holdout.to_numpy(),
    }


def rolling_origin_masks(
    feats: pd.DataFrame, h: int, bounds: dict[str, Bounds], cfg: Config, n_folds: int = 3
) -> list[dict[str, np.ndarray]]:
    """Expanding-window folds inside each backend's train+val span (test is never touched).

    Cut points divide [half of the train span, t_val_end] evenly in cycles. Fold f trains on rows
    whose target is realized before cut_f and evaluates on origins in [cut_f, cut_{f+1}) whose
    target is realized before cut_{f+1}. Holdout backends are excluded.
    """
    folds = []
    tts = feats[target_ts_col(h)]
    has_t = feats[target_col(h)].notna() & tts.notna()
    ts = feats["ts"]
    holdout = feats["backend"].isin(cfg.holdout_backends)
    cut_cols = []
    for b, g in feats.groupby("backend"):
        uniq = pd.DatetimeIndex(sorted(g["ts"].unique()))
        lo = int(len(uniq) * cfg.split_fracs[0] * 0.5)
        hi = min(int(len(uniq) * sum(cfg.split_fracs)), len(uniq) - 1)
        idx = np.linspace(lo, hi, n_folds + 2).astype(int)
        cut_cols.append((b, uniq[idx]))
    cuts = {b: c for b, c in cut_cols}
    for f in range(1, n_folds + 1):
        c0 = feats["backend"].map({b: c[f] for b, c in cuts.items()})
        c1 = feats["backend"].map({b: c[f + 1] for b, c in cuts.items()})
        folds.append(
            {
                "train": (has_t & (tts < c0) & ~holdout).to_numpy(),
                "eval": (has_t & (ts >= c0) & (tts < c1) & ~holdout).to_numpy(),
            }
        )
    return folds

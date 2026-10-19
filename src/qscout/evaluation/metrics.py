"""Forecast metrics. Errors live in log space, so they are scale-free across metrics."""

from __future__ import annotations

import numpy as np


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_score(y, lo, hi, alpha: float) -> np.ndarray:
    """Winkler/interval score (lower is better): width + (2/alpha) * miss distance."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return (hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))


def pinball(y, q, tau: float) -> np.ndarray:
    y, q = np.asarray(y, float), np.asarray(q, float)
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def point_summary(err: np.ndarray, err_persist: np.ndarray) -> dict[str, float]:
    err, err_persist = np.asarray(err, float), np.asarray(err_persist, float)
    mse, mse_p = float(np.mean(err**2)), float(np.mean(err_persist**2))
    mae, mae_p = float(np.mean(np.abs(err))), float(np.mean(np.abs(err_persist)))
    return {
        "n": int(len(err)),
        "mae_log": mae,
        "rmse_log": float(np.sqrt(mse)),
        "skill_mse": float(1.0 - mse / mse_p) if mse_p > 0 else float("nan"),
        "rel_mae": float(mae / mae_p) if mae_p > 0 else float("nan"),  # persistence-scaled MAE
    }

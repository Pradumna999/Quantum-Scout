"""Canonical long-format calibration schema and its validation contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .transforms import ENTITY_TYPE, METRICS

COLUMNS = [
    "backend",
    "snapshot_ts_utc",
    "entity_type",
    "entity_id",
    "metric",
    "value",
    "unit",
    "source",
]
KEY = ["backend", "snapshot_ts_utc", "entity_type", "entity_id", "metric"]
SERIES_KEYS = ["backend", "metric", "entity_type", "entity_id"]

_T_MAX_US = 1e4  # 10 ms: far above any transmon T1/T2, catches unit mistakes


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and column order. Raises if a required column is missing."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df[COLUMNS].copy()
    for c in ("backend", "entity_type", "entity_id", "metric", "unit", "source"):
        out[c] = out[c].astype(str)
    out["snapshot_ts_utc"] = pd.to_datetime(out["snapshot_ts_utc"], utc=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    return out.reset_index(drop=True)


def validate_calibration(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (clean, quarantined). Quarantined rows carry a `reason` column.

    NaN values mean "no measurement" and are dropped silently. Everything else that violates the
    contract is quarantined, never silently repaired.
    """
    df = normalize(df)
    df = df[df["value"].notna()].reset_index(drop=True)
    reason = np.full(len(df), "", dtype=object)

    def mark(mask: pd.Series | np.ndarray, why: str) -> None:
        m = np.asarray(mask, dtype=bool) & (reason == "")
        reason[m] = why

    is_t = df["metric"].isin(["t1", "t2"]).to_numpy()
    vals = df["value"].to_numpy()
    mark(~df["metric"].isin(METRICS), "unknown_metric")
    mark(df["entity_type"] != df["metric"].map(ENTITY_TYPE), "entity_type_mismatch")
    mark(df["snapshot_ts_utc"].isna(), "null_timestamp")
    mark(np.isinf(vals), "non_finite")
    mark(is_t & ((vals <= 0) | (vals > _T_MAX_US)), "out_of_range")
    mark(~is_t & ((vals < 0) | (vals > 1)), "out_of_range")

    ok = reason == ""
    dup = np.zeros(len(df), dtype=bool)
    dup[np.flatnonzero(ok)] = df.loc[ok].duplicated(KEY, keep="last").to_numpy()
    mark(dup, "duplicate")

    bad = df[reason != ""].copy()
    bad["reason"] = reason[reason != ""]
    return df[reason == ""].reset_index(drop=True), bad.reset_index(drop=True)


def soft_flags(clean: pd.DataFrame) -> dict[str, float]:
    """Non-blocking QA statistics worth recording in the data card."""
    t = clean[clean["metric"].isin(["t1", "t2"])]
    wide = t.pivot_table(
        index=["backend", "snapshot_ts_utc", "entity_id"], columns="metric", values="value"
    )
    frac = float((wide["t2"] > 2.05 * wide["t1"]).mean()) if {"t1", "t2"} <= set(wide) else float("nan")
    return {"frac_t2_gt_2t1": frac, "n_rows": float(len(clean))}

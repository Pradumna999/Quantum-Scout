"""Production path: train a bundle, issue nightly forecasts, score them when outcomes arrive."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Config, config_hash
from .evaluation.splits import backend_bounds, window_masks
from .features import build_features, latest_origin_rows, target_col
from .pipeline import HorizonPipeline, fit_horizon
from .store import write_once
from .trust import auroc

FORECAST_COLUMNS = [
    "forecast_id", "issued_at", "origin_ts", "backend", "entity_type", "entity_id", "metric",
    "horizon", "value_at_origin", "point", "lo", "hi", "trust", "tier", "trust_val_auroc",
    "large_thr_log", "model_id", "cfg_hash",
]


@dataclass
class Bundle:
    cfg: Config
    pipelines: dict[int, HorizonPipeline]
    model_id: str
    trained_through: pd.Timestamp

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> Bundle:
        return joblib.load(path)


def train_bundle(cal: pd.DataFrame, cfg: Config) -> Bundle:
    """Fit on all data whose outcomes are known: the earlier part of each backend's timeline
    trains the models, the most recent labelled part calibrates intervals and the trust scorer.
    No data is held out for testing here; evaluate with `run-dev` / `run-final`."""
    feats = build_features(cal, cfg.horizons, cfg)
    cfg2 = cfg.with_(split_fracs=(0.8, 0.1), holdout_backends=())
    bounds = backend_bounds(feats, cfg2)
    pipes: dict[int, HorizonPipeline] = {}
    for h in cfg.horizons:
        m = window_masks(feats, h, bounds, cfg2, "final")
        train = feats[m["train"]]
        # everything after the training window whose outcome is known (val + part of test span)
        has_t = feats[target_col(h)].notna()
        t_train_end = feats["backend"].map({b: v.t_train_end for b, v in bounds.items()})
        calib = feats[has_t & (feats["ts"] >= t_train_end)]
        if len(train) < 200 or len(calib) < 100:
            raise ValueError(f"not enough history for horizon {h}: train={len(train)}, calib={len(calib)}")
        pipes[h] = fit_horizon(train, calib, h, cfg)
    through = pd.Timestamp(cal["snapshot_ts_utc"].max())
    model_id = f"lgbm-cqr-{config_hash(cfg)}-{through:%Y%m%d}"
    return Bundle(cfg, pipes, model_id, through)


def _fid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def forecast_latest(cal: pd.DataFrame, bundle: Bundle, issued_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Forecasts for every series from each backend's most recent calibration cycle."""
    cfg = bundle.cfg
    issued_at = issued_at or pd.Timestamp.now(tz="UTC")
    feats = build_features(cal, cfg.horizons, cfg)
    rows = latest_origin_rows(feats)
    if rows.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    out = []
    for h, pipe in bundle.pipelines.items():
        p = pipe.predict(rows)
        y0 = rows["y0"].to_numpy()
        df = pd.DataFrame(
            {
                "issued_at": issued_at,
                "origin_ts": rows["ts"].to_numpy(),
                "backend": rows["backend"].to_numpy(),
                "entity_type": rows["entity_type"].to_numpy(),
                "entity_id": rows["entity_id"].to_numpy(),
                "metric": rows["metric"].to_numpy(),
                "horizon": h,
                "value_at_origin": np.exp(y0),
                "point": np.exp(y0 + p["p_lgbm"].to_numpy()),
                "lo": np.exp(y0 + p["lo_cqr"].to_numpy()),
                "hi": np.exp(y0 + p["hi_cqr"].to_numpy()),
                "trust": p["trust"].to_numpy(),
                "tier": p["tier"].to_numpy(),
                "trust_val_auroc": float(pipe.trust.val_auroc_),
                "large_thr_log": p["thr_log"].to_numpy(),
                "model_id": bundle.model_id,
                "cfg_hash": config_hash(cfg),
            }
        )
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    res["forecast_id"] = [
        _fid(b, o, e, m, h, bundle.model_id)
        for b, o, e, m, h in zip(res["backend"], res["origin_ts"], res["entity_id"], res["metric"], res["horizon"], strict=True)
    ]
    return res[FORECAST_COLUMNS]


def write_forecast_batches(fc: pd.DataFrame, root: str | Path, on_exists: str = "skip") -> list[Path]:
    """One immutable file per (backend, origin, model). Existing batches are never rewritten."""
    written = []
    for (backend, origin, model_id), g in fc.groupby(["backend", "origin_ts", "model_id"]):
        stamp = pd.Timestamp(origin).tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
        path = Path(root) / "forecasts" / f"backend={backend}" / f"origin={stamp}" / f"{model_id}.parquet"
        p = write_once(g.reset_index(drop=True), path, on_exists=on_exists)
        if p is not None:
            written.append(p)
    return written


def load_forecasts(root: str | Path) -> pd.DataFrame:
    files = sorted((Path(root) / "forecasts").rglob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def score_forecasts(fc: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Attach realized values. The target of a horizon-h forecast is the h-th calibration
    snapshot of that backend after its origin; forecasts whose target has not happened yet
    are left unscored (and omitted from the result)."""
    if fc.empty:
        return pd.DataFrame()
    parts = []
    for backend, g in fc.groupby("backend"):
        snaps = pd.DatetimeIndex(sorted(cal.loc[cal["backend"] == backend, "snapshot_ts_utc"].unique()))
        origin = pd.DatetimeIndex(pd.to_datetime(g["origin_ts"], utc=True))
        pos = snaps.searchsorted(origin, side="left")
        known = (pos < len(snaps)) & (snaps[np.minimum(pos, len(snaps) - 1)] == origin)
        tpos = pos + g["horizon"].to_numpy()
        ok = known & (tpos < len(snaps))
        gg = g[ok].copy()
        gg["target_ts"] = snaps[tpos[ok]]
        parts.append(gg)
    sc = pd.concat(parts, ignore_index=True)
    if sc.empty:
        return sc
    real = cal.rename(columns={"snapshot_ts_utc": "target_ts", "value": "realized"})[
        ["backend", "target_ts", "entity_type", "entity_id", "metric", "realized"]
    ]
    sc["target_ts"] = pd.to_datetime(sc["target_ts"], utc=True)
    real["target_ts"] = pd.to_datetime(real["target_ts"], utc=True)
    sc = sc.merge(real, how="inner", on=["backend", "target_ts", "entity_type", "entity_id", "metric"])
    from .transforms import to_log

    y0, yr, yp = to_log(sc["value_at_origin"]), to_log(sc["realized"]), to_log(sc["point"])
    sc["err_log"] = yr - yp
    sc["err_persist_log"] = yr - y0
    sc["covered"] = (sc["realized"] >= sc["lo"]) & (sc["realized"] <= sc["hi"])
    sc["large_error"] = sc["err_log"].abs() > sc["large_thr_log"]
    return sc


def live_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Forward-test scoreboard per horizon and metric. `n` is rows, not independent events."""
    if scores.empty:
        return pd.DataFrame()
    rows = []
    for (h, metric), g in scores.groupby(["horizon", "metric"]):
        mse, mse_p = float((g["err_log"] ** 2).mean()), float((g["err_persist_log"] ** 2).mean())
        rows.append({
            "horizon": h, "metric": metric, "n": len(g), "n_origins": g["origin_ts"].nunique(),
            "coverage": float(g["covered"].mean()),
            "skill_mse": 1 - mse / mse_p if mse_p > 0 else float("nan"),
            "auroc_trust": auroc(g["large_error"].to_numpy(), g["trust"].to_numpy()),
        })
    return pd.DataFrame(rows)


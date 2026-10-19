"""Experiments E1-E8. Each table is written as a CSV; nothing is hand-edited."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, config_hash
from .conformal import aci_cqr
from .downstream import run_e7
from .drift import jump_flags, page_hinkley_frame, robust_diffs, volatility_z
from .evaluation.metrics import coverage, interval_score, point_summary
from .evaluation.splits import backend_bounds, rolling_origin_masks, window_masks
from .evaluation.stats import block_bootstrap_ci, diebold_mariano, holm
from .features import build_features, target_col, target_ts_col
from .models import LGBMQuantile
from .pipeline import fit_horizon
from .schema import SERIES_KEYS
from .synthetic import make_detection_panel
from .trust import (
    TRUST_FEATURES,
    TrustScorer,
    aurc,
    auroc,
    expected_calibration_error,
)

MODELS = ("persist", "ewma", "rollmed", "lgbm")
METHODS = ("pconf_abs", "pconf", "lconf", "cqr", "cqr_aci")
_PASS = ["vol_ratio", "absd_mean10", "tsl", "ph_stat", "xs_jump_frac"]


def ts_ns(s: pd.Series) -> np.ndarray:
    """Timestamps as int64 nanoseconds, independent of the stored datetime resolution."""
    return (
        pd.to_datetime(s, utc=True).dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
    )


# -- E1 -----------------------------------------------------------------------------------------


def characterize(cal: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of calibration dynamics per backend x metric (log space)."""
    rows = []
    n_cycles = cal.groupby("backend")["snapshot_ts_utc"].nunique()
    for (backend, metric), g in feats.groupby(["backend", "metric"]):
        n_series = g["entity_id"].nunique()
        obs = ((cal["backend"] == backend) & (cal["metric"] == metric)).sum()
        d1 = g["d1"].dropna()
        from scipy.stats import kurtosis

        rows.append(
            {
                "backend": backend,
                "metric": metric,
                "n_series": n_series,
                "n_cycles": int(n_cycles[backend]),
                "missing_frac": float(1 - obs / (n_series * n_cycles[backend])),
                "median_abs_d1": float(d1.abs().median()),
                "p99_abs_d1": float(d1.abs().quantile(0.99)),
                "excess_kurtosis_d1": float(kurtosis(d1)),
                "acf1_d1": float(g["d1"].corr(g["d2"])),
                "jump_rate": float((g["tsl"] == 0).mean()),
                "device_event_rate": float(
                    (g.groupby("ts")["xs_jump_frac"].first() > 0.3).mean()
                ),
                "mean_vol_ratio": float(g["vol_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


# -- tables from the pooled evaluation frame -------------------------------------------------------


def point_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, metric, grp), g in preds.groupby(["horizon", "metric", "group"]):
        y = g["y"].to_numpy()
        for m in MODELS:
            rows.append(
                {"horizon": h, "metric": metric, "group": grp, "model": m,
                 **point_summary(y - g[f"p_{m}"].to_numpy(), y)}
            )
    return pd.DataFrame(rows)


def dm_table(preds: pd.DataFrame) -> pd.DataFrame:
    """DM tests of each model against persistence, per backend/metric/horizon, on per-cycle mean
    squared-error differentials (positive mean_d: the model beats persistence)."""
    rows = []
    for (h, backend, metric, grp), g in preds.groupby(["horizon", "backend", "metric", "group"]):
        y = g["y"].to_numpy()
        base = pd.Series(y**2, index=g.index)
        for m in MODELS[1:]:
            loss = pd.Series((y - g[f"p_{m}"].to_numpy()) ** 2, index=g.index)
            d = (base.groupby(g["ts"]).mean() - loss.groupby(g["ts"]).mean()).sort_index()
            stat, p = diebold_mariano(d.to_numpy(), h)
            rows.append({"horizon": h, "backend": backend, "metric": metric, "group": grp,
                         "model": m, "n_cycles": len(d), "mean_d": float(d.mean()),
                         "dm_stat": stat, "p_value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_value"].to_numpy())
    return out


def interval_table(preds: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, reg = [], []
    for (h, metric, grp), g in preds.groupby(["horizon", "metric", "group"]):
        y = g["y"].to_numpy()
        for m in METHODS:
            lo, hi = g[f"lo_{m}"].to_numpy(), g[f"hi_{m}"].to_numpy()
            rows.append({"horizon": h, "metric": metric, "group": grp, "method": m,
                         "n": len(g), "nominal": 1 - alpha, "coverage": coverage(y, lo, hi),
                         "mean_width": float(np.mean(hi - lo)),
                         "interval_score": float(np.mean(interval_score(y, lo, hi, alpha)))})
    for (h, grp, regime), g in preds.groupby(["horizon", "group", "regime"]):
        y = g["y"].to_numpy()
        for m in METHODS:
            reg.append({"horizon": h, "group": grp, "regime": regime, "method": m, "n": len(g),
                        "nominal": 1 - alpha,
                        "coverage": coverage(y, g[f"lo_{m}"].to_numpy(), g[f"hi_{m}"].to_numpy())})
    return pd.DataFrame(rows), pd.DataFrame(reg)


def trust_tables(
    preds: pd.DataFrame, cfg: Config, per_h_trust: dict[int, float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_h_trust = per_h_trust or {}
    rows, tiers = [], []
    for (h, grp), g in preds.groupby(["horizon", "group"]):
        g = g.reset_index(drop=True)
        lab = g["label"].to_numpy()
        err = np.abs(g["y"].to_numpy() - g["p_lgbm"].to_numpy())

        def delta(d: pd.DataFrame) -> float:
            return auroc(d["label"].to_numpy(), d["trust"].to_numpy()) - auroc(
                d["label"].to_numpy(), d["width_cqr"].to_numpy()
            )

        d_point, d_lo, d_hi = block_bootstrap_ci(g, "ts", delta, cfg.block_len, cfg.n_boot, cfg.seed)
        rows.append(
            {
                "horizon": h, "group": grp, "n": len(g), "large_error_rate": float(lab.mean()),
                "calib_val_auroc": per_h_trust.get(h, float("nan")),
                "auroc_trust": auroc(lab, g["trust"]), "auroc_width_cqr": auroc(lab, g["width_cqr"]),
                "auroc_scale": auroc(lab, g["absd_mean10"]), "auroc_vol_ratio": auroc(lab, g["vol_ratio"].fillna(1)),
                "delta_auroc_vs_width": d_point, "delta_ci_lo": d_lo, "delta_ci_hi": d_hi,
                "aurc_trust": aurc(g["trust"].to_numpy(), err),
                "aurc_width_cqr": aurc(g["width_cqr"].to_numpy(), err),
                "mean_abs_err": float(err.mean()),
                "brier": float(np.mean((g["trust"].to_numpy() - lab) ** 2)),
                "ece": expected_calibration_error(g["trust"].to_numpy(), lab),
            }
        )
        for tier, t in g.groupby("tier"):
            tiers.append({"horizon": h, "group": grp, "tier": tier, "share": len(t) / len(g),
                          "large_error_rate": float(t["label"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(tiers)


def ablation_table(per_h: dict, cfg: Config) -> pd.DataFrame:
    """Refit the trust scorer without one feature at a time (calibration -> evaluation)."""
    rows = []
    hs = sorted(per_h)
    for h in sorted({hs[0], hs[-1]}):
        pipe, calib, ev, pc, pe = per_h[h]
        tf_c = pipe.trust_frame(calib, pc["width_cqr"].to_numpy(), pc["disagreement"].to_numpy())
        lab_c = np.abs(calib[target_col(h)].to_numpy() - pc["p_lgbm"].to_numpy()) > pipe.large_error_threshold(calib)
        tf_e = pipe.trust_frame(ev, pe["width_cqr"].to_numpy(), pe["disagreement"].to_numpy())
        lab_e = np.abs(ev[target_col(h)].to_numpy() - pe["p_lgbm"].to_numpy()) > pipe.large_error_threshold(ev)
        is_h = ev["backend"].isin(cfg.holdout_backends).to_numpy()
        variants = {"full": TRUST_FEATURES, "width_only": ["width_ratio"]}
        variants.update({f"minus_{f}": [x for x in TRUST_FEATURES if x != f] for f in TRUST_FEATURES})
        for name, feats in variants.items():
            sc = TrustScorer(cfg.trust_fprs, feats, cfg.seed, cfg.scale_floor).fit(tf_c, lab_c, calib["ts"].reset_index(drop=True))
            s = sc.predict_proba(tf_e)
            for grp, mask in (("in_dist", ~is_h), ("holdout", is_h)):
                if mask.any():
                    rows.append({"horizon": h, "group": grp, "variant": name, "auroc": auroc(lab_e[mask], s[mask])})
    out = pd.DataFrame(rows)
    if not out.empty:
        full = out[out["variant"] == "full"].set_index(["horizon", "group"])["auroc"]
        out["delta_vs_full"] = out["auroc"].to_numpy() - out.set_index(["horizon", "group"]).index.map(full).to_numpy()
    return out


def rolling_table(feats: pd.DataFrame, cfg: Config, bounds: dict, n_folds: int = 3) -> pd.DataFrame:
    rows = []
    for h in cfg.horizons:
        for f, m in enumerate(rolling_origin_masks(feats, h, bounds, cfg, n_folds), start=1):
            tr, ev = feats[m["train"]], feats[m["eval"]]
            if len(tr) < 200 or ev.empty:
                continue
            model = LGBMQuantile(cfg, quantiles=(0.5,)).fit(tr, tr[target_col(h)].to_numpy())
            y = ev[target_col(h)].to_numpy()
            err = y - model.predict(ev)
            for metric, idx in ev.groupby("metric").indices.items():
                rows.append({"horizon": h, "fold": f, "metric": metric,
                             **point_summary(err[idx], y[idx])})
    return pd.DataFrame(rows)


def transfer_table(point: pd.DataFrame, intervals: pd.DataFrame, trust: pd.DataFrame) -> pd.DataFrame:
    """E6 summary: in-distribution vs held-out backends, averaged over metrics."""
    if point.empty:
        return pd.DataFrame()
    a = point[point["model"] == "lgbm"].groupby(["horizon", "group"])["skill_mse"].mean().rename("lgbm_skill_mse")
    c = intervals[intervals["method"].isin(["cqr", "cqr_aci"])].pivot_table(
        index=["horizon", "group"], columns="method", values="coverage", aggfunc="mean"
    ).add_prefix("coverage_")
    t = trust.set_index(["horizon", "group"])[["auroc_trust", "auroc_width_cqr"]]
    return pd.concat([a, c, t], axis=1).reset_index()


# -- orchestration ---------------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def run_experiments(
    cal: pd.DataFrame,
    cfg: Config,
    mode: str = "dev",
    out_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    n_folds: int = 3,
    run_rolling: bool = True,
) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    feats = build_features(cal, cfg.horizons, cfg)
    bounds = backend_bounds(feats, cfg)
    log(f"features: {len(feats):,} rows, {feats['backend'].nunique()} backends, mode={mode}")

    frames, per_h = [], {}
    for h in cfg.horizons:
        m = window_masks(feats, h, bounds, cfg, mode)
        train, calib, ev = feats[m["train"]], feats[m["calib"]], feats[m["eval"]]
        if len(train) < 200 or len(calib) < 100 or ev.empty:
            log(f"h={h}: skipped (train={len(train)}, calib={len(calib)}, eval={len(ev)})")
            continue
        pipe = fit_horizon(train, calib, h, cfg)
        pc, pe = pipe.predict(calib), pipe.predict(ev)
        y = ev[target_col(h)].to_numpy(dtype=float)
        stream = (ev["backend"] + "|" + ev["metric"]).to_numpy()
        lo, hi, a_used = aci_cqr(
            pipe.cqr, pe["q_lo"].to_numpy(), pe["q_hi"].to_numpy(), y, ts_ns(ev["ts"]),
            ts_ns(ev[target_ts_col(h)]), stream, ev["metric"].to_numpy(), cfg.alpha, cfg.aci_gamma,
        )
        pe["lo_cqr_aci"], pe["hi_cqr_aci"], pe["alpha_t"] = lo, hi, a_used
        cuts = np.nanquantile(calib["vol_ratio"].to_numpy(dtype=float), [1 / 3, 2 / 3])
        vr = np.nan_to_num(ev["vol_ratio"].to_numpy(dtype=float), nan=1.0)
        frame = ev[SERIES_KEYS + ["ts", "cycle", "y0"] + _PASS].reset_index(drop=True).copy()
        frame["horizon"] = h
        frame["y"] = y
        frame["group"] = np.where(ev["backend"].isin(cfg.holdout_backends), "holdout", "in_dist")
        frame["regime"] = np.array(["low", "mid", "high"])[np.digitize(vr, cuts)]
        frame = pd.concat([frame, pe.reset_index(drop=True)], axis=1)
        frame["label"] = np.abs(y - frame["p_lgbm"].to_numpy()) > frame["thr_log"].to_numpy()
        frames.append(frame)
        per_h[h] = (pipe, calib, ev, pc, pe)
        log(f"h={h}: train={len(train):,} calib={len(calib):,} eval={len(ev):,}")
    if not frames:
        raise RuntimeError("no horizon had enough data; use a longer history or smaller horizons")
    preds = pd.concat(frames, ignore_index=True)

    t: dict[str, pd.DataFrame] = {}
    t["e1_characterization"] = characterize(cal, feats)
    t["e2_point"] = point_table(preds)
    t["e2_dm"] = dm_table(preds)
    if run_rolling:
        t["e2_rolling"] = rolling_table(feats, cfg, bounds, n_folds)
    t["e3_intervals"], t["e3_regime"] = interval_table(preds, cfg.alpha)
    val_auc = {h: float(v[0].trust.val_auroc_) for h, v in per_h.items()}
    t["e4_trust"], t["e4_tiers"] = trust_tables(preds, cfg, val_auc)
    t["e5_ablation"] = ablation_table(per_h, cfg)
    t["e6_transfer"] = transfer_table(t["e2_point"], t["e3_intervals"], t["e4_trust"])
    h1 = min(per_h)
    t["e7_per_origin"], t["e7_summary"] = run_e7(preds[preds["horizon"] == h1], cfg)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in t.items():
            df.to_csv(out / f"{name}.csv", index=False)
        preds.to_parquet(out / "preds.parquet", index=False)
        import lightgbm
        import sklearn

        meta = {
            "mode": mode, "config_hash": config_hash(cfg), "config": asdict(cfg),
            "n_feature_rows": int(len(feats)), "n_eval_rows": int(len(preds)),
            "backends": sorted(feats["backend"].unique().tolist()),
            "holdout_backends": list(cfg.holdout_backends), "git_sha": _git_sha(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
                         "sklearn": sklearn.__version__},
            "seconds": round(time.time() - t0, 1),
        }
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        log(f"wrote {len(t)} tables to {out}")
    t["_preds"] = preds
    return t


# -- E8: detectors against known ground truth -------------------------------------------------------


def run_e8(cfg: Config, out_dir: str | Path | None = None, seeds: int = 5) -> pd.DataFrame:
    """Jump flags vs known level jumps; Page-Hinkley vs known volatility regimes."""
    rows = []
    for seed in range(seeds):
        panel = make_detection_panel(seed=seed)
        T, N = panel.logy.shape
        logy = pd.DataFrame(panel.logy)
        _, absd, scale = robust_diffs(logy, cfg.jump_window, cfg.scale_floor)
        flags = jump_flags(absd, scale, cfg.jump_k).to_numpy()
        truth = np.zeros((T, N), dtype=bool)
        for t, n in panel.jumps:
            truth[t, n] = True
        tp = int((flags & truth).sum())
        regime_cover = np.zeros((T, N), dtype=bool)
        for s, e, n in panel.regimes:
            regime_cover[s:e, n] = True
        fp_clean = int((flags & ~truth & ~regime_cover).sum())
        clean_steps = int((~truth & ~regime_cover)[40:].sum())
        rows.append({"seed": seed, "detector": "jump_flag", "recall": tp / max(1, truth.sum()),
                     "false_alarms_per_100_steps": 100 * fp_clean / max(1, clean_steps),
                     "mean_delay": 0.0, "n_events": int(truth.sum())})

        z = volatility_z(absd, cfg.scale_floor)
        _, alarm = page_hinkley_frame(z, cfg.ph_delta, cfg.ph_lambda, cfg.ph_min_n)
        alarm = alarm.to_numpy()
        delays, hits = [], 0
        for s, e, n in panel.regimes:
            hit = np.flatnonzero(alarm[s:e, n])
            if hit.size:
                hits += 1
                delays.append(int(hit[0]))
        regime_series = {n for _, _, n in panel.regimes}
        calm = [n for n in range(N) if n not in regime_series]
        calm_steps = len(calm) * (T - 40)
        calm_alarms = int(alarm[40:, calm].sum()) if calm else 0
        rows.append({"seed": seed, "detector": "page_hinkley_vol", "recall": hits / max(1, len(panel.regimes)),
                     "false_alarms_per_100_steps": 100 * calm_alarms / max(1, calm_steps),
                     "mean_delay": float(np.mean(delays)) if delays else float("nan"),
                     "n_events": len(panel.regimes)})
    df = pd.DataFrame(rows)
    summary = df.groupby("detector").mean(numeric_only=True).drop(columns="seed").reset_index()
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "e8_runs.csv", index=False)
        summary.to_csv(out / "e8_summary.csv", index=False)
    return summary


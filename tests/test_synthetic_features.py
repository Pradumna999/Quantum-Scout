import numpy as np
import pandas as pd

from qscout.features import FEATURE_COLS, build_features, target_col, target_ts_col
from qscout.schema import validate_calibration


def test_synthetic_passes_contract_and_is_physical(synth):
    cal, ev = synth
    clean, bad = validate_calibration(cal)
    assert len(bad) == 0 and len(clean) == len(cal)
    w = cal[cal.metric.isin(["t1", "t2"])].pivot_table(
        index=["backend", "snapshot_ts_utc", "entity_id"], columns="metric", values="value")
    assert (w["t2"] <= 2 * w["t1"] + 1e-9).all()
    assert not ev.empty


def test_features_equal_on_truncated_panel(synth, tiny_cfg, feats):
    """Leakage test 1: features at origin k computed from data <= k equal the full-panel values."""
    cal, _ = synth
    k = sorted(feats["ts"].unique())[80]
    trunc = build_features(cal, tiny_cfg.horizons, tiny_cfg, as_of=k)
    full = feats[feats["ts"] <= k]
    keys = ["backend", "metric", "entity_id", "ts"]
    a = full.sort_values(keys).reset_index(drop=True)
    b = trunc.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b)
    np.testing.assert_allclose(a[FEATURE_COLS].to_numpy(), b[FEATURE_COLS].to_numpy(), equal_nan=True, atol=1e-12)


def test_features_ignore_future_perturbation(synth, tiny_cfg, feats):
    """Leakage test 2: wrecking every observation after k must not change features at <= k."""
    cal, _ = synth
    k = sorted(feats["ts"].unique())[90]
    rng = np.random.default_rng(0)
    bad = cal.copy()
    fut = bad["snapshot_ts_utc"] > k
    bad.loc[fut, "value"] = bad.loc[fut, "value"] * rng.uniform(0.2, 5.0, fut.sum())
    pert = build_features(bad, tiny_cfg.horizons, tiny_cfg)
    keys = ["backend", "metric", "entity_id", "ts"]
    a = feats[feats["ts"] <= k].sort_values(keys).reset_index(drop=True)
    b = pert[pert["ts"] <= k].sort_values(keys).reset_index(drop=True)
    np.testing.assert_allclose(a[FEATURE_COLS].to_numpy(), b[FEATURE_COLS].to_numpy(), equal_nan=True, atol=1e-12)


def test_targets_are_log_changes_to_the_h_th_next_cycle(synth, feats):
    cal, _ = synth
    g = feats[(feats.backend == "synth_a") & (feats.metric == "t1") & (feats.entity_id == "0")].sort_values("ts")
    s = cal[(cal.backend == "synth_a") & (cal.metric == "t1") & (cal.entity_id == "0")].sort_values("snapshot_ts_utc")
    logv = np.log(s["value"].to_numpy())
    ts_all = sorted(cal[cal.backend == "synth_a"]["snapshot_ts_utc"].unique())
    pos = {t: i for i, t in enumerate(ts_all)}
    row = g.iloc[10]
    if pd.notna(row[target_col(1)]):
        t = row["ts"]
        nxt = ts_all[pos[t] + 1]
        s_idx = s.set_index("snapshot_ts_utc")["value"]
        if nxt in s_idx.index:
            assert row[target_col(1)] == np.log(s_idx[nxt]) - np.log(s_idx[t])
        assert row[target_ts_col(1)] == nxt
    assert g[target_col(2)].iloc[-1] != g[target_col(2)].iloc[-1]  # last origin has no target (NaN)
    assert len(logv) > 0


def test_min_history_respected(feats, tiny_cfg):
    assert feats["cycle"].min() >= tiny_cfg.min_history

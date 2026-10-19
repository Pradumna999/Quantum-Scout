import numpy as np
import pandas as pd
import pytest

from qscout.evaluation.splits import backend_bounds, window_masks
from qscout.experiments import run_experiments
from qscout.features import target_col
from qscout.pipeline import fit_horizon


@pytest.fixture(scope="module")
def fitted(feats, tiny_cfg):
    b = backend_bounds(feats, tiny_cfg)
    m = window_masks(feats, 1, b, tiny_cfg, "dev")
    pipe = fit_horizon(feats[m["train"]], feats[m["calib"]], 1, tiny_cfg)
    return pipe, feats[m["eval"]]


def test_predict_shapes_and_invariants(fitted):
    pipe, ev = fitted
    p = pipe.predict(ev)
    assert len(p) == len(ev) and p.index.equals(ev.index)
    for c in p.columns:
        if c != "tier":
            assert np.isfinite(p[c]).all(), c
    assert (p["q_lo"] <= p["p_lgbm"] + 1e-12).all() and (p["p_lgbm"] <= p["q_hi"] + 1e-12).all()
    for m in ("cqr", "pconf", "pconf_abs", "lconf"):
        assert (p[f"lo_{m}"] <= p[f"hi_{m}"]).all()
    assert p["trust"].between(0, 1).all() and set(p["tier"]) <= {"ok", "caution", "unreliable"}
    assert (p["p_persist"] == 0).all()


def test_marginal_coverage_is_near_nominal(fitted):
    pipe, ev = fitted
    p = pipe.predict(ev)
    y = ev[target_col(1)].to_numpy()
    for m in ("cqr", "pconf", "pconf_abs", "lconf"):
        cov = np.mean((y >= p[f"lo_{m}"]) & (y <= p[f"hi_{m}"]))
        assert 0.8 < cov < 0.97, (m, cov)


def test_prediction_is_row_order_invariant(fitted):
    pipe, ev = fitted
    sub = ev.sample(200, random_state=0)
    a = pipe.predict(sub)
    b = pipe.predict(sub.iloc[::-1]).iloc[::-1]
    np.testing.assert_allclose(a["trust"].to_numpy(), b["trust"].to_numpy())
    np.testing.assert_allclose(a["p_lgbm"].to_numpy(), b["p_lgbm"].to_numpy())


def test_experiments_end_to_end(synth, tiny_cfg, tmp_path):
    res = run_experiments(synth[0], tiny_cfg, "dev", tmp_path, log=lambda s: None, n_folds=2)
    for name in ("e1_characterization", "e2_point", "e2_dm", "e2_rolling", "e3_intervals", "e3_regime",
                 "e4_trust", "e4_tiers", "e5_ablation", "e6_transfer", "e7_summary"):
        assert name in res and (tmp_path / f"{name}.csv").exists()
    pt = res["e2_point"]
    assert np.allclose(pt[pt.model == "persist"]["skill_mse"], 0.0)
    iv = res["e3_intervals"]
    assert iv["coverage"].between(0, 1).all() and (iv["mean_width"] > 0).all()
    assert set(res["e3_intervals"]["group"]) == {"in_dist", "holdout"}
    tr = res["e4_trust"]
    assert tr["auroc_trust"].dropna().between(0, 1).all()
    assert (tmp_path / "run_meta.json").exists() and (tmp_path / "preds.parquet").exists()
    meta = pd.read_json(tmp_path / "run_meta.json", typ="series")
    assert meta["mode"] == "dev"


def test_dev_run_never_uses_test_period(synth, tiny_cfg):
    cal = synth[0]
    res = run_experiments(cal, tiny_cfg, "dev", None, log=lambda s: None, run_rolling=False)
    preds = res["_preds"]
    from qscout.features import build_features
    feats = build_features(cal, tiny_cfg.horizons, tiny_cfg)
    b = backend_bounds(feats, tiny_cfg)
    for backend, g in preds.groupby("backend"):
        assert (g["ts"] < b[backend].t_val_end).all()

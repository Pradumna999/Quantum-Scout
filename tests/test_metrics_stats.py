import numpy as np
import pandas as pd
import pytest

from qscout.evaluation.metrics import coverage, interval_score, pinball, point_summary
from qscout.evaluation.stats import block_bootstrap_ci, diebold_mariano, holm
from qscout.trust import TrustScorer, aurc, auroc, expected_calibration_error, risk_coverage


def test_interval_score_hand_computed():
    # width 2; y below by 1 -> 2 + (2/0.1)*1 = 22 ; inside -> 2 ; above by 0.5 -> 2 + 20*0.5 = 12
    s = interval_score([-1.0, 0.0, 1.5], [0.0, -1.0, -1.0], [2.0 - 2.0 + 0.0, 1.0, 1.0], 0.1)
    # row1: lo=0, hi=0 width 0 -> 0 + 20*1 = 20 ; row2: width 2, inside ; row3: width 2 + 20*0.5
    assert s.tolist() == pytest.approx([20.0, 2.0, 12.0])
    assert coverage([0, 5], [-1, -1], [1, 1]) == 0.5


def test_pinball_hand_computed():
    assert pinball([2.0], [1.0], 0.9)[0] == pytest.approx(0.9)
    assert pinball([0.0], [1.0], 0.9)[0] == pytest.approx(0.1)


def test_point_summary_skill_is_zero_for_persistence():
    y = np.array([0.1, -0.2, 0.3])
    s = point_summary(y, y)
    assert s["skill_mse"] == 0 and s["rel_mae"] == 1


def test_dm_known_cases():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1, 400)
    stat, p = diebold_mariano(d, 1)
    assert stat > 5 and p < 1e-4
    stat0, p0 = diebold_mariano(rng.normal(0, 1, 400), 1)
    assert p0 > 0.01
    # h=1 reduces to the t-statistic times the HLN factor sqrt((T-1)/T)
    T = 400
    expect = d.mean() / (d.std(ddof=0) / np.sqrt(T)) * np.sqrt((T - 1) / T)
    assert stat == pytest.approx(expect, rel=1e-9)
    assert np.isnan(diebold_mariano(np.ones(3), 1)[0])


def test_dm_hac_widens_variance_for_autocorrelated_differentials():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 2000)
    d = 0.05 + np.convolve(e, np.ones(5) / 5, mode="same")  # strongly autocorrelated
    s1, _ = diebold_mariano(d, 1)
    s5, _ = diebold_mariano(d, 5)
    assert abs(s5) < abs(s1)


def test_holm_matches_manual_and_ignores_nan():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    adj = holm(p)
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])


def test_block_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(0)
    T = 200
    df = pd.DataFrame({"t": np.repeat(np.arange(T), 5), "x": rng.normal(1.0, 1, T * 5)})
    pt, lo, hi = block_bootstrap_ci(df, "t", lambda d: d["x"].mean(), 5, 200, seed=1)
    assert lo < 1.0 < hi and lo < pt < hi


def test_auroc_handles_nan_scores_and_single_class():
    assert auroc([0, 1, 1, 0], [0.1, 0.9, np.nan, 0.2]) == 1.0
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_risk_coverage_and_ece():
    score = np.array([0.1, 0.2, 0.9, 0.8])
    err = np.array([0.0, 0.0, 1.0, 1.0])
    cov, risk = risk_coverage(score, err)
    assert risk[1] == 0 and risk[-1] == 0.5
    assert aurc(score, err) < aurc(-score, err)
    assert expected_calibration_error(np.array([0.5] * 100), np.array([1, 0] * 50)) == pytest.approx(0.0)


def test_trust_scorer_learns_a_planted_signal_and_falls_back_without_positives():
    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame({
        "backend": "b", "metric": "t1", "entity_type": "qubit",
        "entity_id": rng.integers(0, 20, n).astype(str),
        "ts": pd.to_datetime("2026-01-01", utc=True) + pd.to_timedelta(np.arange(n) // 20, unit="D"),
        "width": 1.0, "absd_mean10": 0.05, "vol_ratio": rng.uniform(0.5, 3.0, n), "tsl": 10.0,
        "ph_stat": 0.0, "xs_jump_frac": 0.0, "disagreement": 0.1,
    })
    p = 1 / (1 + np.exp(-3 * (df["vol_ratio"] - 2.4)))
    y = rng.random(n) < p
    sc = TrustScorer(min_positives=10).fit(df, y, df["ts"])
    assert sc.val_auroc_ > 0.75
    tiers = sc.tier(sc.predict_proba(df))
    assert y[tiers == "unreliable"].mean() > y[tiers == "ok"].mean()
    sc0 = TrustScorer().fit(df, np.zeros(n, dtype=bool), df["ts"])
    assert np.allclose(sc0.predict_proba(df), 0.0)

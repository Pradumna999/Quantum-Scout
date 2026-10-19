import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from qscout.conformal import (
    GroupedCQR,
    GroupedSplitConformal,
    aci_cqr,
    conformal_quantile,
    level_quantile,
)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(0, 100, allow_nan=False), min_size=20, max_size=300), st.sampled_from([0.05, 0.1, 0.2]))
def test_quantile_covers_calibration_scores(scores, alpha):
    q = conformal_quantile(scores, alpha)
    assert np.mean(np.asarray(scores) <= q) >= 1 - alpha - 1e-12


def test_quantile_edge_cases():
    assert conformal_quantile([], 0.1) == float("inf")
    assert conformal_quantile([1.0, 2.0, 3.0], 0.1) == 3.0  # ceil(4*0.9)/3 > 1 -> max
    s = np.arange(1.0, 101.0)
    assert level_quantile(s, 0.0) == 100.0 and level_quantile(s, 1.5) == 1.0


def test_split_conformal_and_cqr_marginal_coverage_on_iid_data():
    rng = np.random.default_rng(0)
    n = 6000
    g = rng.choice(["a", "b"], n)
    scale = np.where(g == "a", 1.0, 20.0)
    y = rng.normal(0, 1, n) * scale
    cal, te = slice(0, 3000), slice(3000, n)
    sc = GroupedSplitConformal().fit(np.zeros(3000), y[cal], g[cal], 0.1)
    lo, hi = sc.interval(np.zeros(3000), g[te])
    for grp in ("a", "b"):
        m = g[te] == grp
        assert abs(np.mean((y[te][m] >= lo[m]) & (y[te][m] <= hi[m])) - 0.9) < 0.03
    qlo, qhi = -1.2 * scale, 1.2 * scale  # deliberately miscalibrated raw quantiles
    cqr = GroupedCQR().fit(qlo[cal], qhi[cal], y[cal], g[cal], 0.1)
    lo, hi = cqr.interval(qlo[te], qhi[te], g[te])
    assert abs(np.mean((y[te] >= lo) & (y[te] <= hi)) - 0.9) < 0.03


def _stream(T, rows, sd_fn, rng):
    ts = np.repeat(np.arange(T), rows)
    y = rng.normal(0, 1, T * rows) * np.repeat([sd_fn(t) for t in range(T)], rows)
    return ts.astype("int64"), y


def test_aci_restores_long_run_coverage_under_variance_shift_and_split_does_not():
    rng = np.random.default_rng(1)
    rows, T = 40, 600
    cal_y = rng.normal(0, 1, 4000)
    cqr = GroupedCQR().fit(-1.64 * np.ones(4000), 1.64 * np.ones(4000), cal_y, ["m"] * 4000, 0.1)
    ts, y = _stream(T, rows, lambda t: 1.0 if t < 100 else 3.0, rng)
    lo = np.full(len(y), -1.64)
    hi = np.full(len(y), 1.64)
    tgt = ts + 1
    lo_a, hi_a, _ = aci_cqr(cqr, lo, hi, y, ts, tgt, np.array(["s"] * len(y)), np.array(["m"] * len(y)), 0.1, 0.05)
    late = ts >= 300
    cov_static = np.mean((y[late] >= lo[late]) & (y[late] <= hi[late]))
    cov_aci = np.mean((y[late] >= lo_a[late]) & (y[late] <= hi_a[late]))
    assert cov_static < 0.6
    assert abs(cov_aci - 0.9) < 0.05


def test_aci_uses_only_feedback_that_has_arrived():
    """With a huge delay no outcome is ever fed back, so alpha_t must stay at nominal."""
    rng = np.random.default_rng(2)
    cqr = GroupedCQR().fit(-np.ones(500), np.ones(500), rng.normal(0, 1, 500), ["m"] * 500, 0.1)
    ts, y = _stream(100, 10, lambda t: 5.0, rng)
    _, _, a = aci_cqr(cqr, np.full(len(y), -1.0), np.full(len(y), 1.0), y, ts, ts + 10**6,
                      np.array(["s"] * len(y)), np.array(["m"] * len(y)), 0.1, 0.05)
    assert np.allclose(a, 0.1)
    # and with delay 1, alpha at origin t reflects only origins <= t-1
    _, _, a1 = aci_cqr(cqr, np.full(len(y), -1.0), np.full(len(y), 1.0), y, ts, ts + 1,
                       np.array(["s"] * len(y)), np.array(["m"] * len(y)), 0.1, 0.05)
    assert np.allclose(a1[ts == 0], 0.1) and not np.allclose(a1[ts == 5], 0.1)
    assert pd.Series(a1).notna().all()

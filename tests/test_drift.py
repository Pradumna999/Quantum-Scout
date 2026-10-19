import numpy as np
import pandas as pd
import pytest

from qscout.config import load_config
from qscout.drift import jump_flags, page_hinkley, pelt_changepoints, robust_diffs, volatility_z
from qscout.experiments import run_e8


def test_page_hinkley_detects_mean_shift_and_resets():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(1, 0.3, 200), rng.normal(2.5, 0.3, 100)])
    stat, alarm = page_hinkley(x, delta=0.25, lam=8, min_n=15)
    first = np.flatnonzero(alarm)[0]
    assert 200 <= first < 230
    assert stat[first] > 8
    assert stat[first + 1] < stat[first]  # state reset after an alarm


def test_page_hinkley_quiet_stream_rarely_alarms_and_skips_nan():
    rng = np.random.default_rng(1)
    x = rng.normal(1, 0.3, 2000)
    x[::7] = np.nan
    _, alarm = page_hinkley(x, 0.25, 8, 15)
    assert alarm.sum() <= 1


def test_jump_flag_catches_big_step_not_noise():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.03, 120).cumsum() * 0.1 + rng.normal(0, 0.03, 120)
    y[80:] += 1.0
    _, absd, scale = robust_diffs(pd.DataFrame({"a": y}), 30, 0.005)
    flags = jump_flags(absd, scale, 4.0)["a"].to_numpy()
    assert flags[80] and flags.sum() <= 3


def test_volatility_z_is_about_one_in_stationary_regime():
    rng = np.random.default_rng(3)
    absd = pd.DataFrame({"a": np.abs(rng.normal(0, 0.05, 1000))})
    z = volatility_z(absd, 0.005)["a"].dropna()
    assert 0.8 < z.mean() < 1.2


def test_e8_detectors_have_signal():
    s = run_e8(load_config(), seeds=2).set_index("detector")
    assert s.loc["page_hinkley_vol", "recall"] > 0.5
    assert s.loc["page_hinkley_vol", "false_alarms_per_100_steps"] < 0.5
    assert s.loc["jump_flag", "false_alarms_per_100_steps"] < 1.0


def test_pelt_finds_a_level_shift():
    pytest.importorskip("ruptures")
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0, 0.05, 100), rng.normal(1, 0.05, 100)])
    cps = pelt_changepoints(x)
    assert any(abs(c - 100) <= 3 for c in cps)

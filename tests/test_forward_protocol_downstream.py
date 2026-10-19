import itertools

import numpy as np
import pandas as pd
import pytest

from qscout.config import load_config
from qscout.downstream import best_chain
from qscout.protocol import ProtocolError, acquire_final_lock, check_protocol_frozen
from qscout.serve import (
    Bundle,
    forecast_latest,
    live_summary,
    load_forecasts,
    score_forecasts,
    train_bundle,
    write_forecast_batches,
)
from qscout.store import AppendOnlyViolation


@pytest.fixture(scope="module")
def loop(synth):
    cfg = load_config(horizons=(1, 2), lgbm={"n_estimators": 20, "min_child_samples": 20}, min_group=20)
    cal = synth[0]
    cal = cal[cal.backend.isin(["synth_a", "synth_b"])]
    ts = sorted(cal.snapshot_ts_utc.unique())
    past = cal[cal.snapshot_ts_utc <= ts[-8]]
    return cfg, cal, past, train_bundle(past, cfg)


def test_forecast_log_is_append_only_and_scorable(loop, tmp_path):
    cfg, cal, past, bundle = loop
    fc = forecast_latest(past, bundle, issued_at=pd.Timestamp("2026-01-01", tz="UTC"))
    assert fc["forecast_id"].is_unique and (fc["lo"] <= fc["hi"]).all()
    assert (fc["origin_ts"] == fc.groupby("backend")["origin_ts"].transform("max")).all()
    assert len(write_forecast_batches(fc, tmp_path)) == 2
    assert write_forecast_batches(fc, tmp_path) == []  # idempotent
    with pytest.raises(AppendOnlyViolation):
        write_forecast_batches(fc, tmp_path, on_exists="raise")
    for f in (tmp_path / "forecasts").rglob("*.parquet"):
        assert not (f.stat().st_mode & 0o222)
    logged = load_forecasts(tmp_path)
    assert len(logged) == len(fc)

    sc = score_forecasts(logged, cal)
    assert set(sc["horizon"]) == {1, 2} and len(sc) > 0
    # realized value equals the h-th later snapshot of that series
    r = sc.iloc[0]
    snaps = sorted(cal[cal.backend == r.backend].snapshot_ts_utc.unique())
    origin_i = snaps.index(pd.Timestamp(r.origin_ts))
    assert pd.Timestamp(r.target_ts) == snaps[origin_i + int(r.horizon)]
    row = cal[(cal.backend == r.backend) & (cal.snapshot_ts_utc == r.target_ts)
              & (cal.entity_id == r.entity_id) & (cal.metric == r.metric)]
    assert row["value"].iloc[0] == pytest.approx(r.realized)
    s = live_summary(sc)
    assert s["coverage"].between(0, 1).all()


def test_unrevealed_targets_are_not_scored(loop, tmp_path):
    cfg, cal, past, bundle = loop
    fc = forecast_latest(past, bundle)
    assert score_forecasts(fc, past).empty  # no future data yet


def test_bundle_roundtrip(loop, tmp_path):
    cfg, cal, past, bundle = loop
    path = bundle.save(tmp_path / "b.joblib")
    b2 = Bundle.load(path)
    f1 = forecast_latest(past, bundle, pd.Timestamp("2026-01-01", tz="UTC"))
    f2 = forecast_latest(past, b2, pd.Timestamp("2026-01-01", tz="UTC"))
    np.testing.assert_allclose(f1["point"].to_numpy(), f2["point"].to_numpy())


def test_best_chain_matches_brute_force():
    rng = np.random.default_rng(0)
    nodes = {i: float(rng.uniform(0.01, 0.1)) for i in range(7)}
    edges = {(a, b): float(rng.uniform(0.01, 0.2)) for a, b in itertools.combinations(range(7), 2) if rng.random() < 0.6}
    n = 4
    best = None
    for path in itertools.permutations(range(7), n):
        if all((min(a, b), max(a, b)) in edges for a, b in zip(path[:-1], path[1:], strict=True)):
            c = sum(nodes[q] for q in path) + sum(edges[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:], strict=True))
            best = c if best is None or c < best else best
    got = best_chain(nodes, edges, n, beam=500)
    assert got is not None and got[0] == pytest.approx(best)
    assert best_chain({0: 1.0, 1: 1.0}, {}, 2) is None


def test_protocol_guard(tmp_path, monkeypatch):
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "docs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs/PROTOCOL.md").write_text("v1")
    (tmp_path / "configs/default.yaml").write_text("seed: 0")
    git("add", ".")
    git("commit", "-qm", "init")
    with pytest.raises(ProtocolError):
        check_protocol_frozen("protocol-v1", tmp_path)  # tag missing
    git("tag", "protocol-v1")
    commit = check_protocol_frozen("protocol-v1", tmp_path)
    assert len(commit) == 40
    (tmp_path / "configs/default.yaml").write_text("seed: 1")
    with pytest.raises(ProtocolError):
        check_protocol_frozen("protocol-v1", tmp_path)  # config changed after freeze
    git("checkout", "--", "configs/default.yaml")

    out = tmp_path / "results/final"
    acquire_final_lock(out, commit)
    with pytest.raises(ProtocolError):
        acquire_final_lock(out, commit)  # test set already evaluated
    acquire_final_lock(out, commit, force=True)

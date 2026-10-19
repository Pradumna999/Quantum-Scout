import numpy as np
import pytest

from qscout.evaluation.splits import backend_bounds, rolling_origin_masks, window_masks
from qscout.features import target_col, target_ts_col


@pytest.mark.parametrize("mode", ["dev", "final"])
@pytest.mark.parametrize("h", [1, 2])
def test_purging_and_ordering(feats, tiny_cfg, mode, h):
    b = backend_bounds(feats, tiny_cfg)
    m = window_masks(feats, h, b, tiny_cfg, mode)
    tr, ca, ev = (feats[m[k]] for k in ("train", "calib", "eval"))
    assert len(tr) and len(ca) and len(ev)
    for name, d in (("train", tr), ("calib", ca), ("eval", ev)):
        assert d[target_col(h)].notna().all(), name
    for backend, bd in b.items():
        t = tr[tr.backend == backend]
        c = ca[ca.backend == backend]
        e = ev[ev.backend == backend]
        if backend in tiny_cfg.holdout_backends:
            assert t.empty and c.empty
            continue
        # no training label is realized at or after the end of the training window
        assert (t[target_ts_col(h)] < bd.t_train_end).all()
        # calibration origins start after training window; labels realized before calibration end
        cal_end = bd.t_calib_mid if mode == "dev" else bd.t_val_end
        assert (c["ts"] >= bd.t_train_end).all() and (c[target_ts_col(h)] < cal_end).all()
        assert (e["ts"] >= cal_end).all()
        # the sets are disjoint in rows
        assert not (set(t.index) & set(c.index)) and not (set(c.index) & set(e.index))
    # holdout backend appears only in eval
    assert set(ev.backend) >= set(tiny_cfg.holdout_backends)


def test_final_eval_is_strictly_after_val(feats, tiny_cfg):
    b = backend_bounds(feats, tiny_cfg)
    m = window_masks(feats, 1, b, tiny_cfg, "final")
    ev = feats[m["eval"]]
    for backend, bd in b.items():
        assert (ev[ev.backend == backend]["ts"] >= bd.t_val_end).all()


def test_rolling_origin_folds_expand_and_never_touch_test(feats, tiny_cfg):
    b = backend_bounds(feats, tiny_cfg)
    folds = rolling_origin_masks(feats, 1, b, tiny_cfg, n_folds=3)
    sizes = [int(f["train"].sum()) for f in folds]
    assert sizes == sorted(sizes) and sizes[0] > 0
    for f in folds:
        assert not (f["train"] & f["eval"]).any()
        for backend, bd in b.items():
            e = feats[f["eval"] & (feats.backend == backend).to_numpy()]
            assert (e[target_ts_col(1)] <= bd.t_val_end).all()


def test_bad_mode_raises(feats, tiny_cfg):
    with pytest.raises(ValueError):
        window_masks(feats, 1, backend_bounds(feats, tiny_cfg), tiny_cfg, "nope")


def test_no_row_in_two_roles_across_horizons(feats, tiny_cfg):
    b = backend_bounds(feats, tiny_cfg)
    for h in tiny_cfg.horizons:
        m = window_masks(feats, h, b, tiny_cfg, "dev")
        assert not (m["train"] & m["calib"]).any() and not (m["train"] & m["eval"]).any()
        assert np.all(~(m["calib"] & m["eval"]))

import pytest

from qscout.config import load_config
from qscout.features import build_features
from qscout.synthetic import make_synthetic


@pytest.fixture(scope="session")
def tiny_cfg():
    return load_config(
        horizons=(1, 2),
        holdout_backends=("synth_c",),
        n_boot=10,
        min_group=20,
        lgbm={"n_estimators": 20, "min_child_samples": 20},
    )


@pytest.fixture(scope="session")
def synth():
    cal, ev = make_synthetic(n_devices=3, rows=2, cols=4, cycles=200, seed=7)
    return cal, ev


@pytest.fixture(scope="session")
def feats(synth, tiny_cfg):
    return build_features(synth[0], tiny_cfg.horizons, tiny_cfg)

import os

import pandas as pd

from qscout.cli import main
from qscout.store import load_calibration


def test_cli_full_loop(tmp_path, capsys):
    root = str(tmp_path / "d")
    assert main(["synth", "--root", root, "--devices", "2", "--rows", "2", "--cols", "4", "--cycles", "170"]) == 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("horizons: [1, 2]\nn_boot: 5\nmin_group: 20\nlgbm: {n_estimators: 15, min_child_samples: 20}\n")
    model = str(tmp_path / "m.joblib")
    assert main(["train", "--root", root, "--config", str(cfg), "--model", model]) == 0
    assert main(["nightly", "--root", root, "--model", model]) == 0
    assert main(["nightly", "--root", root, "--model", model]) == 0  # idempotent
    assert "0 new batch" in capsys.readouterr().out
    assert main(["score", "--root", root]) == 0
    assert main(["run-dev", "--root", root, "--config", str(cfg), "--out", str(tmp_path / "r"), "--folds", "2",
                 "--holdout", "synth_b"]) == 0
    assert (tmp_path / "r" / "e4_trust.csv").exists()
    # the final run refuses without a frozen protocol tag
    rc = main(["run-final", "--root", root, "--config", str(cfg), "--out", str(tmp_path / "f"),
               "--protocol-tag", "nope", "--repo", str(tmp_path)])
    assert rc == 2 and not (tmp_path / "f" / "e4_trust.csv").exists()
    assert len(load_calibration(root)) > 0


def test_dashboard_pages_render_without_error(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from qscout.config import load_config
    from qscout.serve import forecast_latest, train_bundle, write_forecast_batches
    from qscout.store import write_curated
    from qscout.synthetic import make_synthetic

    root = tmp_path / "data"
    cal, _ = make_synthetic(2, 2, 4, 170, 0)
    write_curated(cal, root)
    past = cal[cal.snapshot_ts_utc <= sorted(cal.snapshot_ts_utc.unique())[-6]]
    cfg = load_config(horizons=(1, 2), lgbm={"n_estimators": 15, "min_child_samples": 20}, min_group=20)
    write_forecast_batches(forecast_latest(past, train_bundle(past, cfg)), root)
    monkeypatch.setenv("QSCOUT_DATA", str(root))
    monkeypatch.setenv("QSCOUT_RESULTS", str(tmp_path / "none"))
    app = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
    at = AppTest.from_file(app, default_timeout=90).run()
    assert not at.exception
    for page in ("Series", "Forward test", "Backtests"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    assert isinstance(pd.Timestamp.now(), pd.Timestamp)

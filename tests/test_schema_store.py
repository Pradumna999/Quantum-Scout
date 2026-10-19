import numpy as np
import pandas as pd
import pytest

from qscout.schema import COLUMNS, validate_calibration
from qscout.store import AppendOnlyViolation, build_curated, write_once, write_raw_snapshot


def _row(**kw):
    base = dict(backend="b", snapshot_ts_utc="2026-01-01T00:00:00Z", entity_type="qubit",
                entity_id="0", metric="t1", value=100.0, unit="us", source="t")
    base.update(kw)
    return base


def test_validation_quarantines_each_violation():
    df = pd.DataFrame([
        _row(),                                         # ok
        _row(entity_id="1", value=np.nan),              # dropped silently
        _row(entity_id="2", value=-5.0),                # out of range
        _row(entity_id="3", metric="readout_error", value=1.5),
        _row(entity_id="4", metric="bogus"),
        _row(entity_id="5", metric="two_qubit_error"),  # qubit entity on edge metric
        _row(entity_id="6", value=np.inf),
        _row(entity_id="7", value=5.0), _row(entity_id="7", value=6.0),  # duplicate key
    ])
    clean, bad = validate_calibration(df)
    reasons = dict(zip(bad["entity_id"], bad["reason"], strict=True))
    assert reasons["2"] == "out_of_range" and reasons["3"] == "out_of_range"
    assert reasons["4"] == "unknown_metric" and reasons["5"] == "entity_type_mismatch"
    assert reasons["6"] == "non_finite"
    assert "1" not in set(clean["entity_id"]) | set(bad["entity_id"])
    dup = clean[clean["entity_id"] == "7"]
    assert len(dup) == 1 and dup["value"].iloc[0] == 6.0  # last wins
    assert (bad["entity_id"] == "7").sum() == 1


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        validate_calibration(pd.DataFrame({"backend": ["b"]}))


def test_raw_snapshot_idempotent_readonly_and_curated(tmp_path):
    df = pd.DataFrame([_row(), _row(entity_id="1", value=90.0)])[COLUMNS]
    p = write_raw_snapshot(df, tmp_path)
    assert p is not None and not (p.stat().st_mode & 0o222)
    assert write_raw_snapshot(df, tmp_path) is None  # same calibration timestamp -> duplicate
    later = df.assign(snapshot_ts_utc="2026-01-02T00:00:00Z")
    assert write_raw_snapshot(later, tmp_path) is not None
    info = build_curated(tmp_path)
    assert info["n_rows"] == 4 and info["n_quarantined"] == 0


def test_snapshot_requires_single_timestamp(tmp_path):
    df = pd.DataFrame([_row(), _row(snapshot_ts_utc="2026-01-02T00:00:00Z")])
    with pytest.raises(ValueError):
        write_raw_snapshot(df, tmp_path)


def test_write_once_refuses_overwrite(tmp_path):
    df = pd.DataFrame({"a": [1]})
    write_once(df, tmp_path / "x.parquet")
    with pytest.raises(AppendOnlyViolation):
        write_once(df, tmp_path / "x.parquet")
    assert write_once(df, tmp_path / "x.parquet", on_exists="skip") is None

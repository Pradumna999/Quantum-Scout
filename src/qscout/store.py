"""Immutable raw snapshots, curated table, and append-only helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .schema import KEY, normalize, soft_flags, validate_calibration


class AppendOnlyViolation(RuntimeError):
    pass


def content_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or list(df.columns)
    d = df[cols].sort_values(cols).reset_index(drop=True)
    h = pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()
    return hashlib.sha256(h).hexdigest()[:16]


def _stamp(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")


def write_once(df: pd.DataFrame, path: Path, on_exists: str = "raise") -> Path | None:
    """Write a parquet file that is never modified afterwards (mode 0444)."""
    path = Path(path)
    if path.exists():
        if on_exists == "skip":
            return None
        raise AppendOnlyViolation(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.chmod(0o444)
    return path


def write_raw_snapshot(df: pd.DataFrame, root: str | Path) -> Path | None:
    """Store one backend/one calibration timestamp. Returns None if it is already stored."""
    df = normalize(df)
    if df["backend"].nunique() != 1 or df["snapshot_ts_utc"].nunique() != 1:
        raise ValueError("a raw snapshot must contain exactly one backend and one timestamp")
    ts = df["snapshot_ts_utc"].iloc[0]
    backend = df["backend"].iloc[0]
    d = Path(root) / "raw" / f"backend={backend}" / f"date={ts:%Y-%m-%d}"
    if d.exists() and list(d.glob(f"snap_{_stamp(ts)}_*.parquet")):
        return None
    return write_once(df, d / f"snap_{_stamp(ts)}_{content_hash(df)}.parquet", on_exists="skip")


def build_curated(root: str | Path) -> dict:
    """Validate and merge all raw snapshots into data/curated/calibration.parquet."""
    root = Path(root)
    files = sorted((root / "raw").rglob("snap_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw snapshots under {root / 'raw'}")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    clean, bad = validate_calibration(raw)
    write_curated(clean, root)
    out = root / "curated"
    bad.to_parquet(out / "quarantine.parquet", index=False)
    return {
        "n_files": len(files),
        "n_rows": len(clean),
        "n_quarantined": len(bad),
        **soft_flags(clean),
    }


def write_curated(clean: pd.DataFrame, root: str | Path) -> Path:
    out = Path(root) / "curated"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration.parquet"
    clean.sort_values(KEY).reset_index(drop=True).to_parquet(path, index=False)
    return path


def load_calibration(root: str | Path) -> pd.DataFrame:
    path = Path(root) / "curated" / "calibration.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `qscout build-curated` (or `qscout synth`)")
    return pd.read_parquet(path)

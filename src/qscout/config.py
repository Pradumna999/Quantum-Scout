"""Typed configuration. YAML values override the defaults below."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


def _default_lgbm() -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }


@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 2, 3, 7)
    quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)
    alpha: float = 0.10
    split_fracs: tuple[float, float] = (0.6, 0.2)
    holdout_backends: tuple[str, ...] = ()
    min_history: int = 20
    seed: int = 0

    jump_k: float = 4.0
    jump_window: int = 30
    scale_floor: float = 0.005
    tsl_cap: int = 60
    ph_delta: float = 0.25
    ph_lambda: float = 8.0
    ph_min_n: int = 15

    aci_gamma: float = 0.01
    min_group: int = 50

    large_error_pct: float = 0.90
    trust_fprs: tuple[float, float] = (0.25, 0.10)

    block_len: int = 7
    n_boot: int = 200

    lgbm: dict[str, Any] = field(default_factory=_default_lgbm)

    def with_(self, **kw: Any) -> Config:
        return replace(self, **kw)


_TUPLE_FIELDS = {"horizons", "quantiles", "split_fracs", "holdout_backends", "trust_fprs"}


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update(overrides)
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for k in _TUPLE_FIELDS & set(raw):
        raw[k] = tuple(raw[k])
    if "lgbm" in raw:
        raw["lgbm"] = {**_default_lgbm(), **raw["lgbm"]}
    cfg = Config(**raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.horizons or any(h < 1 for h in cfg.horizons):
        raise ValueError("horizons must be positive integers")
    if not (0 < cfg.alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    tr, va = cfg.split_fracs
    if not (0 < tr < 1 and 0 < va < 1 and tr + va < 1):
        raise ValueError("split_fracs must be two positive fractions summing to < 1")
    if list(cfg.quantiles) != sorted(cfg.quantiles) or cfg.quantiles[1] != 0.5:
        raise ValueError("quantiles must be ascending with the median in the middle")
    if abs((cfg.quantiles[2] - cfg.quantiles[0]) - (1 - cfg.alpha)) > 1e-9:
        raise ValueError("quantile span must equal 1 - alpha (e.g. 0.05/0.95 for alpha=0.10)")


def config_hash(cfg: Config) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

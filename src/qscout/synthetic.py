"""Synthetic calibration data with known ground truth.

Used for (a) pipeline development before real data is in, (b) unit/property tests and
(c) experiment E8, where change-points are known. Nothing computed on this data says anything
about real hardware.

Dynamics per metric (all in log space): per-entity AR(1) + persistent level jumps + shared
device-level events + two-state (TLS-like) telegraph switching on a subset of entities +
temporary volatility regimes + observation noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import COLUMNS
from .transforms import ENTITY_TYPE, METRICS, UNIT


@dataclass(frozen=True)
class MetricSpec:
    mu: float          # log-level
    spread: float      # sd of entity base level
    phi: float         # AR(1) coefficient
    sigma_ar: float    # AR innovation sd
    sigma_obs: float   # measurement noise sd (irreducible)
    jump_p: float      # per-entity per-cycle level-jump probability
    jump_sd: float
    tls_delta: float   # log-shift while the TLS-like state is "on"
    dev_sd: float      # device-event magnitude sd


SPECS: dict[str, MetricSpec] = {
    "t1": MetricSpec(np.log(150.0), 0.40, 0.92, 0.05, 0.06, 0.004, 0.35, -0.7, 0.15),
    "t2": MetricSpec(np.log(100.0), 0.45, 0.90, 0.06, 0.08, 0.004, 0.35, -0.5, 0.15),
    "readout_error": MetricSpec(np.log(0.015), 0.50, 0.90, 0.07, 0.10, 0.004, 0.50, 0.8, 0.20),
    "sx_error": MetricSpec(np.log(3e-4), 0.40, 0.90, 0.06, 0.12, 0.003, 0.40, 0.5, 0.15),
    "two_qubit_error": MetricSpec(np.log(7e-3), 0.50, 0.88, 0.10, 0.15, 0.006, 0.60, 1.0, 0.20),
}

DEVICE_EVENT_P = 0.01
TLS_FRACTION = 0.2
TLS_P_ON, TLS_P_OFF = 0.03, 0.10
REGIME_FRACTION, REGIME_LEN, REGIME_MULT = 0.25, 40, 3.0
MISSING_CYCLE_P = 0.02
DEAD_QUBIT_FRACTION = 0.03


def grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    return edges


def _simulate_metric(
    spec: MetricSpec,
    n: int,
    T: int,
    rng: np.random.Generator,
    device_event: np.ndarray,
    regime_mask: np.ndarray,
    tls_mask: np.ndarray,
) -> np.ndarray:
    """Returns the log-valued (T, n) array of observed values."""
    base = spec.mu + rng.normal(0, spec.spread, n)
    x = np.zeros((T, n))
    a = np.zeros(n)
    level = np.zeros(n)
    tls = np.zeros(n, dtype=bool)
    for t in range(T):
        sig = np.where(regime_mask[t], spec.sigma_ar * REGIME_MULT, spec.sigma_ar)
        a = spec.phi * a + rng.normal(0, 1, n) * sig
        if device_event[t]:
            common = rng.normal(0, spec.dev_sd)
            level += common + rng.normal(0, 0.3 * spec.dev_sd, n)
        jumps = rng.random(n) < spec.jump_p
        level += jumps * rng.normal(0, spec.jump_sd, n)
        u = rng.random(n)
        tls = np.where(tls, u > TLS_P_OFF, u < TLS_P_ON) & tls_mask
        x[t] = base + a + level + tls * spec.tls_delta + rng.normal(0, spec.sigma_obs, n)
    return x


def make_device(
    name: str,
    rows: int = 3,
    cols: int = 9,
    cycles: int = 400,
    seed: int = 0,
    start: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_q = rows * cols
    edges = grid_edges(rows, cols)
    T = cycles
    device_event = rng.random(T) < DEVICE_EVENT_P

    # calibration timestamps: roughly daily with jitter, some cycles missing entirely
    base_ts = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.arange(T), unit="D")
    jitter = pd.to_timedelta(rng.uniform(1.0, 4.0, T), unit="h")
    ts_all = base_ts + jitter
    keep_cycle = rng.random(T) > MISSING_CYCLE_P
    keep_cycle[0] = True

    dead = rng.random(n_q) < DEAD_QUBIT_FRACTION
    dead_start = rng.integers(T // 4, T - 60, n_q)
    dead_len = rng.integers(20, 60, n_q)
    alive = np.ones((T, n_q), dtype=bool)
    for q in np.flatnonzero(dead):
        alive[dead_start[q] : dead_start[q] + dead_len[q], q] = False

    frames: list[pd.DataFrame] = []
    t1_val: np.ndarray | None = None
    events: list[tuple] = [(name, ts_all[t], "device_event", "*", "*") for t in np.flatnonzero(device_event)]

    for metric in METRICS:
        spec = SPECS[metric]
        n = len(edges) if ENTITY_TYPE[metric] == "edge" else n_q
        regime_mask = np.zeros((T, n), dtype=bool)
        for e in np.flatnonzero(rng.random(n) < REGIME_FRACTION):
            s = int(rng.integers(T // 5, max(T // 5 + 1, T - REGIME_LEN)))
            regime_mask[s : s + REGIME_LEN, e] = True
        tls_mask = rng.random(n) < TLS_FRACTION
        logx = _simulate_metric(spec, n, T, rng, device_event, regime_mask, tls_mask)
        val = np.exp(logx)
        if metric == "t1":
            t1_val = val
        if metric == "t2" and t1_val is not None:
            val = np.minimum(val, 2.0 * t1_val)  # physical bound T2 <= 2*T1
        if metric.endswith("_error"):
            val = np.clip(val, 1e-6, 0.5)
        ids = (
            [f"{min(a, b)}-{max(a, b)}" for a, b in edges]
            if ENTITY_TYPE[metric] == "edge"
            else [str(i) for i in range(n_q)]
        )
        obs = np.ones((T, n), dtype=bool)
        if ENTITY_TYPE[metric] == "qubit":
            obs &= alive
        else:
            for j, (a, b) in enumerate(edges):
                obs[:, j] = alive[:, a] & alive[:, b]
        obs &= keep_cycle[:, None]
        t_idx, e_idx = np.nonzero(obs)
        frames.append(
            pd.DataFrame(
                {
                    "backend": name,
                    "snapshot_ts_utc": ts_all[t_idx],
                    "entity_type": ENTITY_TYPE[metric],
                    "entity_id": np.asarray(ids)[e_idx],
                    "metric": metric,
                    "value": val[t_idx, e_idx],
                    "unit": UNIT[metric],
                    "source": "synthetic",
                }
            )
        )
    cal = pd.concat(frames, ignore_index=True)[COLUMNS]
    ev = pd.DataFrame(events, columns=["backend", "ts", "kind", "entity_id", "metric"])
    return cal, ev


def make_synthetic(
    n_devices: int = 3, rows: int = 3, cols: int = 9, cycles: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cals, evs = [], []
    for i in range(n_devices):
        c, e = make_device(f"synth_{chr(97 + i)}", rows, cols, cycles, seed + 1000 * i)
        cals.append(c)
        evs.append(e)
    return pd.concat(cals, ignore_index=True), pd.concat(evs, ignore_index=True)


# --------------------------------------------------------------------------------------------
# E8: series with known change-points
# --------------------------------------------------------------------------------------------


@dataclass
class DetectionPanel:
    logy: np.ndarray                              # (T, N)
    jumps: list[tuple[int, int]]                  # (t, series) level jumps
    regimes: list[tuple[int, int, int]]           # (t_start, t_end, series) volatility regimes


def make_detection_panel(
    n_series: int = 200,
    T: int = 400,
    seed: int = 0,
    sigma: float = 0.05,
    phi: float = 0.9,
    jump_sd: float = 0.4,
    jump_p: float = 0.004,
    regime_mult: float = 3.0,
    regime_len: int = 60,
    regime_fraction: float = 0.5,
    obs_sd: float = 0.05,
) -> DetectionPanel:
    rng = np.random.default_rng(seed)
    x = np.zeros((T, n_series))
    a = np.zeros(n_series)
    level = np.zeros(n_series)
    sig = np.full((T, n_series), sigma)
    regimes = []
    for n in np.flatnonzero(rng.random(n_series) < regime_fraction):
        s = int(rng.integers(120, T - regime_len))
        sig[s : s + regime_len, n] = sigma * regime_mult
        regimes.append((s, s + regime_len, int(n)))
    jumps = []
    for t in range(T):
        a = phi * a + rng.normal(0, 1, n_series) * sig[t]
        j = (rng.random(n_series) < jump_p) & (t >= 40)  # not during warm-up
        level += j * rng.normal(0, jump_sd, n_series) * np.where(rng.random(n_series) < 0.5, 1, -1)
        jumps.extend((t, int(n)) for n in np.flatnonzero(j))
        x[t] = a + level + rng.normal(0, obs_sd, n_series)
    return DetectionPanel(x, jumps, regimes)

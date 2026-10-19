"""Metric catalogue and the log transform used everywhere in modelling."""

from __future__ import annotations

import numpy as np

METRICS: tuple[str, ...] = ("t1", "t2", "readout_error", "sx_error", "two_qubit_error")
ENTITY_TYPE: dict[str, str] = {
    "t1": "qubit",
    "t2": "qubit",
    "readout_error": "qubit",
    "sx_error": "qubit",
    "two_qubit_error": "edge",
}
UNIT: dict[str, str] = {
    "t1": "us",
    "t2": "us",
    "readout_error": "1",
    "sx_error": "1",
    "two_qubit_error": "1",
}
METRIC_ID: dict[str, int] = {m: i for i, m in enumerate(METRICS)}
EPS = 1e-6  # floor before log; errors of exactly 0 do occur in calibration data


def to_log(x):
    """Natural log with a floor. Works on scalars, arrays and DataFrames."""
    return np.log(np.maximum(x, EPS))


def from_log(x):
    return np.exp(x)

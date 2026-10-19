"""Extract calibration snapshots from IBM backends via the Qiskit Target interface.

Only the duck-typed surface of `backend.target` is used (num_qubits, qubit_properties,
operation_names, target[name] -> {qargs: InstructionProperties}), so it works for real
`IBMBackend` objects and for the fake backends in `qiskit_ibm_runtime.fake_provider`.

Timestamp semantics: the snapshot timestamp is the *calibration* time reported by the backend
(`properties().last_update_date`), not the download time, so repeated downloads between two
calibrations collapse to a single snapshot.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..schema import COLUMNS, normalize
from ..store import write_raw_snapshot
from ..transforms import UNIT

log = logging.getLogger(__name__)

TWO_QUBIT_GATES = ("cz", "ecr", "cx")  # first one present on the target is used
ONE_QUBIT_GATE = "sx"
SOURCE = "ibm_runtime"


def _calibration_time(backend) -> pd.Timestamp | None:
    try:
        props = backend.properties()
    except Exception as exc:  # noqa: BLE001 - network/auth problems must not crash a fleet scan
        log.warning("properties() failed for %s: %s", getattr(backend, "name", "?"), exc)
        return None
    ts = getattr(props, "last_update_date", None) if props is not None else None
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def extract_snapshot(backend, snapshot_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """One backend, one calibration -> long-format rows (may be empty)."""
    ts = snapshot_ts if snapshot_ts is not None else _calibration_time(backend)
    if ts is None:
        return pd.DataFrame(columns=COLUMNS)
    target = backend.target
    name = backend.name
    rows: list[tuple] = []

    def add(entity_type: str, entity_id: str, metric: str, value) -> None:
        if value is None:
            return
        rows.append((name, ts, entity_type, entity_id, metric, float(value), UNIT[metric], SOURCE))

    qprops = target.qubit_properties or []
    for q, qp in enumerate(qprops):
        if qp is None:
            continue
        t1 = getattr(qp, "t1", None)
        t2 = getattr(qp, "t2", None)
        add("qubit", str(q), "t1", None if t1 is None else t1 * 1e6)  # seconds -> microseconds
        add("qubit", str(q), "t2", None if t2 is None else t2 * 1e6)

    ops = set(target.operation_names)
    if "measure" in ops:
        for qargs, props in target["measure"].items():
            if props is not None and len(qargs) == 1:
                add("qubit", str(qargs[0]), "readout_error", props.error)
    if ONE_QUBIT_GATE in ops:
        for qargs, props in target[ONE_QUBIT_GATE].items():
            if props is not None and len(qargs) == 1:
                add("qubit", str(qargs[0]), "sx_error", props.error)

    gate2 = next((g for g in TWO_QUBIT_GATES if g in ops), None)
    if gate2 is not None:
        acc: dict[tuple[int, int], list[float]] = {}
        for qargs, props in target[gate2].items():
            if props is None or props.error is None or len(qargs) != 2:
                continue
            a, b = sorted(int(x) for x in qargs)
            acc.setdefault((a, b), []).append(float(props.error))
        for (a, b), errs in acc.items():  # both directions, if present, are averaged
            add("edge", f"{a}-{b}", "two_qubit_error", sum(errs) / len(errs))

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return normalize(pd.DataFrame(rows, columns=COLUMNS))


def get_service(token: str | None = None, channel: str | None = None, instance: str | None = None):
    """Create a QiskitRuntimeService. With no arguments it uses the saved account or the
    QISKIT_IBM_TOKEN / QISKIT_IBM_CHANNEL / QISKIT_IBM_INSTANCE environment variables."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    kw = {k: v for k, v in dict(token=token, channel=channel, instance=instance).items() if v}
    return QiskitRuntimeService(**kw)


def snapshot_fleet(
    service,
    root: str | Path,
    backend_names: Iterable[str] | None = None,
    operational_only: bool = True,
) -> dict[str, str]:
    """Download the current calibration of each backend and store new snapshots.

    Returns {backend_name: "stored" | "duplicate" | "empty" | "error: ..."}.
    """
    if backend_names:
        backends = [service.backend(n) for n in backend_names]
    else:
        kw = {"simulator": False}
        if operational_only:
            kw["operational"] = True
        backends = service.backends(**kw)
    status: dict[str, str] = {}
    for b in backends:
        try:
            df = extract_snapshot(b)
            if df.empty:
                status[b.name] = "empty"
                continue
            status[b.name] = "stored" if write_raw_snapshot(df, root) else "duplicate"
        except Exception as exc:  # noqa: BLE001
            log.exception("snapshot failed for %s", getattr(b, "name", "?"))
            status[getattr(b, "name", "?")] = f"error: {exc}"
    return status

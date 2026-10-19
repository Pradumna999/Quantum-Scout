"""IBM extraction is checked against real Qiskit `Target` objects (fake backends).

It runs in a subprocess: importing Qiskit and LightGBM into one interpreter can exhaust glibc's
static TLS block on some machines, and the production code never needs both at once.
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

if importlib.util.find_spec("qiskit_ibm_runtime") is None:
    pytest.skip("qiskit-ibm-runtime not installed", allow_module_level=True)

SCRIPT = textwrap.dedent(
    """
    import pandas as pd
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeSherbrooke
    from qscout.ingest.ibm import extract_snapshot
    from qscout.schema import validate_calibration

    for fake, gate in ((FakeSherbrooke, "ecr"), (FakeFez, "cz")):
        b = fake()
        df = extract_snapshot(b)
        n = b.target.num_qubits
        counts = df.groupby("metric").size().to_dict()
        for m in ("t1", "t2", "readout_error", "sx_error"):
            assert counts[m] == n, (b.name, m, counts)
        pairs = {tuple(sorted(q)) for q in b.target[gate]}
        assert counts["two_qubit_error"] == len(pairs), (b.name, counts)
        clean, bad = validate_calibration(df)
        assert len(bad) == 0 and len(clean) == len(df), bad.head()
        t1_q0 = df[(df.metric == "t1") & (df.entity_id == "0")]["value"].iloc[0]
        assert abs(t1_q0 - b.target.qubit_properties[0].t1 * 1e6) < 1e-9
        expect = pd.Timestamp(b.properties().last_update_date).tz_convert("UTC")
        assert (df["snapshot_ts_utc"] == expect).all()
        assert (df["backend"] == b.name).all()

    class NoProps:
        name = "x"
        def properties(self):
            return None
    assert extract_snapshot(NoProps()).empty
    print("OK")
    """
)


def test_extract_from_real_target_objects():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]

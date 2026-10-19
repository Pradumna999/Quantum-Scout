# Data card (fill in after the source audit; this is a blocking step)

- **Source and licence:** _e.g. the IBM calibration-drift dataset you use, and/or own snapshots via `qscout ingest-ibm`_
- **Provenance check:** how the schema was verified; mapping of source columns to `qscout.schema.COLUMNS`
- **Coverage:** backends, processor families, first/last calibration timestamp, number of snapshots per backend
- **Cadence:** typical gap between calibrations, and irregularities (missing days, several calibrations per day)
- **Known gaps:** disabled qubits, coupling-map or firmware changes, retired devices
- **Contract results:** output of `qscout build-curated` (rows, quarantined rows by reason, share with T2 > 2*T1)
- **Units:** T1/T2 in microseconds; error rates dimensionless
- **Timestamp semantics:** the backend's calibration time (`properties().last_update_date`), UTC
- **Limitations:** calibration values are measurements with estimation noise and are a proxy for circuit-level error

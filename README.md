# Q-Scout

Forecasting, calibrated uncertainty and trust scoring for quantum-hardware calibration drift.

```
IBM calibrations ─► raw (immutable) ─► curated + contract ─► point-in-time features
      ─► LightGBM quantiles ─► CQR / ACI intervals ─► trust score ─► append-only forecast log
      ─► scored against realized calibrations ─► dashboard
```

Everything is judged against **persistence** (carry the last calibration forward). Targets are the change in
log value from the origin cycle to the h-th next calibration cycle, so a prediction of 0 *is* persistence.

## Setup (Python 3.10+; tested on 3.12)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add only [ibm] if you just want ingestion
pytest                         # ~1 minute; everything should pass
```

## Run-book

**1. Smoke-test the whole pipeline on synthetic data (plumbing only, says nothing about hardware).**
```bash
qscout synth --root data_synth
qscout run-dev --root data_synth --holdout synth_c --out results/synth_dev
qscout e8 --out results/e8
```

**2. Start collecting real data now (every day of delay is lost forward-test data).**
```bash
export QISKIT_IBM_TOKEN=...            # and QISKIT_IBM_INSTANCE if you use several instances
export QISKIT_IBM_CHANNEL=ibm_quantum_platform
qscout ingest-ibm --root data          # all operational non-simulator backends; or --backends NAME ...
qscout build-curated --root data       # validates, quarantines bad rows, writes data/curated/
```
Run `ingest-ibm` at least twice a day (or enable `.github/workflows/nightly_ingest.yml`). Snapshots are keyed by
the backend's own calibration timestamp, so repeats are harmless.

**3. Audit the data before any modelling.** Fill in `docs/DATA_CARD.md` from the `build-curated` output.
If you use a public calibration-drift dataset instead of (or in addition to) your own snapshots, write a loader that
maps it to `qscout.schema.COLUMNS` (long format: backend, snapshot_ts_utc, entity_type, entity_id, metric, value,
unit, source), pass it through `validate_calibration`, and `write_curated`. Verify its provenance and schema first.

**4. Development experiments (E1-E7). Never touches the test period.**
```bash
qscout run-dev --root data --config configs/default.yaml --holdout BACKEND_A --out results/dev
```
Tables land in `results/dev/*.csv` (E1 characterization, E2 point skill + DM tests + rolling origin, E3 coverage by
method and volatility regime, E4 trust, E5 ablation, E6 transfer, E7 downstream) plus `run_meta.json`.
Use `--holdout` to keep whole backends out of fitting (needs at least 3 backends to be meaningful).

**5. Freeze the protocol, then evaluate the test period once.**
```bash
$EDITOR docs/PROTOCOL.md               # fill in, commit
git add -A && git commit -m "freeze protocol" && git tag protocol-v1
qscout run-final --root data --holdout BACKEND_A --protocol-tag protocol-v1 --out results/final
```
Refuses to run if the tag is missing, `docs/PROTOCOL.md` or `configs/` differ from the tag, or `results/final`
already holds a lock (`--force` overrides and is the only way to run it twice).

**6. Train, forecast, score (the forward test).**
```bash
qscout train   --root data --model models/bundle.joblib
qscout nightly --root data --model models/bundle.joblib   # append-only; safe to re-run
qscout score   --root data                                 # fills data/scores/ once outcomes exist
```
Retrain periodically (the bundle records its `model_id`; each forecast stores it). Forecast batches are written once
and made read-only; scoring is derived and can be regenerated at any time.

**7. Dashboard.**
```bash
QSCOUT_DATA=data QSCOUT_RESULTS=results streamlit run dashboard/app.py
```

## Layout

| Path | Purpose |
|---|---|
| `src/qscout/schema.py`, `store.py` | long-format contract, immutable raw snapshots, curated table |
| `src/qscout/ingest/ibm.py` | IBM `Target` -> long format (checked against real `Target` objects from the fake backends) |
| `src/qscout/features.py`, `drift.py` | causal features, jump flags, Page-Hinkley |
| `src/qscout/models.py`, `conformal.py`, `trust.py`, `pipeline.py` | forecasters, split/CQR/ACI, trust scorer, per-horizon stack |
| `src/qscout/evaluation/` | purged temporal splits, metrics, DM / Holm / block bootstrap |
| `src/qscout/experiments.py`, `downstream.py` | E1-E8 |
| `src/qscout/serve.py` | bundle, nightly forecasts, scoring |
| `src/qscout/protocol.py` | frozen-protocol and one-shot test guards |
| `dashboard/app.py` | Streamlit app (reads stored results only) |
| `docs/` | protocol template, data card, model card |

## Design notes you should know before trusting a number

- **Leakage control** is enforced by tests: features at origin *k* equal those from a panel truncated at *k*, and
  are unchanged when all later data is corrupted; no training label is realized after the end of the training window.
- **Rows are not independent.** All qubits on a device move together on device-level events, so row counts overstate
  the sample. Confidence intervals use a moving block bootstrap over time; the forward-test `n_origins` is the
  honest sample size.
- **ACI** deviates from the textbook in two documented ways: one alpha state per (backend, metric) updated with the
  mean miss rate of all series at an origin, and finite extrapolation beyond the largest calibration score instead of an
  infinite interval when alpha_t < 0 (`conformal.level_quantile`).
- **Trust tiers carry `trust_val_auroc`.** If it is near 0.5 the tiers are noise. On the synthetic data it is, because
  the synthetic large errors come from unpredictable future jumps. That is a property of that data, not a result.
- **Not implemented** (deliberately, to keep what exists verified): DuckDB store (Parquet + pandas is used), Hydra
  (one YAML), MLflow (CSV/JSON results with config hash and git SHA), FastAPI (the dashboard reads files), N-BEATS /
  PatchTST / Chronos baselines, CRPS (only three quantiles are modelled, so interval score and pinball are used),
  statsforecast ETS/ARIMA. Add them behind the `fit/predict` interface in `models.py`.
- Importing Qiskit and LightGBM into one interpreter can fail on some machines with `libgomp ... cannot allocate
  memory in static TLS block`. No CLI command needs both; the IBM extraction test therefore runs in a subprocess.
- The GitHub workflows are provided but were **not executed** in the environment this was built in.

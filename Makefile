PY ?= python
ROOT ?= data
CONFIG ?= configs/default.yaml

.PHONY: install test lint synth dev e8 train nightly score dashboard clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	ruff check src tests dashboard

synth:            ## synthetic data for plumbing only
	qscout synth --root data_synth

dev:              ## experiments E1-E7, test period untouched
	qscout run-dev --root $(ROOT) --config $(CONFIG) --out results/dev

e8:
	qscout e8 --config $(CONFIG) --out results/e8

train:
	qscout train --root $(ROOT) --config $(CONFIG) --model models/bundle.joblib

nightly:
	qscout nightly --root $(ROOT) --model models/bundle.joblib

score:
	qscout score --root $(ROOT)

dashboard:
	QSCOUT_DATA=$(ROOT) QSCOUT_RESULTS=results streamlit run dashboard/app.py

clean:
	rm -rf results models .pytest_cache .ruff_cache

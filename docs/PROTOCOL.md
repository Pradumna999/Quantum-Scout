# Q-Scout evaluation protocol (template: complete, commit, then tag `protocol-v1`)

Freeze this file and `configs/` with `git tag protocol-v1` **before** running `qscout run-final`.
`run-final` refuses to run unless the tag exists, `docs/PROTOCOL.md` and `configs/` are identical to
the tag, and no earlier final run has locked `results/final`. Later deviations go in the
changelog at the bottom, never as edits above it.

## Data
- Source(s), time span, backends, n snapshots: _fill in from the data card_
- Backends held out for transfer (E6): _fill in_
- Split fractions: train 0.6 / val 0.2 / test 0.2 per backend (config `split_fracs`)

## Targets
Change in log value from the origin calibration to the h-th next calibration, h in {1, 2, 3, 7},
for T1, T2, readout error, 1Q (sx) error, 2Q error.

## Pre-specified hypotheses and decision rules
| ID | Hypothesis | Decision rule |
|----|------------|---------------|
| H1 | The best learned forecaster beats persistence on some metrics, not all | skill (1 - MSE/MSE_persist) >= 0.05 and Holm-adjusted DM p < 0.05, per metric x horizon |
| H2 | Marginal split conformal under-covers in high-volatility regimes; ACI restores long-run coverage | coverage in the "high" regime < 0.85 for `pconf_abs`; `cqr_aci` within 0.03 of 0.90 |
| H3 | The trust score detects large errors better than interval width alone | delta AUROC (trust - width_cqr) >= 0.03 with the 95% block-bootstrap CI excluding 0 |
| H4 | Models transfer to held-out backends | report coverage, skill, AUROC for `holdout` vs `in_dist` (no threshold) |
| H5 | Forecast-informed qubit selection beats last-calibration selection | `forecast_vs_last` gain CI excludes 0 (E7) |

"Large error" = |error of the LightGBM median| above the series' own 90th percentile of
|persistence error| on training rows.

## Procedure
1. `qscout run-dev` for all development (never touches the test period).
2. Freeze: commit, `git tag protocol-v1`.
3. `qscout run-final --protocol-tag protocol-v1` once. Report every pre-specified row, positive or not.
4. Forward test: `qscout nightly` + `qscout score` accumulate append-only forecasts independent of 1-3.

## Changelog
_none_

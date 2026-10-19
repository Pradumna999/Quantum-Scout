# Model card: Q-Scout forecaster, intervals, trust tiers

- **Task:** forecast calibration metrics h calibration cycles ahead; attach a 90% interval and a trust tier.
- **Forecast target:** change in log value vs the origin (0 = persistence). Always compare against persistence.
- **Intervals:** CQR around LightGBM quantiles, calibrated per metric on a later time window (`pipeline.fit_horizon`).
  In evaluation, ACI adapts the interval online with delayed feedback (`conformal.aci_cqr`); the nightly path uses static CQR.
- **Trust tiers:** logistic score of drift/uncertainty signals estimating P(large error). `ok / caution / unreliable`
  are set by false-alarm rates (25% / 10% of non-large-error rows on a held-back calibration slice).
- **Read `trust_val_auroc` before using tiers.** It is the AUROC on that held-back slice. Near 0.5 means the tiers carry no signal.
- **Not for:** predicting circuit outputs, replacing calibration, or intra-day drift (daily snapshots cannot see it).
- **Known failure modes:** jumps and TLS switches that have not happened yet are unpredictable by construction;
  prediction quality falls after firmware or procedure changes; few devices means weak evidence on transfer.

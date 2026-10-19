"""Command-line entry point: `qscout <command>` or `python -m qscout.cli <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import config_hash, load_config
from .protocol import ProtocolError, acquire_final_lock, check_protocol_frozen
from .store import build_curated, load_calibration, write_curated


def _cfg(args: argparse.Namespace):
    over = {}
    if getattr(args, "holdout", None):
        over["holdout_backends"] = tuple(args.holdout)
    return load_config(args.config, **over)


def cmd_synth(args: argparse.Namespace) -> int:
    from .synthetic import make_synthetic

    cal, ev = make_synthetic(args.devices, args.rows, args.cols, args.cycles, args.seed)
    path = write_curated(cal, args.root)
    ev.to_parquet(Path(args.root) / "curated" / "synthetic_events.parquet", index=False)
    print(f"wrote {len(cal):,} synthetic rows to {path}  (synthetic: results say nothing about hardware)")
    return 0


def cmd_ingest_ibm(args: argparse.Namespace) -> int:
    from .ingest.ibm import get_service, snapshot_fleet

    service = get_service(args.token, args.channel, args.instance)
    status = snapshot_fleet(service, args.root, args.backends or None)
    print(json.dumps(status, indent=2))
    return 1 if any(v.startswith("error") for v in status.values()) else 0


def cmd_build_curated(args: argparse.Namespace) -> int:
    print(json.dumps(build_curated(args.root), indent=2))
    return 0


def cmd_run_dev(args: argparse.Namespace) -> int:
    from .experiments import run_experiments

    cfg = _cfg(args)
    run_experiments(load_calibration(args.root), cfg, "dev", args.out, n_folds=args.folds)
    return 0


def cmd_run_final(args: argparse.Namespace) -> int:
    from .experiments import run_experiments

    cfg = _cfg(args)
    try:
        commit = check_protocol_frozen(args.protocol_tag, args.repo)
        acquire_final_lock(args.out, commit, args.force)
    except ProtocolError as exc:
        print(f"refusing to evaluate the test set: {exc}", file=sys.stderr)
        return 2
    run_experiments(load_calibration(args.root), cfg, "final", args.out, n_folds=args.folds)
    return 0


def cmd_e8(args: argparse.Namespace) -> int:
    from .experiments import run_e8

    print(run_e8(_cfg(args), args.out, seeds=args.seeds).round(3).to_string(index=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .serve import train_bundle

    cfg = _cfg(args)
    bundle = train_bundle(load_calibration(args.root), cfg)
    path = bundle.save(args.model)
    print(f"saved {bundle.model_id} (cfg {config_hash(cfg)}) to {path}")
    return 0


def cmd_nightly(args: argparse.Namespace) -> int:
    from .serve import Bundle, forecast_latest, write_forecast_batches

    cal = load_calibration(args.root)
    fc = forecast_latest(cal, Bundle.load(args.model))
    written = write_forecast_batches(fc, args.root)
    print(f"{len(fc):,} forecasts issued, {len(written)} new batch file(s)")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .serve import live_summary, load_forecasts, score_forecasts

    cal = load_calibration(args.root)
    sc = score_forecasts(load_forecasts(args.root), cal)
    out = Path(args.root) / "scores"
    out.mkdir(parents=True, exist_ok=True)
    sc.to_parquet(out / "scores.parquet", index=False)
    summary = live_summary(sc)
    summary.to_csv(out / "live_summary.csv", index=False)
    print(f"{len(sc):,} scored forecasts")
    if not summary.empty:
        print(summary.round(3).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qscout")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, config=True, holdout=False):
        sp.add_argument("--root", default="data", help="data root (default: data)")
        if config:
            sp.add_argument("--config", default="configs/default.yaml")
        if holdout:
            sp.add_argument("--holdout", nargs="*", help="backends held out of fitting (E6)")

    s = sub.add_parser("synth", help="write a synthetic curated dataset (plumbing/dev only)")
    common(s, config=False)
    s.add_argument("--devices", type=int, default=3)
    s.add_argument("--rows", type=int, default=3)
    s.add_argument("--cols", type=int, default=9)
    s.add_argument("--cycles", type=int, default=400)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("ingest-ibm", help="download current IBM calibrations into data/raw")
    common(s, config=False)
    s.add_argument("--backends", nargs="*")
    s.add_argument("--token")
    s.add_argument("--channel")
    s.add_argument("--instance")
    s.set_defaults(fn=cmd_ingest_ibm)

    s = sub.add_parser("build-curated", help="validate raw snapshots into data/curated")
    common(s, config=False)
    s.set_defaults(fn=cmd_build_curated)

    s = sub.add_parser("run-dev", help="experiments E1-E7 without touching the test period")
    common(s, holdout=True)
    s.add_argument("--out", default="results/dev")
    s.add_argument("--folds", type=int, default=3)
    s.set_defaults(fn=cmd_run_dev)

    s = sub.add_parser("run-final", help="one-shot test-set evaluation (needs a frozen protocol tag)")
    common(s, holdout=True)
    s.add_argument("--out", default="results/final")
    s.add_argument("--folds", type=int, default=3)
    s.add_argument("--protocol-tag", required=True)
    s.add_argument("--repo", default=".")
    s.add_argument("--force", action="store_true", help="re-run even though a lock exists")
    s.set_defaults(fn=cmd_run_final)

    s = sub.add_parser("e8", help="detectors vs synthetic ground truth")
    common(s, config=True)
    s.add_argument("--out", default="results/e8")
    s.add_argument("--seeds", type=int, default=5)
    s.set_defaults(fn=cmd_e8)

    s = sub.add_parser("train", help="fit the production bundle")
    common(s)
    s.add_argument("--model", default="models/bundle.joblib")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("nightly", help="issue forecasts from the latest calibration (append-only)")
    common(s, config=False)
    s.add_argument("--model", default="models/bundle.joblib")
    s.set_defaults(fn=cmd_nightly)

    s = sub.add_parser("score", help="score logged forecasts against realized values")
    common(s, config=False)
    s.set_defaults(fn=cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

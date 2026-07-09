"""Command-line interface for the feature-extraction pipeline.

    affectlens inventory --clips DIR
    affectlens extract   --clips DIR --out DIR [--ratings PATH]
    affectlens baseline  --clips DIR --ratings PATH
    affectlens encode    --features FEATURES.csv --signal SIGNAL.csv
    affectlens selftest

(Equivalently: ``python -m affectlens.cli <command>``.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import clips as clips_mod
from . import encoding
from . import pipeline
from .config import DEFAULT_RATING_INTERVAL_S, ExtractionConfig


def _cmd_inventory(args) -> int:
    print(clips_mod.inventory_to_json(args.clips))
    return 0


def _cmd_extract(args) -> int:
    config = ExtractionConfig(
        visual=not args.no_visual, audio=not args.no_audio, rating_interval_s=args.interval
    )
    if args.ratings:
        per_clip, _ = pipeline.run(
            args.clips, args.ratings, config=config, use_semantic=not args.no_semantic
        )
    else:
        # No ratings: bin each clip on its own duration-derived grid.
        per_clip = pipeline.extract_all(
            args.clips, config=config, use_semantic=not args.no_semantic
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cf in per_clip:
        cf.X.to_csv(out / f"{_safe(cf.clip)}__features.csv")
        if cf.Y is not None:
            cf.Y.to_csv(out / f"{_safe(cf.clip)}__ratings.csv")
    print(f"wrote features for {len(per_clip)} clip(s) to {out}")
    return 0


def _cmd_baseline(args) -> int:
    config = ExtractionConfig(
        visual=not args.no_visual, audio=not args.no_audio, rating_interval_s=args.interval
    )
    per_clip, result = pipeline.run(
        args.clips, args.ratings, config=config, use_semantic=not args.no_semantic
    )
    print(f"clips matched:     {len(per_clip)}")
    print(f"bins (pooled):     {result.n_bins}")
    print(f"features in:       {result.n_features_in}")
    print(f"mean Pearson r:    {result.mean_r:.3f}")
    print()
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(result.to_frame().to_string(index=False))
    return 0


def _cmd_encode(args) -> int:
    # Features: a CSV written by `extract` (first column is the bin start time).
    X = pd.read_csv(args.features, index_col=0)
    # Signal: a CSV with a time column and a value column.
    sig = pd.read_csv(args.signal)
    tcol = args.signal_time_col or sig.columns[0]
    vcol = args.signal_value_col or sig.columns[1]
    signal = encoding.bin_signal(
        sig[tcol].to_numpy(), sig[vcol].to_numpy(), X.index.to_numpy(), args.interval
    )
    lags = [int(x) for x in args.lags.split(",")] if args.lags else [0]
    corr = encoding.correlate_signal(X, signal, lag_bins=lags)
    enc = encoding.encode_signal(X, signal, lag_bins=args.lag)

    print(f"bins:              {len(X)}")
    print(f"features:          {X.shape[1]}")
    print(f"encoding model r:  {enc.r:.3f}  (r2={enc.r2:.3f}, lag={enc.lag_bins} bins)")
    print("\ntop features driving the signal (|weight|):")
    for name, w in enc.weights[:10]:
        print(f"  {w:+.3f}  {name}")
    print("\nper-feature correlation (best lag):")
    with pd.option_context("display.max_rows", 15, "display.width", 120):
        print(corr.head(15).to_string(index=False))
    return 0


def _cmd_selftest(args) -> int:
    from .synthetic import build_synthetic_dataset

    with build_synthetic_dataset() as (clips_dir, ratings_path):
        _, result = pipeline.run(clips_dir, ratings_path)
    print("self-test pipeline ran end-to-end.")
    print(f"mean Pearson r on synthetic data: {result.mean_r:.3f}")
    print(result.to_frame().to_string(index=False))
    ok = result.mean_r == result.mean_r  # not NaN
    return 0 if ok else 1


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="affectlens",
        description=(
            "Extract time-varying visual, audio, and semantic features from video, "
            "audio, and music clips, and relate them to human ratings or a recorded "
            "signal. Every command below has its own --help."
        ),
        epilog=(
            "examples:\n"
            "  affectlens inventory --clips data/clips\n"
            "  affectlens extract   --clips data/clips --out out/ --interval 1.0\n"
            "  affectlens baseline  --clips data/clips --ratings data/ratings.csv\n"
            "  affectlens encode    --features out/clip__features.csv --signal sig.csv --lags 0,1,2\n"
            "  affectlens selftest\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    pi = sub.add_parser(
        "inventory",
        help="probe clip durations, resolution, and audio/video streams",
        description="Print a JSON report for every clip in a directory: duration, "
        "resolution, frame rate, and whether it carries audio and/or video.",
    )
    pi.add_argument("--clips", required=True, metavar="DIR",
                    help="directory of video/audio clips to probe")
    pi.set_defaults(func=_cmd_inventory)

    for name, func, needs_out in (("extract", _cmd_extract, True), ("baseline", _cmd_baseline, False)):
        if needs_out:
            sp = sub.add_parser(
                name,
                help="extract per-clip feature time courses to CSV",
                description="Extract aligned feature time courses for each clip and "
                "write them to CSV -- one row per time bin, one column per feature.",
            )
        else:
            sp = sub.add_parser(
                name,
                help="score how well features predict human ratings (cross-validated)",
                description="Fit a leave-one-clip-out cross-validated Ridge model that "
                "predicts each rated dimension from the clip features, and print the "
                "per-dimension held-out Pearson r / R2.",
            )
        sp.add_argument("--clips", required=True, metavar="DIR",
                        help="directory of video/audio clips")
        # Ratings are what `baseline` scores against; `extract` can run without
        # them on a duration-derived grid (the `encode` workflow).
        sp.add_argument(
            "--ratings", required=not needs_out, metavar="PATH",
            help="ratings CSV/Excel to score against"
            + ("; optional -- without it, features are binned on a duration-derived grid"
               if needs_out else ""),
        )
        if needs_out:
            sp.add_argument("--out", required=True, metavar="DIR",
                            help="output directory for the per-clip feature CSVs")
        sp.add_argument("--interval", type=float, default=DEFAULT_RATING_INTERVAL_S, metavar="SEC",
                        help="feature time-bin width in seconds "
                        "(default: %(default)s; e.g. 1.0 for finer temporal resolution)")
        sp.add_argument("--no-visual", action="store_true",
                        help="skip visual features (treat clips as audio-only)")
        sp.add_argument("--no-audio", action="store_true",
                        help="skip audio features (treat clips as silent video)")
        sp.add_argument("--no-semantic", action="store_true",
                        help="skip semantic/dialogue features")
        sp.set_defaults(func=func)

    pe = sub.add_parser(
        "encode",
        help="relate extracted features to an external signal, with lag search",
        description="Correlate a clip's feature time courses against a separately "
        "recorded signal (e.g. a physiological or brain channel), scanning lags, and "
        "fit a cross-validated ridge encoding model that reports held-out r and the "
        "features it leans on.",
    )
    pe.add_argument("--features", required=True, metavar="CSV",
                    help="a features CSV written by `extract`")
    pe.add_argument("--signal", required=True, metavar="CSV",
                    help="CSV with a time column and a value column")
    pe.add_argument("--signal-time-col", default=None, metavar="COL",
                    help="name of the signal's time column (default: first column)")
    pe.add_argument("--signal-value-col", default=None, metavar="COL",
                    help="name of the signal's value column (default: second column)")
    pe.add_argument("--interval", type=float, default=DEFAULT_RATING_INTERVAL_S, metavar="SEC",
                    help="feature bin width in seconds; must match how the features "
                    "were extracted (default: %(default)s)")
    pe.add_argument("--lag", type=int, default=0, metavar="BINS",
                    help="lag (in bins) for the encoding model (default: %(default)s)")
    pe.add_argument("--lags", default="0", metavar="L,L,...",
                    help="comma-separated lags (in bins) to scan for the per-feature correlation")
    pe.set_defaults(func=_cmd_encode)

    ps = sub.add_parser(
        "selftest",
        help="run the whole pipeline on generated synthetic data",
        description="Generate synthetic clips + ratings and run extract -> align -> "
        "baseline -> encode end-to-end, as a self-contained smoke test (no data needed).",
    )
    ps.set_defaults(func=_cmd_selftest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end demo on synthetic data — no external files needed.

Run:  python examples/demo.py

Generates a few synthetic clips + ratings, extracts features, reproduces the
ratings with the cross-validated baseline, then fabricates a "recorded signal"
that lags the clip's motion and shows the encoding model recovering it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from affectlens import encoding, pipeline  # noqa: E402
from affectlens.synthetic import build_synthetic_dataset  # noqa: E402


def main() -> None:
    with build_synthetic_dataset(n_clips=3) as (clips_dir, ratings_csv):
        print("== Reproducing human ratings from clip content ==")
        per_clip, result = pipeline.run(clips_dir, ratings_csv)
        print(f"clips matched: {len(per_clip)}   mean Pearson r: {result.mean_r:.3f}\n")
        print(result.to_frame().to_string(index=False))

        print("\n== Explaining a recorded signal from features ==")
        # Pool bins across clips (a real recording has many time points; a single
        # 20 s synthetic clip has only a handful).
        import pandas as pd

        cols = sorted(set.intersection(*(set(cf.X.columns) for cf in per_clip)))
        X = pd.concat([cf.X[cols] for cf in per_clip], ignore_index=True)
        # Fabricate a signal that follows motion by ~one bin, + noise.
        motion_cols = [c for c in X.columns if "motion" in c]
        rng = np.random.default_rng(0)
        base = X[motion_cols].mean(axis=1).to_numpy() if motion_cols else rng.random(len(X))
        signal = np.empty(len(X))
        signal[1:] = base[:-1]
        signal[0] = np.nan
        signal += rng.normal(0, 0.02, len(X))

        enc = encoding.encode_signal(X, signal, lag_bins=1)
        print(f"encoding model held-out r: {enc.r:.3f}  (lag {enc.lag_bins} bin)")
        print("top features the model leans on:")
        for name, w in enc.weights[:5]:
            print(f"  {w:+.3f}  {name}")


if __name__ == "__main__":
    main()

"""End-to-end orchestration: clips + ratings -> aligned features -> baseline.

This wires the modules into the end-to-end flow:

    clips ─┬─ low-level (visual + audio)   ─┐
           └─ high-level (semantic dialogue) ┼─ align to rating grid ─ X
    ratings ── normalize ── consensus ───────┴───────────────────────── Y
                                                          └─ baseline score
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import align, baseline, clips as clips_mod, highlevel, lowlevel, ratings as ratings_mod
from .config import ExtractionConfig
from .ratings import RatingSchema


@dataclass
class ClipFeatures:
    clip: str
    X: pd.DataFrame  # bins x features, indexed by t_start
    Y: pd.DataFrame | None = None  # bins x rated features, indexed by t_start


def extract_clip_features(
    clip_path: str | Path,
    rating_times: np.ndarray,
    config: ExtractionConfig | None = None,
    use_semantic: bool = True,
) -> pd.DataFrame:
    """Extract and align all feature families for one clip onto its rating grid."""
    config = config or ExtractionConfig()
    streams = lowlevel.extract_lowlevel(clip_path, config)
    if use_semantic:
        sem = highlevel.semantic_stream(clip_path)
        if not sem.empty:
            streams["semantic"] = sem
    return align.build_design_matrix(streams, rating_times, config)


def extract_clip_auto(
    clip_path: str | Path,
    config: ExtractionConfig | None = None,
    use_semantic: bool = True,
) -> pd.DataFrame:
    """Extract one clip's features on a duration-derived grid, with no ratings.

    Grid spacing is ``config.rating_interval_s``; the clip length is read from
    the *decoded* stream extent, not the container header -- some files (e.g.
    certain AVIs) report a bogus multi-hour header duration that would blow up
    the bin grid. Returns the bins x features matrix, or an empty frame if the
    clip has no readable streams. Used by both ``extract_all`` and the web UI so
    every no-ratings path grids clips identically.
    """
    config = config or ExtractionConfig()
    streams = lowlevel.extract_lowlevel(clip_path, config)
    if use_semantic:
        sem = highlevel.semantic_stream(clip_path)
        if not sem.empty:
            streams["semantic"] = sem
    dur = max((float(s["t"].iloc[-1]) for s in streams.values()
               if s is not None and len(s)), default=0.0)
    if dur <= 0:
        return pd.DataFrame()
    n_bins = max(1, int(round(dur / config.rating_interval_s)))
    times = np.arange(n_bins) * config.rating_interval_s
    return align.build_design_matrix(streams, times, config)


def extract_all(
    clips_dir: str | Path,
    config: ExtractionConfig | None = None,
    use_semantic: bool = True,
) -> list[ClipFeatures]:
    """Extract features for every clip in a directory, with no ratings.

    Use this when the goal is relating features to an external signal
    (``encode``) rather than to ratings. Each clip is gridded by
    :func:`extract_clip_auto`.
    """
    config = config or ExtractionConfig()
    per_clip: list[ClipFeatures] = []
    for info in clips_mod.inventory(Path(clips_dir)):
        if info.error is not None:
            continue
        X = extract_clip_auto(info.path, config, use_semantic)
        if X.empty:
            continue
        per_clip.append(ClipFeatures(clip=Path(info.path).stem, X=X))
    if not per_clip:
        raise RuntimeError(f"no readable clips found in {clips_dir}")
    return per_clip


def run(
    clips_dir: str | Path,
    ratings_path: str | Path,
    schema: RatingSchema | None = None,
    config: ExtractionConfig | None = None,
    use_semantic: bool = True,
) -> tuple[list[ClipFeatures], baseline.BaselineResult]:
    """Full pipeline over a directory of clips + a rating source.

    Clips are matched to ratings by the ``clip`` field in the normalized ratings
    (matched against the clip filename stem, case-insensitive). Returns per-clip
    aligned features and a pooled cross-clip baseline result.
    """
    config = config or ExtractionConfig()
    long = ratings_mod.load_ratings(ratings_path, schema)
    cons = ratings_mod.consensus(long)

    clips_dir = Path(clips_dir)
    inv = clips_mod.inventory(clips_dir)
    clip_paths = {Path(c.path).stem.lower(): c.path for c in inv if c.error is None}

    rated_clips = list(cons["clip"].unique())
    per_clip: list[ClipFeatures] = []

    for clip_name in rated_clips:
        key = str(clip_name).lower()
        path = clip_paths.get(key)
        if path is None:
            # Try a looser contains-match (rating labels may differ from filenames).
            match = next((p for stem, p in clip_paths.items() if key in stem or stem in key), None)
            path = match
        if path is None:
            continue

        Y = ratings_mod.to_target_matrix(cons, clip=clip_name)
        X = extract_clip_features(path, Y.index.to_numpy(), config, use_semantic)
        Xa, Ya = align.align_xy(X, Y)
        if len(Xa) == 0:
            continue
        per_clip.append(ClipFeatures(clip=str(clip_name), X=Xa, Y=Ya))

    if not per_clip:
        raise RuntimeError(
            "no clips could be matched to ratings; check that clip filenames "
            "correspond to the 'clip' labels in the rating files"
        )

    # Pool across clips for a leave-one-clip-out baseline. Only shared feature
    # columns are kept so the matrices are consistent.
    common_cols = set.intersection(*(set(cf.X.columns) for cf in per_clip))
    common_cols = sorted(common_cols)
    X_all = pd.concat([cf.X[common_cols] for cf in per_clip], ignore_index=True)
    Y_all = pd.concat([cf.Y for cf in per_clip], ignore_index=True)
    groups = np.concatenate([[i] * len(cf.X) for i, cf in enumerate(per_clip)])

    result = baseline.score_baseline(X_all, Y_all, groups=groups)
    return per_clip, result

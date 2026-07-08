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


def extract_all(
    clips_dir: str | Path,
    config: ExtractionConfig | None = None,
    use_semantic: bool = True,
) -> list[ClipFeatures]:
    """Extract features for every clip in a directory, with no ratings.

    The time grid is derived from each clip's *decoded* extent at
    ``config.rating_interval_s`` spacing. Use this when the goal is relating
    features to an external signal (``encode``) rather than to ratings.

    We intentionally do not trust the container-header duration here: some files
    (e.g. certain AVIs) report a bogus multi-hour duration from a malformed
    header, which would blow up the bin grid. Extracting the streams first and
    reading their last timestamp gives the true clip length.
    """
    config = config or ExtractionConfig()
    inv = clips_mod.inventory(Path(clips_dir))
    interval = config.rating_interval_s
    per_clip: list[ClipFeatures] = []
    for info in inv:
        if info.error is not None:
            continue
        streams = lowlevel.extract_lowlevel(info.path, config)
        if use_semantic:
            sem = highlevel.semantic_stream(info.path)
            if not sem.empty:
                streams["semantic"] = sem
        # True extent from the decoded streams, not the (possibly bogus) header.
        dur = max((float(s["t"].iloc[-1]) for s in streams.values()
                   if s is not None and len(s)), default=0.0)
        if dur <= 0:
            continue
        n_bins = max(1, int(round(dur / interval)))
        times = np.arange(n_bins) * interval
        X = align.build_design_matrix(streams, times, config)
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

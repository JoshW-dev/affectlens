"""Align feature time series onto the behavioral rating grid.

The rating grid (bins of a few seconds) is the shared time base. Each low-level
feature stream is sampled on its own faster clock; here we aggregate every
stream within each rating bin so the design matrix X and the target Y line up
row-for-row. Aggregating with several statistics (mean/std/max) preserves
within-bin dynamics -- e.g. a mid-bin motion or loudness spike that a bare mean
would wash out. This is what lets a coarse rating grid still carry sharp events
(surprise, energy spikes): a bin's ``*_max`` and ``*_std`` retain the spike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExtractionConfig


def bin_edges_from_ratings(rating_times: np.ndarray, interval_s: float) -> np.ndarray:
    """Build bin edges from the sorted rating onset times.

    Uses successive onsets as edges; the final bin is closed with ``interval_s``.
    """
    times = np.asarray(sorted(set(float(t) for t in rating_times)), dtype=float)
    if times.size == 0:
        return np.zeros(0)
    edges = np.append(times, times[-1] + interval_s)
    return edges


def aggregate_to_bins(
    stream: pd.DataFrame,
    edges: np.ndarray,
    aggregations: tuple[str, ...],
    prefix: str = "",
) -> pd.DataFrame:
    """Aggregate a per-sample feature stream into rating bins.

    ``stream`` must have a ``t`` column plus one column per feature. Returns a
    frame indexed by bin start time (``edges[:-1]``) with ``<prefix><feat>_<agg>``
    columns.
    """
    if stream.empty or edges.size < 2:
        return pd.DataFrame(index=pd.Index(edges[:-1] if edges.size else [], name="t_start"))

    feats = [c for c in stream.columns if c != "t"]
    bin_idx = np.digitize(stream["t"].to_numpy(), edges) - 1
    n_bins = len(edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    work = stream.loc[valid, feats].copy()
    work["__bin"] = bin_idx[valid]

    grouped = work.groupby("__bin")
    out = {}
    for agg in aggregations:
        agged = grouped.agg(agg)
        for f in feats:
            out[f"{prefix}{f}_{agg}"] = agged[f]
    result = pd.DataFrame(out)
    result = result.reindex(range(n_bins))
    result.index = pd.Index(edges[:-1], name="t_start")
    return result


def build_design_matrix(
    streams: dict[str, pd.DataFrame],
    rating_times: np.ndarray,
    config: ExtractionConfig | None = None,
) -> pd.DataFrame:
    """Combine all feature streams into one bins x features design matrix.

    ``streams`` maps a family name (e.g. 'visual', 'audio', 'semantic') to a
    per-sample frame with a ``t`` column. The result is indexed by bin start time
    aligned to ``rating_times``.
    """
    config = config or ExtractionConfig()
    edges = bin_edges_from_ratings(rating_times, config.rating_interval_s)
    parts = [
        aggregate_to_bins(s, edges, config.bin_aggregations, prefix=f"{name}__")
        for name, s in streams.items()
        if s is not None and not s.empty
    ]
    if not parts:
        return pd.DataFrame(index=pd.Index(edges[:-1] if edges.size else [], name="t_start"))
    X = pd.concat(parts, axis=1)
    # Bins with no samples (e.g. trailing silence) -> forward/zero fill rather
    # than drop, so X and Y keep the same rows.
    X = X.sort_index()
    return X


def align_xy(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join a design matrix and target matrix on their bin index.

    Both are indexed by ``t_start``. Returns (X, Y) restricted to shared bins,
    with all-NaN feature columns dropped and remaining NaNs mean-imputed.
    """
    common = X.index.intersection(Y.index)
    Xa = X.loc[common].copy()
    Ya = Y.loc[common].copy()
    Xa = Xa.dropna(axis=1, how="all")
    Xa = Xa.fillna(Xa.mean(numeric_only=True)).fillna(0.0)
    return Xa, Ya

"""Correlate feature time courses against an external signal.

The complement to ``baseline.py``. Where the baseline predicts *human ratings*
from features, this module relates the features to a separately recorded
continuous signal -- any time course you want to explain with the stimulus, such
as a physiological measure or a neuroimaging channel (an EEG band envelope, an
fMRI ROI or vertex time series, pupil size, heart rate...). Two views:

  correlate_signal -- per-feature Pearson r with the signal, optionally scanning a
                      set of lags. A response often follows the stimulus feature
                      by a fixed delay (e.g. the fMRI hemodynamic response peaks
                      several seconds after the event); scanning lags finds it.
  encode_signal    -- a cross-validated ridge *encoding model* predicting the
                      signal from all features jointly, reporting held-out r and
                      the per-feature weights (which features the model leans on;
                      an importance ranking, not a clean causal attribution).

The signal is resampled onto the same bins as the feature matrix X, so both
share one time base. Lags are expressed in bins; at a 4.5 s bin size, a lag of 1
bin is ~4.5 s -- a reasonable first guess for an fMRI hemodynamic delay.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import align
from .config import DEFAULT_RATING_INTERVAL_S


def bin_signal(
    signal_times: np.ndarray,
    signal_values: np.ndarray,
    bin_starts: np.ndarray,
    interval_s: float = DEFAULT_RATING_INTERVAL_S,
) -> np.ndarray:
    """Average an arbitrarily-sampled signal onto the feature bins.

    ``bin_starts`` are the feature matrix's bin start times (its index). Returns
    one value per bin (NaN where no signal sample falls in the bin).
    """
    bin_starts = np.asarray(bin_starts, dtype=float)
    if bin_starts.size == 0:
        return np.zeros(0)
    edges = np.append(bin_starts, bin_starts[-1] + interval_s)
    stream = pd.DataFrame({"t": np.asarray(signal_times, float), "signal": np.asarray(signal_values, float)})
    binned = align.aggregate_to_bins(stream, edges, ("mean",))
    return binned["signal_mean"].to_numpy()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _apply_lag(x: np.ndarray, signal: np.ndarray, lag_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift so feature at bin t is matched to signal at bin t+lag."""
    if lag_bins == 0:
        return x, signal
    if lag_bins > 0:
        return x[:-lag_bins], signal[lag_bins:]
    return x[-lag_bins:], signal[:lag_bins]


def correlate_signal(
    X: pd.DataFrame,
    signal: np.ndarray,
    lag_bins: list[int] | tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """Per-feature Pearson correlation with the signal, best over the given lags.

    Returns a DataFrame: feature, best_lag, r (signed r at the lag of max |r|),
    sorted by descending |r|.
    """
    X = X.select_dtypes(include=[np.number])
    signal = np.asarray(signal, dtype=float)
    rows = []
    for feat in X.columns:
        fvals = X[feat].to_numpy(dtype=float)
        best_r, best_lag = float("nan"), 0
        for lag in lag_bins:
            fx, sy = _apply_lag(fvals, signal, lag)
            mask = ~np.isnan(fx) & ~np.isnan(sy)
            if mask.sum() < 3:
                continue
            r = _pearson(fx[mask], sy[mask])
            if np.isnan(best_r) or (not np.isnan(r) and abs(r) > abs(best_r)):
                best_r, best_lag = r, lag
        rows.append({"feature": feat, "best_lag": best_lag, "r": best_r})
    out = pd.DataFrame(rows)
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


@dataclass
class EncodingResult:
    r: float  # held-out Pearson r (cross-validated)
    r2: float
    n: int
    lag_bins: int
    weights: list[tuple[str, float]] = field(default_factory=list)


def encode_signal(
    X: pd.DataFrame,
    signal: np.ndarray,
    lag_bins: int = 0,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    n_splits: int = 5,
    shuffle: bool = False,
) -> EncodingResult:
    """Cross-validated ridge encoding model: predict the signal from all features.

    Reports held-out Pearson r / R2 and the standardized per-feature weights
    (largest |weight| first) -- i.e. which features the model leans on (read as
    an importance ranking, not a clean causal attribution: ridge spreads weight
    across correlated features).

    Cross-validation uses **contiguous** folds by default (``shuffle=False``).
    Recorded signals and their stimulus features are both autocorrelated in
    time, so shuffled folds would place a test bin's temporal neighbours in the
    training set and leak, inflating the held-out score. Contiguous folds are
    the honest "predict an unseen stretch" test. Pass ``shuffle=True`` only if
    your bins are genuinely exchangeable.

    Any bin with a NaN in *any* feature is dropped (not imputed) on this path, so
    sparse features -- e.g. ``semantic__*`` columns, which are NaN in every
    dialogue-free bin -- can drop most rows and yield an all-NaN result; a
    warning is emitted when that happens.
    """
    X = X.select_dtypes(include=[np.number])
    signal = np.asarray(signal, dtype=float)
    fx = X.to_numpy(dtype=float)
    fx, sy = _apply_lag(fx, signal, lag_bins)

    row_ok = ~np.isnan(sy) & ~np.isnan(fx).any(axis=1)
    n_dropped = int((~row_ok).sum())
    if n_dropped > 0.5 * len(sy) and len(sy):
        warnings.warn(
            f"encode_signal dropped {n_dropped}/{len(sy)} bins with a NaN feature or "
            "signal value (sparse features such as semantic columns are NaN in "
            "event-free bins and are not imputed here).",
            stacklevel=2,
        )
    fx, sy = fx[row_ok], sy[row_ok]
    n = len(sy)
    if n < max(6, n_splits):
        return EncodingResult(float("nan"), float("nan"), n, lag_bins)

    preds = np.full(n, np.nan)
    splitter = KFold(n_splits=min(n_splits, n), shuffle=shuffle, random_state=0 if shuffle else None)
    for tr, te in splitter.split(fx):
        scaler = StandardScaler()
        model = RidgeCV(alphas=alphas)
        model.fit(scaler.fit_transform(fx[tr]), sy[tr])
        preds[te] = model.predict(scaler.transform(fx[te]))

    valid = ~np.isnan(preds)
    r = _pearson(preds[valid], sy[valid])
    ss_res = float(np.sum((sy[valid] - preds[valid]) ** 2))
    ss_tot = float(np.sum((sy[valid] - sy[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    scaler = StandardScaler()
    model = RidgeCV(alphas=alphas)
    model.fit(scaler.fit_transform(fx), sy)
    order = np.argsort(np.abs(model.coef_))[::-1]
    weights = [(X.columns[i], float(model.coef_[i])) for i in order]

    return EncodingResult(r=r, r2=r2, n=n, lag_bins=lag_bins, weights=weights)

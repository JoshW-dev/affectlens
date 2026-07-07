"""Baseline model: reproduce human ratings from extracted features.

The core proof point: clips in, ratings out, scored against the human labels as
ground truth. We fit a regularized linear model (RidgeCV) per rated feature and
report cross-validated agreement.

Scoring is cross-validated so we measure generalization, not fit. When the data
spans multiple clips we use leave-one-clip-out folds -- the honest test of
"predict a clip the model has not seen". With a single clip we fall back to
K-fold over bins.

Deliberately a linear baseline: it is a clear reference, it is interpretable
(per-feature coefficients say which regressors drive each rating), and it is the
number a richer model must beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureScore:
    feature: str
    pearson_r: float
    r2: float
    n: int
    top_predictors: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class BaselineResult:
    scores: list[FeatureScore]
    n_bins: int
    n_features_in: int

    @property
    def mean_r(self) -> float:
        vals = [s.pearson_r for s in self.scores if not np.isnan(s.pearson_r)]
        return float(np.mean(vals)) if vals else float("nan")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"rated_feature": s.feature, "pearson_r": s.pearson_r, "r2": s.r2, "n": s.n}
                for s in self.scores
            ]
        ).sort_values("pearson_r", ascending=False, ignore_index=True)


def _cv_splits(n: int, groups: np.ndarray | None):
    if groups is not None and len(np.unique(groups)) >= 2:
        n_splits = min(len(np.unique(groups)), 5)
        return GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups), groups
    n_splits = min(5, n) if n >= 2 else 2
    if n_splits < 2:
        return None, None
    return KFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.arange(n)), None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_baseline(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: np.ndarray | None = None,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
) -> BaselineResult:
    """Cross-validated ridge regression from features X to each target in Y.

    ``groups`` (e.g. clip id per row) triggers leave-one-group-out CV.
    """
    X = X.select_dtypes(include=[np.number])
    n = len(X)
    scores: list[FeatureScore] = []

    split_iter, used_groups = _cv_splits(n, groups)
    splits = list(split_iter) if split_iter is not None else []

    for feat in Y.columns:
        y = pd.to_numeric(Y[feat], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 3 or not splits:
            scores.append(FeatureScore(feat, float("nan"), float("nan"), int(mask.sum())))
            continue

        preds = np.full(n, np.nan)
        for train_idx, test_idx in splits:
            tr = [i for i in train_idx if mask[i]]
            te = [i for i in test_idx if mask[i]]
            if len(tr) < 2 or not te:
                continue
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X.iloc[tr])
            Xte = scaler.transform(X.iloc[te])
            model = RidgeCV(alphas=alphas)
            model.fit(Xtr, y[tr])
            preds[te] = model.predict(Xte)

        valid = mask & ~np.isnan(preds)
        if valid.sum() < 3:
            scores.append(FeatureScore(feat, float("nan"), float("nan"), int(valid.sum())))
            continue

        r = _pearson(preds[valid], y[valid])
        ss_res = float(np.sum((y[valid] - preds[valid]) ** 2))
        ss_tot = float(np.sum((y[valid] - y[valid].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        # Interpretability: refit on all data to read coefficient magnitudes.
        scaler = StandardScaler()
        model = RidgeCV(alphas=alphas)
        model.fit(scaler.fit_transform(X.iloc[np.where(mask)[0]]), y[mask])
        order = np.argsort(np.abs(model.coef_))[::-1][:5]
        top = [(X.columns[i], float(model.coef_[i])) for i in order]

        scores.append(FeatureScore(feat, r, r2, int(valid.sum()), top))

    return BaselineResult(scores=scores, n_bins=n, n_features_in=X.shape[1])

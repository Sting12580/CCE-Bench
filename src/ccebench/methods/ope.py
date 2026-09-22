"""Readable MIPS and OffCEM-style estimators on precomputed features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_X_y


@dataclass(frozen=True)
class DensityRatioModel:
    classifier: LogisticRegression
    source_to_target_prior_ratio: float
    clip: float

    def weights(self, source_features: ArrayLike) -> np.ndarray:
        x = check_array(source_features, dtype=float)
        p_target = self.classifier.predict_proba(x)[:, 1]
        eps = np.finfo(float).eps
        odds = p_target / np.clip(1.0 - p_target, eps, None)
        return np.clip(odds * self.source_to_target_prior_ratio, 0.0, self.clip)


def fit_density_ratio(
    source_features: ArrayLike,
    target_features: ArrayLike,
    *,
    clip: float = 20.0,
    c: float = 1.0,
    random_state: int = 0,
) -> DensityRatioModel:
    """Fit ``p_target(x) / p_source(x)`` with probabilistic classification."""

    source = check_array(source_features, dtype=float)
    target = check_array(target_features, dtype=float)
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and Target features must have equal width")
    if clip <= 0:
        raise ValueError("clip must be positive")
    x = np.vstack([source, target])
    domain = np.r_[np.zeros(len(source)), np.ones(len(target))]
    classifier = LogisticRegression(C=c, max_iter=2000, random_state=random_state)
    classifier.fit(x, domain)
    # Classifier odds include the empirical class prior. Remove it to recover
    # the density ratio when Source and Target pools have different sizes.
    prior_ratio = len(source) / len(target)
    return DensityRatioModel(classifier, prior_ratio, clip)


def effective_sample_size(weights: ArrayLike) -> float:
    """Return ESS as a fraction of the original sample size."""

    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0 or np.any(w < 0):
        raise ValueError("weights must be a non-empty non-negative vector")
    denominator = len(w) * np.square(w).sum()
    return float(np.square(w.sum()) / denominator) if denominator > 0 else 0.0


def mips(source_scores: ArrayLike, weights: ArrayLike, *, self_normalized: bool = True) -> float:
    """Estimate a Target mean by importance weighting Source scores."""

    y = np.asarray(source_scores, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if y.shape != w.shape or y.size == 0:
        raise ValueError("scores and weights must be non-empty vectors with equal shape")
    if self_normalized:
        if w.sum() <= 0:
            raise ValueError("self-normalized MIPS requires a positive weight sum")
        return float(np.dot(w, y) / w.sum())
    return float(np.mean(w * y))


@dataclass(frozen=True)
class OffCEMResult:
    value: float
    dm_term: float
    residual_correction: float
    ess_fraction: float


def offcem_style(
    source_features: ArrayLike,
    source_scores: ArrayLike,
    target_features: ArrayLike,
    *,
    outcome_model=None,
    weight_clip: float = 20.0,
    random_state: int = 0,
) -> OffCEMResult:
    """Combine a DM Target mean with a weighted Source residual correction.

    This is a compact project-facing analogue, not a claim of exact reproduction
    of every assumption and component in the published OffCEM algorithm.
    """

    source, scores = check_X_y(
        source_features, source_scores, dtype=float, y_numeric=True
    )
    target = check_array(target_features, dtype=float)
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and Target features must have equal width")
    base = outcome_model
    if base is None:
        base = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )
    model = clone(base).fit(source, scores)
    dm_term = float(model.predict(target).mean())
    ratio = fit_density_ratio(
        source, target, clip=weight_clip, random_state=random_state
    )
    weights = ratio.weights(source)
    residual = scores - model.predict(source)
    correction = mips(residual, weights, self_normalized=True)
    return OffCEMResult(
        value=dm_term + correction,
        dm_term=dm_term,
        residual_correction=correction,
        ess_fraction=effective_sample_size(weights),
    )

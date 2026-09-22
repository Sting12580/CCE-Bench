"""Context-aware residual bridge on precomputed support features."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


def _allow_nan_array(values: ArrayLike) -> np.ndarray:
    try:
        return check_array(values, dtype=float, ensure_all_finite="allow-nan")
    except TypeError:  # scikit-learn < 1.6
        return check_array(values, dtype=float, force_all_finite="allow-nan")


class ContextResidualBridge(BaseEstimator):
    """Predict a correction to a frozen Direct Method prediction."""

    def __init__(
        self,
        model=None,
        *,
        correction_scale: float = 1.0,
        clip: tuple[float, float] | None = (0.0, 1.0),
    ):
        self.model = model
        self.correction_scale = correction_scale
        self.clip = clip

    def fit(
        self,
        context_features: ArrayLike,
        scores: ArrayLike,
        base_predictions: ArrayLike,
    ) -> "ContextResidualBridge":
        x = _allow_nan_array(context_features)
        y = np.asarray(scores, dtype=float).reshape(-1)
        if len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("scores must be finite with one value per row")
        base = np.asarray(base_predictions, dtype=float).reshape(-1)
        if base.shape != y.shape:
            raise ValueError("base_predictions and scores must have equal shape")
        estimator = self.model
        if estimator is None:
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=10.0),
            )
        self.model_ = clone(estimator).fit(x, y - base)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, context_features: ArrayLike, base_predictions: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "model_")
        x = _allow_nan_array(context_features)
        base = np.asarray(base_predictions, dtype=float).reshape(-1)
        if len(base) != len(x):
            raise ValueError("base_predictions must have one value per row")
        correction = self.correction_scale * np.asarray(self.model_.predict(x), dtype=float)
        pred = base + correction
        if self.clip is not None:
            pred = np.clip(pred, *self.clip)
        return pred

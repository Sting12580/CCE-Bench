"""Direct Method on precomputed representations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class DirectMethod(BaseEstimator):
    """Fit an outcome model on Source rows and predict Target rows.

    The class deliberately receives numeric features. Text encoding and any
    Source-only learned representation are separate, inspectable steps.
    """

    def __init__(self, model=None, clip: tuple[float, float] | None = (0.0, 1.0)):
        self.model = model
        self.clip = clip

    def fit(self, features: ArrayLike, scores: ArrayLike) -> "DirectMethod":
        x, y = check_X_y(features, scores, dtype=float, y_numeric=True)
        base = self.model
        if base is None:
            base = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=0,
            )
        self.model_ = clone(base)
        self.model_.fit(x, y)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "model_")
        x = check_array(features, dtype=float)
        pred = np.asarray(self.model_.predict(x), dtype=float)
        if self.clip is not None:
            pred = np.clip(pred, *self.clip)
        return pred

    def estimate(self, features: ArrayLike) -> float:
        """Return the mean predicted Target score."""

        return float(self.predict(features).mean())

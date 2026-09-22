"""Interpretable non-negative weighting for multiple judge scores."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SimplexJudgeCombiner(BaseEstimator):
    """Learn judge weights constrained to be non-negative and sum to one."""

    def __init__(self, *, l2: float = 0.0, max_weight: float = 1.0):
        self.l2 = l2
        self.max_weight = max_weight

    def fit(self, judge_scores: ArrayLike, gold_scores: ArrayLike):
        x, y = check_X_y(judge_scores, gold_scores, dtype=float, y_numeric=True)
        n_judges = x.shape[1]
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        if not 1.0 / n_judges <= self.max_weight <= 1.0:
            raise ValueError("max_weight must be between 1/n_judges and 1")

        def objective(weights: np.ndarray) -> float:
            error = x @ weights - y
            return float(np.mean(error**2) + self.l2 * np.dot(weights, weights))

        result = minimize(
            objective,
            x0=np.full(n_judges, 1.0 / n_judges),
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n_judges,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"judge-weight optimization failed: {result.message}")
        weights = np.clip(result.x, 0.0, self.max_weight)
        self.weights_ = weights / weights.sum()
        self.n_features_in_ = n_judges
        self.optimization_message_ = result.message
        return self

    def predict(self, judge_scores: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "weights_")
        x = check_array(judge_scores, dtype=float)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("judge-score width differs from the fitted combiner")
        return x @ self.weights_

"""Frozen and Source-reward-informed representation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def pair_features(
    prompt_embeddings: ArrayLike,
    answer_embeddings: ArrayLike,
    *,
    include_absolute_difference: bool = False,
) -> np.ndarray:
    """Combine equal-width prompt and answer embeddings.

    The default follows the project's common ``[q, a, q * a]`` interaction
    construction. Encoders such as BERT or BGE remain upstream choices.
    """

    q = check_array(prompt_embeddings, dtype=float)
    a = check_array(answer_embeddings, dtype=float)
    if q.shape != a.shape:
        raise ValueError("prompt and answer embeddings must have the same shape")
    blocks = [q, a, q * a]
    if include_absolute_difference:
        blocks.append(np.abs(q - a))
    return np.hstack(blocks)


class RewardInformedProjection(TransformerMixin, BaseEstimator):
    """A compact supervised projection fitted from Source scores only.

    This readable public implementation uses the hidden layer of a one-layer
    MLP as the learned embedding. It illustrates the information boundary of
    the project's larger two-tower implementation; it is not artifact replay.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        *,
        alpha: float = 1e-2,
        max_iter: int = 500,
        random_state: int = 0,
    ):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, source_features: ArrayLike, source_scores: ArrayLike):
        x, y = check_X_y(source_features, source_scores, dtype=float, y_numeric=True)
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.scaler_ = StandardScaler().fit(x)
        scaled = self.scaler_.transform(x)
        self.network_ = MLPRegressor(
            hidden_layer_sizes=(self.latent_dim,),
            activation="relu",
            solver="lbfgs",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(scaled, y)
        return self

    def transform(self, features: ArrayLike) -> np.ndarray:
        check_is_fitted(self, ["scaler_", "network_"])
        x = check_array(features, dtype=float)
        scaled = self.scaler_.transform(x)
        hidden = scaled @ self.network_.coefs_[0] + self.network_.intercepts_[0]
        return np.maximum(hidden, 0.0)

    def fit_transform(self, source_features: ArrayLike, source_scores: ArrayLike) -> np.ndarray:
        return self.fit(source_features, source_scores).transform(source_features)

"""Minimal DANN regressor for already-computed text embeddings.

PyTorch is optional. This module keeps the encoder/backbone choice separate
from the domain-adaptation objective: BERT/BGE features enter as numeric arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without the optional extra
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError('DANN requires PyTorch; install with pip install -e ".[domain]"')


if torch is not None:

    class _GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, strength):
            ctx.strength = strength
            return values.view_as(values)

        @staticmethod
        def backward(ctx, gradient):
            return -ctx.strength * gradient, None


    class DANNRegressor(nn.Module):
        """Shared representation with reward and adversarial domain heads."""

        def __init__(self, input_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.reward_head = nn.Linear(hidden_dim, 1)
            self.domain_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, features, adversarial_strength: float = 0.0):
            representation = self.encoder(features)
            reward = self.reward_head(representation).squeeze(-1)
            reversed_representation = _GradientReverse.apply(
                representation, adversarial_strength
            )
            domain_logit = self.domain_head(reversed_representation).squeeze(-1)
            return reward, domain_logit

        def predict(self, features: ArrayLike) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
                reward, _ = self(x, adversarial_strength=0.0)
            return reward.cpu().numpy()

else:

    class DANNRegressor:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs):
            _require_torch()


@dataclass(frozen=True)
class DANNFit:
    model: DANNRegressor
    reward_loss: tuple[float, ...]
    domain_loss: tuple[float, ...]


def fit_dann(
    source_features: ArrayLike,
    source_scores: ArrayLike,
    target_features: ArrayLike,
    *,
    adversarial_strength: float = 0.1,
    hidden_dim: int = 64,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    random_state: int = 0,
) -> DANNFit:
    """Fit DANN without accepting Target scores."""

    _require_torch()
    source = np.asarray(source_features, dtype=np.float32)
    scores = np.asarray(source_scores, dtype=np.float32).reshape(-1)
    target = np.asarray(target_features, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Source and Target features must be 2D with equal width")
    if len(scores) != len(source):
        raise ValueError("source_scores must have one value per Source row")
    if adversarial_strength < 0:
        raise ValueError("adversarial_strength must be non-negative")

    torch.manual_seed(random_state)
    model = DANNRegressor(source.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    x_source = torch.as_tensor(source)
    y_source = torch.as_tensor(scores)
    x_target = torch.as_tensor(target)
    source_domain = torch.zeros(len(source))
    target_domain = torch.ones(len(target))
    reward_history: list[float] = []
    domain_history: list[float] = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        source_reward, source_logit = model(x_source, adversarial_strength)
        _, target_logit = model(x_target, adversarial_strength)
        reward_loss = mse(source_reward, y_source)
        domain_loss = bce(source_logit, source_domain) + bce(target_logit, target_domain)
        (reward_loss + domain_loss).backward()
        optimizer.step()
        reward_history.append(float(reward_loss.detach()))
        domain_history.append(float(domain_loss.detach()))

    return DANNFit(model, tuple(reward_history), tuple(domain_history))


"""Metrics used in the compact public examples.

Inputs are expected to be normalized to [0, 1]. For native score scales,
divide predictions and labels by the documented scale width first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("metrics require at least one row")
    return pred - truth


def nsae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Normalized score absolute error: ``abs(mean(prediction - truth))``."""

    return float(abs(_errors(y_true, y_pred).mean()))


def nmae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Normalized row mean absolute error."""

    return float(np.abs(_errors(y_true, y_pred)).mean())


def nrmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Normalized row root mean squared error."""

    return float(np.sqrt(np.square(_errors(y_true, y_pred)).mean()))


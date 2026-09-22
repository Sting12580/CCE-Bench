"""
Classifier-based density ratio estimation for MIPS (Saito & Joachims 2022).

Estimates w(x, phi) = p(phi | x, pi_target) / p(phi | x, pi_b)
without ever computing pi_b or pi_target in action space.

Training:
  - Positive samples: (x_i, phi_target_i) from target policy
  - Negative samples: (x_i, phi_b_i)     from behavior policy
  - Train binary classifier P(class=target | x, phi)
  - With balanced classes, w = P / (1 - P) by Bayes' rule.

Real-data note:
  pi_b is unknown (human clinician), but logged (x, phi(a_clinician)) IS a
  sample from p(phi | x, pi_b) by definition. So the "negative" pool comes
  for free from logged data. For "positive" we generate a_agent ~ pi_target
  for each logged x and embed.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def fit_density_ratio_classifier(
    features_target: np.ndarray,
    features_b: np.ndarray,
    seed: int = 0,
    C: float = 0.1,
    calibrate: bool = False,
):
    """Train logistic P(class=target | x, phi).

    Inputs are already-featurized concatenations of (x_features, phi_features).
    For L2-normalized embeddings (OpenAI, MedCPT, BGE-M3 outputs), DO NOT
    standardize: rows live on the unit sphere and feature-wise standardization
    pushes them off, destroying the geometric structure the linear classifier
    relies on. The default C=0.1 (strong L2) and clip-based weight bounding
    handle high-dim ill-conditioning instead. LBFGS still emits matmul
    overflow warnings during line search; these are line-search artifacts
    that do not affect the converged solution and are suppressed below.
    For low-dim toy settings, bump C to 100.0. Set calibrate=True for Platt
    scaling.

    Returns a fitted sklearn classifier exposing .predict_proba(...).
    """
    X = np.vstack([features_target, features_b])
    y = np.concatenate([
        np.ones(len(features_target), dtype=int),
        np.zeros(len(features_b), dtype=int),
    ])
    base = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    else:
        clf = base
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)
    return clf


def density_ratio(
    clf: LogisticRegression,
    features: np.ndarray,
    clip: float | None = 20.0,
) -> np.ndarray:
    """w(x, phi) = P / (1 - P), clipped to [0, clip] for stability."""
    weights, _ = density_ratio_with_diagnostics(clf, features, clip=clip)
    return weights


def density_ratio_with_diagnostics(
    clf: LogisticRegression,
    features: np.ndarray,
    clip: float | None = 20.0,
) -> tuple[np.ndarray, dict]:
    """Return density-ratio weights plus raw/clipping diagnostics.

    ``clip=None`` disables the final upper-bound clipping while preserving the
    numerical probability clipping used by the original ``density_ratio``.
    """
    raw_probabilities = clf.predict_proba(features)[:, 1]
    probabilities = np.clip(raw_probabilities, 1e-6, 1 - 1e-6)
    raw_weights = probabilities / (1 - probabilities)
    if clip is None:
        clipped_weights = raw_weights
        clip_rate = 0.0
    else:
        clipped_weights = np.clip(raw_weights, 0.0, clip)
        clip_rate = float(np.mean(raw_weights > clip))

    diagnostics = {
        "raw_probabilities": raw_probabilities,
        "probabilities": probabilities,
        "raw_weights": raw_weights,
        "clipped_weights": clipped_weights,
        "clip": clip,
        "clip_rate": clip_rate,
    }
    diagnostics.update(_quantile_summary(raw_weights, prefix="raw_weight"))
    diagnostics.update(_quantile_summary(clipped_weights, prefix="weight"))
    return clipped_weights, diagnostics


def summarize_density_ratio_weights(
    weights: np.ndarray,
    y_behavior: np.ndarray,
    raw_weights: np.ndarray | None = None,
    clip: float | None = None,
) -> dict:
    """Summarize positivity, clipping, and MIPS/SNIPS behavior."""
    w = _as_finite_1d("weights", weights)
    y = _as_finite_1d("y_behavior", y_behavior)
    if len(w) != len(y):
        raise ValueError("weights and y_behavior must have the same length.")

    raw_w = w if raw_weights is None else _as_finite_1d("raw_weights", raw_weights)
    if len(raw_w) != len(w):
        raise ValueError("raw_weights and weights must have the same length.")

    sum_w = float(w.sum())
    mean_w = float(w.mean())
    ess_fraction = effective_sample_size(w)
    v_snips = float((w * y).sum() / sum_w) if sum_w > 0 else 0.0
    v_unnormalized = float((w * y).mean())
    corr = _safe_corr(w, y)

    if clip is None:
        clip_rate = 0.0
    elif raw_weights is not None:
        clip_rate = float(np.mean(raw_w > clip))
    else:
        clip_rate = float(np.mean(w >= clip))

    summary = {
        "clip": clip,
        "n": int(len(w)),
        "mean_w": mean_w,
        "std_w": float(w.std(ddof=0)),
        "raw_max_w": float(raw_w.max()),
        "clip_rate": clip_rate,
        "ess": float(ess_fraction * len(w)),
        "ess_fraction": ess_fraction,
        "top1_weight_mass": _top_weight_mass(w, 0.01),
        "top5_weight_mass": _top_weight_mass(w, 0.05),
        "top10_weight_mass": _top_weight_mass(w, 0.10),
        "corr_w_y_behavior": corr,
        "v_snips": v_snips,
        "v_unnormalized_mips": v_unnormalized,
        "snips_minus_unnormalized_mips": float(v_snips - v_unnormalized),
    }
    summary.update(_quantile_summary(w, prefix="w"))
    summary.update(_quantile_summary(raw_w, prefix="raw_w"))
    return summary


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (sum w)^2 / sum w^2. High ESS means stable IS estimator.

    Returned as fraction of n in [0, 1]; ESS / n >= 0.30 is a common
    plan W7 sanity threshold.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 ** 2) / s2 / n)


def _quantile_summary(values: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(values, dtype=float).reshape(-1)
    quantiles = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    return {
        f"{prefix}_q{int(round(q * 100)):02d}": float(np.quantile(arr, q))
        for q in quantiles
    }


def _as_finite_1d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _top_weight_mass(weights: np.ndarray, fraction: float) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(len(w) * fraction)))
    return float(np.sort(w)[::-1][:k].sum() / total)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0

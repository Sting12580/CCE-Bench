"""Frozen zero-label estimators for the CCE-Benchmark primary-method run.

This module contains only public-input estimators.  It deliberately has no
private-gold loader or evaluation entry point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


FOLD_IDS = (0, 1, 2, 3, 4)


def array_sha256(value: np.ndarray) -> str:
    """Hash an array with its dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sequence_sha256(values: Sequence[Any]) -> str:
    """Hash an ordered sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_score(score_native: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Map a frozen native score scale to the unit interval."""

    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("The frozen native score range is invalid.")
    score = np.asarray(score_native, dtype=np.float64)
    if not np.all(np.isfinite(score)):
        raise ValueError("Source scores must be finite.")
    if np.any(score < minimum) or np.any(score > maximum):
        raise ValueError("Source scores fall outside the frozen native range.")
    return (score - minimum) / (maximum - minimum)


def source_mean_predictions(
    y_source_unit: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, float]:
    """Return M1's equal-Source-row constant prediction and value."""

    y = np.asarray(y_source_unit, dtype=np.float64).reshape(-1)
    if len(y) == 0 or target_count <= 0 or not np.all(np.isfinite(y)):
        raise ValueError("M1 requires finite Source scores and at least one Target row.")
    value = float(np.mean(y))
    return np.full(target_count, value, dtype=np.float64), value


def self_normalized_value(y_source_unit: np.ndarray, weights: np.ndarray) -> float:
    """Compute the Hájek self-normalized Source value."""

    y = np.asarray(y_source_unit, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(y) == 0 or y.shape != w.shape:
        raise ValueError("SN-MIPS outcomes and weights must be nonempty and aligned.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("SN-MIPS outcomes/weights must be finite and weights nonnegative.")
    denominator = float(np.sum(w))
    if denominator <= 0:
        raise ValueError("SN-MIPS weight sum must be positive.")
    return float(np.sum(w * y) / denominator)


def effective_sample_size(weights: np.ndarray) -> float:
    """Return ``(sum w)^2 / sum(w^2)`` with fail-closed guards."""

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) == 0 or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("ESS requires finite nonnegative weights.")
    numerator = float(np.sum(w)) ** 2
    denominator = float(np.sum(w * w))
    if numerator <= 0 or denominator <= 0:
        raise ValueError("ESS is zero or undefined.")
    return numerator / denominator


def sndr_predictions(
    target_dm_unit: np.ndarray,
    y_source_unit: np.ndarray,
    source_dm_oof_unit: np.ndarray,
    source_weights_oof: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the frozen SNDR-BGE scalar correction, then row-wise clipping."""

    target_dm = np.asarray(target_dm_unit, dtype=np.float64).reshape(-1)
    y_source = np.asarray(y_source_unit, dtype=np.float64).reshape(-1)
    source_dm = np.asarray(source_dm_oof_unit, dtype=np.float64).reshape(-1)
    weights = np.asarray(source_weights_oof, dtype=np.float64).reshape(-1)
    if len(target_dm) == 0 or len(y_source) == 0:
        raise ValueError("SNDR-BGE requires nonempty Source and Target rows.")
    if y_source.shape != source_dm.shape or y_source.shape != weights.shape:
        raise ValueError("SNDR-BGE Source arrays must be aligned.")
    arrays = (target_dm, y_source, source_dm, weights)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("SNDR-BGE inputs must be finite.")
    if np.any(weights < 0):
        raise ValueError("SNDR-BGE weights must be nonnegative.")
    denominator = float(np.sum(weights))
    if denominator <= 0:
        raise ValueError("SNDR-BGE weight sum must be positive.")
    correction = float(np.sum(weights * (y_source - source_dm)) / denominator)
    unprojected = target_dm + correction
    projected = np.clip(unprojected, 0.0, 1.0)
    diagnostics = {
        "dm_term_unit": float(np.mean(target_dm)),
        "correction_unit": correction,
        "unprojected_value_unit": float(np.mean(unprojected)),
        "projected_value_unit": float(np.mean(projected)),
        "projection_delta_unit": float(np.mean(projected) - np.mean(unprojected)),
    }
    return projected, diagnostics


def split_key_tokens(key: str) -> set[str]:
    """Normalize snake/kebab/camel-case keys for the Target leak scan."""

    with_camel_breaks = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    return {
        token
        for token in re.split(r"[^a-z0-9]+", with_camel_breaks.lower())
        if token
    }


def recursive_mapping_keys(value: Any) -> list[str]:
    """Collect mapping keys recursively without inspecting text values."""

    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(recursive_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(recursive_mapping_keys(nested))
    return keys


def forbidden_target_keys(
    row: dict[str, Any],
    *,
    forbidden_tokens: Sequence[str],
    exact_exceptions: Sequence[str],
) -> set[str]:
    """Return forbidden Target keys under the frozen token policy."""

    forbidden = {str(value).lower() for value in forbidden_tokens}
    exceptions = {str(value).lower() for value in exact_exceptions}
    bad: set[str] = set()
    for key in recursive_mapping_keys(row):
        if key.strip().lower() in exceptions:
            continue
        if split_key_tokens(key) & forbidden:
            bad.add(key)
    return bad


def _validate_folds(source_folds: np.ndarray, target_folds: np.ndarray) -> None:
    source = np.asarray(source_folds, dtype=np.int64).reshape(-1)
    target = np.asarray(target_folds, dtype=np.int64).reshape(-1)
    if set(source.tolist()) != set(FOLD_IDS) or set(target.tolist()) != set(FOLD_IDS):
        raise ValueError("Both Source and Target rows must cover fixed fold IDs 0..4.")


def _balanced_domain_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_source = int(np.sum(labels == 0))
    n_target = int(np.sum(labels == 1))
    if n_source <= 0 or n_target <= 0 or n_source + n_target != len(labels):
        raise ValueError("A density fit must contain both Source and Target rows.")
    return np.where(labels == 0, 0.5 / n_source, 0.5 / n_target).astype(np.float64)


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    seed: int,
    max_iter: int,
) -> LogisticRegression:
    sample_weight = _balanced_domain_sample_weight(labels)
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=float(c_value),
        max_iter=int(max_iter),
        tol=1e-4,
        fit_intercept=True,
        random_state=int(seed),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, labels, sample_weight=sample_weight)
    convergence = [row for row in caught if issubclass(row.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"Density/calibration convergence failure: {convergence[0].message}")
    if int(np.max(model.n_iter_)) >= int(max_iter):
        raise RuntimeError("Density/calibration exhausted max_iter.")
    return model


def crossfit_sn_mips(
    *,
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_folds: np.ndarray,
    target_folds: np.ndarray,
    y_source_unit: np.ndarray,
    seed: int,
    density_c: float = 0.1,
    platt_c: float = 1.0,
    max_iter: int = 2000,
    probability_epsilon: float = 1e-6,
    weight_clip: float = 20.0,
) -> dict[str, Any]:
    """Five-fold item-grouped SN-MIPS with nested frozen-fold Platt calibration."""

    source_x = np.asarray(source_features, dtype=np.float32)
    target_x = np.asarray(target_features, dtype=np.float32)
    source_fold = np.asarray(source_folds, dtype=np.int64).reshape(-1)
    target_fold = np.asarray(target_folds, dtype=np.int64).reshape(-1)
    y_source = np.asarray(y_source_unit, dtype=np.float64).reshape(-1)
    if source_x.ndim != 2 or target_x.ndim != 2 or source_x.shape[1] != target_x.shape[1]:
        raise ValueError("Source and Target density features must be aligned 2D matrices.")
    if len(source_x) != len(source_fold) or len(source_x) != len(y_source):
        raise ValueError("Source features, folds, and outcomes are not aligned.")
    if len(target_x) != len(target_fold):
        raise ValueError("Target features and folds are not aligned.")
    if source_x.shape[1] != 3072:
        raise ValueError("The frozen M13 density feature dimension is 3072.")
    if any(not np.all(np.isfinite(value)) for value in (source_x, target_x, y_source)):
        raise ValueError("M13 inputs must be finite.")
    _validate_folds(source_fold, target_fold)

    features = np.vstack([source_x, target_x]).astype(np.float32, copy=False)
    labels = np.concatenate(
        [np.zeros(len(source_x), dtype=np.int64), np.ones(len(target_x), dtype=np.int64)]
    )
    folds = np.concatenate([source_fold, target_fold])
    oof_probability = np.full(len(features), np.nan, dtype=np.float64)
    outer_records: list[dict[str, Any]] = []

    # Each three-fold base model is shared by the two orientations of its
    # excluded fold pair.  This is algebraically identical to 20 inner fits but
    # requires only C(5,2)=10 fits.
    inner_decisions: dict[tuple[int, int], np.ndarray] = {}
    three_fold_fit_hashes: dict[tuple[int, int], str] = {}
    for first in FOLD_IDS:
        for second in FOLD_IDS:
            if second <= first:
                continue
            fit_mask = (folds != first) & (folds != second)
            model = _fit_logistic(
                features[fit_mask],
                labels[fit_mask],
                c_value=density_c,
                seed=seed,
                max_iter=max_iter,
            )
            for heldout in (first, second):
                mask = folds == heldout
                inner_decisions[(first if heldout == second else second, heldout)] = (
                    np.asarray(model.decision_function(features[mask]), dtype=np.float64)
                )
            three_fold_fit_hashes[(first, second)] = array_sha256(
                np.concatenate([model.coef_.reshape(-1), model.intercept_.reshape(-1)])
            )

    for outer in FOLD_IDS:
        calibration_scores: list[np.ndarray] = []
        calibration_labels: list[np.ndarray] = []
        calibration_folds: list[np.ndarray] = []
        for inner in FOLD_IDS:
            if inner == outer:
                continue
            scores = inner_decisions[(outer, inner)]
            inner_labels = labels[folds == inner]
            calibration_scores.append(scores)
            calibration_labels.append(inner_labels)
            calibration_folds.append(np.full(len(scores), inner, dtype=np.int64))
        score_vector = np.concatenate(calibration_scores)
        label_vector = np.concatenate(calibration_labels)
        fold_vector = np.concatenate(calibration_folds)
        if set(fold_vector.tolist()) != set(FOLD_IDS) - {outer}:
            raise RuntimeError("Nested Platt coverage drifted from the frozen folds.")
        platt = _fit_logistic(
            score_vector.reshape(-1, 1),
            label_vector,
            c_value=platt_c,
            seed=seed,
            max_iter=max_iter,
        )
        outer_train = folds != outer
        outer_test = folds == outer
        base = _fit_logistic(
            features[outer_train],
            labels[outer_train],
            c_value=density_c,
            seed=seed,
            max_iter=max_iter,
        )
        outer_score = np.asarray(base.decision_function(features[outer_test]), dtype=np.float64)
        calibrated = np.asarray(
            platt.predict_proba(outer_score.reshape(-1, 1))[:, 1],
            dtype=np.float64,
        )
        oof_probability[outer_test] = calibrated
        outer_records.append(
            {
                "outer_fold": outer,
                "base_fit_rows": int(np.sum(outer_train)),
                "outer_oof_rows": int(np.sum(outer_test)),
                "calibration_rows": int(len(label_vector)),
                "calibration_fold_ids": sorted(set(fold_vector.tolist())),
                "base_parameter_sha256": array_sha256(
                    np.concatenate([base.coef_.reshape(-1), base.intercept_.reshape(-1)])
                ),
                "platt_parameter_sha256": array_sha256(
                    np.concatenate([platt.coef_.reshape(-1), platt.intercept_.reshape(-1)])
                ),
            }
        )

    if not np.all(np.isfinite(oof_probability)):
        raise RuntimeError("M13 failed to produce complete finite OOF probabilities.")
    clipped_probability = np.clip(
        oof_probability,
        float(probability_epsilon),
        1.0 - float(probability_epsilon),
    )
    raw_ratio = clipped_probability / (1.0 - clipped_probability)
    clipped_ratio = np.clip(raw_ratio, 0.0, float(weight_clip))
    source_probability = clipped_probability[: len(source_x)]
    target_probability = clipped_probability[len(source_x) :]
    source_raw_ratio = raw_ratio[: len(source_x)]
    source_weight = clipped_ratio[: len(source_x)]
    value = self_normalized_value(y_source, source_weight)
    ess = effective_sample_size(source_weight)
    oof_auc = float(roc_auc_score(labels, oof_probability))
    oof_accuracy = float(np.mean((oof_probability >= 0.5).astype(np.int64) == labels))
    return {
        "v_hat_unit": value,
        "source_probability_oof": source_probability,
        "target_probability_oof": target_probability,
        "source_raw_ratio_oof": source_raw_ratio,
        "source_weight_oof": source_weight,
        "ess": ess,
        "ess_fraction": ess / len(source_weight),
        "positivity_warning": 0.0 < ess / len(source_weight) < 0.10,
        "oof_domain_auc": oof_auc,
        "oof_domain_accuracy": oof_accuracy,
        "outer_records": outer_records,
        "three_fold_base_fit_sha256": {
            f"exclude_{first}_{second}": digest
            for (first, second), digest in sorted(three_fold_fit_hashes.items())
        },
    }


def _l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


def _build_reward_projector(
    context_dim: int,
    action_dim: int,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
):
    import torch
    from torch import nn

    class RewardProjector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh(),
            )
            self.action = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh(),
            )
            self.reward = nn.Sequential(
                nn.Linear(latent_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def encode_context(self, value):
            return self.context(value)

        def encode_action(self, value):
            return self.action(value)

        def forward(self, context, action):
            z_context = self.encode_context(context)
            z_action = self.encode_action(action)
            return self.reward(
                torch.cat([z_context, z_action, z_context * z_action], dim=1)
            )

    return RewardProjector()


def fixed_fold_outcome_predictions(
    *,
    base_question: np.ndarray,
    base_answer: np.ndarray,
    y_source_unit: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    seed: int,
    latent_dim: int = 64,
    hidden_dim: int = 64,
    pca_dim: int = 128,
    max_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    dropout: float = 0.0,
    patience: int = 50,
) -> dict[str, Any]:
    """Fit one fixed-fold M14 outcome nuisance and predict its outer Source fold."""

    import torch
    from torch import nn

    question = np.asarray(base_question, dtype=np.float32)
    answer = np.asarray(base_answer, dtype=np.float32)
    y = np.asarray(y_source_unit, dtype=np.float64).reshape(-1)
    fold = np.asarray(folds, dtype=np.int64).reshape(-1)
    if question.shape != answer.shape or question.ndim != 2:
        raise ValueError("M14 base question/answer embeddings must be aligned matrices.")
    if len(question) != len(y) or len(question) != len(fold):
        raise ValueError("M14 embeddings, outcomes, and folds are not aligned.")
    if outer_fold not in FOLD_IDS or set(fold.tolist()) != set(FOLD_IDS):
        raise ValueError("M14 outer fold must use the fixed IDs 0..4.")
    inner_validation_fold = (int(outer_fold) + 1) % 5
    outer_train = fold != outer_fold
    outer_test = fold == outer_fold
    representation_train = outer_train & (fold != inner_validation_fold)
    representation_validation = fold == inner_validation_fold
    if np.any(outer_test & representation_train) or np.any(
        outer_test & representation_validation
    ):
        raise RuntimeError("M14 outer-fold leakage detected before fitting.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    x_train_np = question[representation_train]
    a_train_np = answer[representation_train]
    y_train_np = y[representation_train].astype(np.float32)
    x_val_np = question[representation_validation]
    a_val_np = answer[representation_validation]
    y_val_np = y[representation_validation].astype(np.float32)
    y_mean = float(np.mean(y_train_np))
    y_std = float(np.std(y_train_np, ddof=0) + 1e-6)
    train_target = ((y_train_np - y_mean) / y_std).reshape(-1, 1)
    val_target = ((y_val_np - y_mean) / y_std).reshape(-1, 1)
    model = _build_reward_projector(
        context_dim=question.shape[1],
        action_dim=answer.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    x_train = torch.from_numpy(x_train_np)
    a_train = torch.from_numpy(a_train_np)
    train_y = torch.from_numpy(train_target)
    x_val = torch.from_numpy(x_val_np)
    a_val = torch.from_numpy(a_val_np)
    val_y = torch.from_numpy(val_target)
    best_state: dict[str, Any] | None = None
    best_metric = float("inf")
    best_epoch = 0
    stale = 0
    epochs_trained = 0
    for epoch in range(max_epochs):
        epochs_trained = epoch + 1
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train, a_train), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            metric = float(loss_fn(model(x_val, a_val), val_y).item())
        if metric + 1e-7 < best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            stale = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("M14 outcome projector did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        learned_question = _l2_normalize(
            model.encode_context(torch.from_numpy(question)).cpu().numpy()
        )
        learned_answer = _l2_normalize(
            model.encode_action(torch.from_numpy(answer)).cpu().numpy()
        )

    # PCA is the reward-free part of the M3 representation pipeline.  After the
    # projector checkpoint is selected on the fixed inner validation fold, PCA
    # uses every outer-training row, exactly as a full M3 nuisance fit would;
    # the outer test fold remains completely excluded.
    pca_fit = np.vstack([question[outer_train], answer[outer_train]])
    if pca_dim > min(pca_fit.shape):
        raise ValueError("M14 frozen PCA dimension is infeasible for this fold.")
    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(pca_fit)
    pca_question = _l2_normalize(pca.transform(question).astype(np.float32))
    pca_answer = _l2_normalize(pca.transform(answer).astype(np.float32))
    z_question = np.concatenate([pca_question, learned_question], axis=1)
    z_answer = np.concatenate([pca_answer, learned_answer], axis=1)
    features = np.concatenate([z_question, z_answer, z_question * z_answer], axis=1)
    if features.shape[1] != 576:
        raise RuntimeError("M14 outcome feature dimension drifted from 576.")
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )
    gbr.fit(features[outer_train], y[outer_train])
    prediction = np.clip(
        np.asarray(gbr.predict(features[outer_test]), dtype=np.float64), 0.0, 1.0
    )
    return {
        "outer_indices": np.flatnonzero(outer_test),
        "prediction_unit": prediction,
        "diagnostics": {
            "outer_fold": int(outer_fold),
            "inner_validation_fold": int(inner_validation_fold),
            "representation_train_fold_ids": sorted(
                set(fold[representation_train].tolist())
            ),
            "outer_train_fold_ids": sorted(set(fold[outer_train].tolist())),
            "outer_test_fold_ids": sorted(set(fold[outer_test].tolist())),
            "representation_train_rows": int(np.sum(representation_train)),
            "representation_validation_rows": int(np.sum(representation_validation)),
            "outer_gbr_train_rows": int(np.sum(outer_train)),
            "outer_oof_rows": int(np.sum(outer_test)),
            "best_epoch": int(best_epoch),
            "epochs_trained": int(epochs_trained),
            "best_validation_mse_standardized": float(best_metric),
            "pca_fit_rows": int(len(pca_fit)),
            "pca_explained_variance_ratio_sum": float(
                np.sum(pca.explained_variance_ratio_)
            ),
            "outer_train_indices_sha256": array_sha256(
                np.flatnonzero(outer_train).astype(np.int64)
            ),
            "outer_test_indices_sha256": array_sha256(
                np.flatnonzero(outer_test).astype(np.int64)
            ),
        },
    }


def crossfit_outcome_oof(
    *,
    base_question: np.ndarray,
    base_answer: np.ndarray,
    y_source_unit: np.ndarray,
    folds: np.ndarray,
    seed: int,
    parallel_outer_folds: int | None = None,
) -> dict[str, Any]:
    """Build complete five-fold Source OOF predictions for M14."""

    prediction = np.full(len(y_source_unit), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    coverage = np.zeros(len(y_source_unit), dtype=np.int64)
    jobs = (
        min(len(FOLD_IDS), 5)
        if parallel_outer_folds is None and len(y_source_unit) >= 5_000
        else int(parallel_outer_folds or 1)
    )
    if jobs < 1 or jobs > len(FOLD_IDS):
        raise ValueError("M14 parallel_outer_folds must be between 1 and 5.")
    if jobs == 1:
        fitted_folds = [
            fixed_fold_outcome_predictions(
                base_question=base_question,
                base_answer=base_answer,
                y_source_unit=y_source_unit,
                folds=folds,
                outer_fold=outer,
                seed=seed,
            )
            for outer in FOLD_IDS
        ]
    else:
        from joblib import Parallel, delayed, parallel_config

        with parallel_config(
            backend="loky",
            n_jobs=jobs,
            inner_max_num_threads=1,
        ):
            fitted_folds = Parallel()(  # type: ignore[misc]
                delayed(fixed_fold_outcome_predictions)(
                    base_question=base_question,
                    base_answer=base_answer,
                    y_source_unit=y_source_unit,
                    folds=folds,
                    outer_fold=outer,
                    seed=seed,
                )
                for outer in FOLD_IDS
            )
    for fitted in fitted_folds:
        indices = np.asarray(fitted["outer_indices"], dtype=np.int64)
        prediction[indices] = np.asarray(fitted["prediction_unit"], dtype=np.float64)
        coverage[indices] += 1
        fold_records.append(dict(fitted["diagnostics"]))
    if not np.all(coverage == 1) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("M14 Source OOF coverage must be exactly one per row.")
    return {
        "prediction_unit": prediction,
        "fold_records": fold_records,
        "coverage_sha256": array_sha256(coverage),
        "parallel_outer_fold_processes": jobs,
    }


@dataclass(frozen=True)
class FrozenBertDannConfig:
    """Exact M6/M6-control shared hyperparameters."""

    hidden_dim: int = 128
    dropout: float = 0.1
    max_length: int = 512
    source_batch_size: int = 8
    target_batch_size: int = 8
    max_epochs: int = 20
    learning_rate_encoder: float = 2e-5
    learning_rate_heads: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    lambda_domain_max: float = 0.1
    patience: int = 5
    unfreeze_last_n_layers: int = 2
    minimum_positive_lambda_epochs: int = 3


def nominal_dann_lambda(epoch: int, config: FrozenBertDannConfig) -> float:
    """Legacy epoch-level DANN schedule frozen by the registry."""

    if epoch < config.warmup_epochs:
        return 0.0
    denominator = max(config.max_epochs - config.warmup_epochs, 1)
    progress = float(epoch - config.warmup_epochs) / float(denominator)
    return float(config.lambda_domain_max) * float(
        2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
    )


def _epoch_batch_schedule(
    *,
    source_train_indices: np.ndarray,
    target_count: int,
    seed: int,
    config: FrozenBertDannConfig,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], dict[str, Any]]:
    source_indices = np.asarray(source_train_indices, dtype=np.int64).reshape(-1)
    if len(source_indices) == 0 or target_count <= 0:
        raise ValueError("M6 batch scheduling requires Source train and Target rows.")
    schedules: list[list[tuple[np.ndarray, np.ndarray]]] = []
    source_epoch_digests: list[str] = []
    target_epoch_digests: list[str] = []
    for epoch in range(config.max_epochs):
        source_rng = np.random.default_rng(np.random.SeedSequence([seed, 6101, epoch]))
        target_rng = np.random.default_rng(np.random.SeedSequence([seed, 6102, epoch]))
        source_order = source_indices[source_rng.permutation(len(source_indices))]
        source_batches = [
            source_order[start : start + config.source_batch_size]
            for start in range(0, len(source_order), config.source_batch_size)
        ]
        target_needed = len(source_batches) * config.target_batch_size
        target_parts: list[np.ndarray] = []
        produced = 0
        while produced < target_needed:
            part = target_rng.permutation(target_count).astype(np.int64)
            target_parts.append(part)
            produced += len(part)
        target_order = np.concatenate(target_parts)[:target_needed]
        target_batches = [
            target_order[index * config.target_batch_size : (index + 1) * config.target_batch_size]
            for index in range(len(source_batches))
        ]
        schedules.append(list(zip(source_batches, target_batches, strict=True)))
        source_epoch_digests.append(array_sha256(source_order))
        target_epoch_digests.append(array_sha256(target_order))
    audit = {
        "source_epoch_index_sha256": source_epoch_digests,
        "target_epoch_index_sha256": target_epoch_digests,
        "source_schedule_sha256": sequence_sha256(source_epoch_digests),
        "target_schedule_sha256": sequence_sha256(target_epoch_digests),
        "paired_schedule_sha256": sequence_sha256(
            [*source_epoch_digests, *target_epoch_digests]
        ),
    }
    return schedules, audit


def _token_batch(tokenized: dict[str, Any], indices: np.ndarray, device: str) -> dict[str, Any]:
    import torch

    tensor_indices = torch.as_tensor(indices, dtype=torch.long)
    return {name: tensor[tensor_indices].to(device) for name, tensor in tokenized.items()}


def _predict_bert_batches(
    model: Any,
    tokenized: dict[str, Any],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(next(iter(tokenized.values()))), batch_size):
            stop = min(start + batch_size, len(next(iter(tokenized.values()))))
            indices = np.arange(start, stop, dtype=np.int64)
            encoded = _token_batch(tokenized, indices, device)
            z_value = model.encode(encoded)
            predictions.append(
                model.reward_from_z(z_value).detach().cpu().numpy().reshape(-1)
            )
    return np.concatenate(predictions).astype(np.float64)


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fit_bert_arm(
    *,
    tokenized_source: dict[str, Any],
    tokenized_target: dict[str, Any],
    y_source_unit: np.ndarray,
    source_train_indices: np.ndarray,
    source_validation_indices: np.ndarray,
    schedules: list[list[tuple[np.ndarray, np.ndarray]]],
    model_snapshot: Path,
    device: str,
    seed: int,
    lambda_domain_max: float,
    checkpoint_path: Path | None,
    config: FrozenBertDannConfig,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from transformers import AutoModel

    from cce_data.estimators import dann_finetuned_encoder as dann

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    encoder = AutoModel.from_pretrained(str(model_snapshot), local_files_only=True)
    trainability = dann._configure_encoder_trainability(
        encoder, config.unfreeze_last_n_layers
    )
    model = dann._EncoderDANNRewardModel(
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        head_hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        pooling="cls",
    ).to(device)
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    encoder_parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    reward_parameters = list(model.reward.parameters())
    domain_parameters = list(model.domain.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": config.learning_rate_encoder},
            {
                "params": reward_parameters + domain_parameters,
                "lr": config.learning_rate_heads,
            },
        ],
        weight_decay=config.weight_decay,
    )
    reward_loss_function = nn.MSELoss()
    domain_loss_function = nn.BCEWithLogitsLoss()
    train_indices = np.asarray(source_train_indices, dtype=np.int64)
    validation_indices = np.asarray(source_validation_indices, dtype=np.int64)
    y = np.asarray(y_source_unit, dtype=np.float32).reshape(-1)
    y_mean = float(np.mean(y[train_indices]))
    y_std = float(np.std(y[train_indices], ddof=0) + 1e-6)
    standardized = (y - y_mean) / y_std
    positive_nominal_epochs = 0
    best_state: dict[str, Any] | None = None
    best_metric = float("inf")
    best_epoch = 0
    stale = 0
    epoch_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch, batches in enumerate(schedules):
        model.train()
        nominal_lambda = nominal_dann_lambda(epoch, config)
        actual_lambda = 0.0 if lambda_domain_max == 0 else nominal_lambda
        if nominal_lambda > 0:
            positive_nominal_epochs += 1
        reward_losses: list[float] = []
        domain_losses: list[float] = []
        epoch_started = time.perf_counter()
        for source_indices, target_indices in batches:
            optimizer.zero_grad()
            source_encoded = _token_batch(tokenized_source, source_indices, device)
            target_encoded = _token_batch(tokenized_target, target_indices, device)
            source_z = model.encode(source_encoded)
            target_z = model.encode(target_encoded)
            source_prediction = model.reward_from_z(source_z).reshape(-1)
            source_target = torch.from_numpy(standardized[source_indices]).to(device)
            reward_loss = reward_loss_function(source_prediction, source_target)
            domain_loss = dann._balanced_domain_loss(
                model=model,
                z_behavior=source_z,
                z_target=target_z,
                lambda_domain=actual_lambda,
                loss_fn=domain_loss_function,
            )
            total_loss = reward_loss + domain_loss
            total_loss.backward()
            optimizer.step()
            reward_losses.append(float(reward_loss.detach().cpu().item()))
            domain_losses.append(float(domain_loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            validation_parts: list[np.ndarray] = []
            for start in range(0, len(validation_indices), config.source_batch_size):
                batch_indices = validation_indices[
                    start : start + config.source_batch_size
                ]
                validation_encoded = _token_batch(
                    tokenized_source, batch_indices, device
                )
                validation_parts.append(
                    model.reward_from_z(model.encode(validation_encoded))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            validation_prediction = np.concatenate(validation_parts)
            validation_mse = float(
                np.mean(
                    (
                        validation_prediction
                        - standardized[validation_indices]
                    )
                    ** 2
                )
            )
        eligible = positive_nominal_epochs >= config.minimum_positive_lambda_epochs
        selected = False
        if eligible and validation_mse + 1e-7 < best_metric:
            best_metric = validation_mse
            best_epoch = epoch + 1
            stale = 0
            selected = True
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        elif eligible:
            stale += 1
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "nominal_lambda": float(nominal_lambda),
                "actual_lambda": float(actual_lambda),
                "reward_loss": float(np.mean(reward_losses)),
                "balanced_domain_loss": float(np.mean(domain_losses)),
                "source_validation_reward_mse_standardized": validation_mse,
                "checkpoint_eligible": bool(eligible),
                "checkpoint_selected": bool(selected),
                "wall_time_seconds": time.perf_counter() - epoch_started,
            }
        )
        if eligible and config.patience > 0 and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("M6 arm produced no checkpoint after the eligibility gate.")
    model.load_state_dict(best_state)
    source_prediction_std = _predict_bert_batches(
        model,
        tokenized_source,
        batch_size=config.source_batch_size,
        device=device,
    )
    target_prediction_std = _predict_bert_batches(
        model,
        tokenized_target,
        batch_size=config.target_batch_size,
        device=device,
    )
    source_prediction = np.clip(source_prediction_std * y_std + y_mean, 0.0, 1.0)
    target_prediction = np.clip(target_prediction_std * y_std + y_mean, 0.0, 1.0)
    best_state_sha256 = _state_dict_sha256(best_state)
    checkpoint_payload = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        checkpoint_payload = {
            "schema_version": "cce-primary-m6-trainable-checkpoint-v1",
            "seed": int(seed),
            "lambda_domain_max": float(lambda_domain_max),
            "initial_state_sha256": initial_state_sha256,
            "best_state_sha256": best_state_sha256,
            "best_epoch": int(best_epoch),
            "trainable_state_dict": {
                name: best_state[name].detach().cpu()
                for name in sorted(trainable_names)
                if name in best_state
            },
        }
        torch.save(checkpoint_payload, checkpoint_path)
    diagnostics = {
        "config": asdict(config),
        "seed": int(seed),
        "lambda_domain_max": float(lambda_domain_max),
        "initial_state_sha256": initial_state_sha256,
        "best_state_sha256": best_state_sha256,
        "best_epoch": int(best_epoch),
        "best_validation_mse_standardized": float(best_metric),
        "epochs_trained": len(epoch_records),
        "epoch_records": epoch_records,
        "source_train_indices_sha256": array_sha256(train_indices),
        "source_validation_indices_sha256": array_sha256(validation_indices),
        "source_train_rows": int(len(train_indices)),
        "source_validation_rows": int(len(validation_indices)),
        "target_rows": int(len(target_prediction)),
        "reward_standardization_mean_train_only": y_mean,
        "reward_standardization_std_train_only": y_std,
        "encoder_layer_container": trainability["layer_container"],
        "unfrozen_encoder_layer_names": trainability["unfrozen_layer_names"],
        "target_rewards_used": False,
        "target_domain_branch_executed": True,
        "target_domain_gradient_to_shared_encoder": "zero" if lambda_domain_max == 0 else "reversed",
        "elapsed_seconds": time.perf_counter() - started,
    }
    del checkpoint_payload, best_state, model, encoder, optimizer
    gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return {
        "source_prediction_unit": source_prediction,
        "target_prediction_unit": target_prediction,
        "diagnostics": diagnostics,
    }


def fit_fixed_fold_bert_dann_pair(
    *,
    questions_source: Sequence[str],
    answers_source: Sequence[str],
    y_source_unit: np.ndarray,
    source_train_indices: np.ndarray,
    source_validation_indices: np.ndarray,
    questions_target: Sequence[str],
    answers_target: Sequence[str],
    model_snapshot: Path,
    device: str,
    seed: int,
    primary_checkpoint_path: Path | None = None,
    control_checkpoint_path: Path | None = None,
    config: FrozenBertDannConfig | None = None,
) -> dict[str, Any]:
    """Fit paired λ=.1 and λ=0 BERT arms with identical initialization/batches."""

    from transformers import AutoTokenizer

    cfg = config or FrozenBertDannConfig()
    if len(questions_source) != len(answers_source) or len(questions_source) != len(
        y_source_unit
    ):
        raise ValueError("M6 Source text/outcome lengths differ.")
    if len(questions_target) != len(answers_target):
        raise ValueError("M6 Target text lengths differ.")
    if set(np.asarray(source_train_indices).tolist()) & set(
        np.asarray(source_validation_indices).tolist()
    ):
        raise ValueError("M6 fixed Source train and validation indices overlap.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True)
    tokenized_source = tokenizer(
        list(questions_source),
        list(answers_source),
        padding="max_length",
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    tokenized_target = tokenizer(
        list(questions_target),
        list(answers_target),
        padding="max_length",
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    schedules, schedule_audit = _epoch_batch_schedule(
        source_train_indices=np.asarray(source_train_indices, dtype=np.int64),
        target_count=len(questions_target),
        seed=seed,
        config=cfg,
    )
    primary = _fit_bert_arm(
        tokenized_source=tokenized_source,
        tokenized_target=tokenized_target,
        y_source_unit=y_source_unit,
        source_train_indices=source_train_indices,
        source_validation_indices=source_validation_indices,
        schedules=schedules,
        model_snapshot=model_snapshot,
        device=device,
        seed=seed,
        lambda_domain_max=cfg.lambda_domain_max,
        checkpoint_path=primary_checkpoint_path,
        config=cfg,
    )
    control = _fit_bert_arm(
        tokenized_source=tokenized_source,
        tokenized_target=tokenized_target,
        y_source_unit=y_source_unit,
        source_train_indices=source_train_indices,
        source_validation_indices=source_validation_indices,
        schedules=schedules,
        model_snapshot=model_snapshot,
        device=device,
        seed=seed,
        lambda_domain_max=0.0,
        checkpoint_path=control_checkpoint_path,
        config=cfg,
    )
    if (
        primary["diagnostics"]["initial_state_sha256"]
        != control["diagnostics"]["initial_state_sha256"]
    ):
        raise RuntimeError("M6 primary/control model initialization digests differ.")
    for result in (primary, control):
        result["diagnostics"]["batch_schedule"] = schedule_audit
    return {
        "primary": primary,
        "control": control,
        "matched_audit": {
            "status": "PASS",
            "initial_state_sha256": primary["diagnostics"]["initial_state_sha256"],
            **schedule_audit,
        },
    }

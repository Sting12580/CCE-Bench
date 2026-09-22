"""Corrected post-gold M13/M14 estimators for the fixed CCE benchmark.

This module is deliberately separate from the frozen v1 implementation.  It
contains no private-gold loader.  M13-v2 learns a Source-reward-informed 576D
representation under strict item-fold cross-fitting, then estimates balanced
Source-to-Target density ratios.  M14-v2 is computed by the unchanged SNDR
formula in :mod:`cce_primary_zero_label` using the corrected M13 weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import random
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cce_data.estimators.cce_primary_zero_label import (
    FOLD_IDS,
    _build_reward_projector,
    _l2_normalize,
    array_sha256,
    effective_sample_size,
    self_normalized_value,
)


@dataclass(frozen=True)
class CorrectedM13Config:
    """Frozen corrected M13 hyperparameters."""

    latent_dim: int = 64
    hidden_dim: int = 64
    pca_dim: int = 128
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.0
    patience: int = 50
    density_c: float = 0.1
    platt_c: float = 1.0
    logistic_max_iter: int = 2000
    probability_epsilon: float = 1e-6
    weight_clip: float = 20.0


def balanced_domain_sample_weight_mean_one(labels: np.ndarray) -> np.ndarray:
    """Give both domains equal total mass while preserving mean weight one.

    This scale matters for regularized scikit-learn estimators.  The v1 helper
    had total weight one, which made ``C=.1`` effectively shrink with sample
    size.  Here the two class totals are each ``N/2`` and the total is ``N``.
    """

    domain = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_source = int(np.sum(domain == 0))
    n_target = int(np.sum(domain == 1))
    n_total = len(domain)
    if n_source <= 0 or n_target <= 0 or n_source + n_target != n_total:
        raise ValueError("A density/calibration fit requires both domain labels.")
    weights = np.where(
        domain == 0,
        n_total / (2.0 * n_source),
        n_total / (2.0 * n_target),
    ).astype(np.float64)
    if not np.isclose(float(np.mean(weights)), 1.0, atol=1e-12):
        raise RuntimeError("Mean-one domain-weight invariant failed.")
    if not np.isclose(float(np.sum(weights[domain == 0])), n_total / 2.0, atol=1e-10):
        raise RuntimeError("Source domain mass is not N/2.")
    if not np.isclose(float(np.sum(weights[domain == 1])), n_total / 2.0, atol=1e-10):
        raise RuntimeError("Target domain mass is not N/2.")
    return weights


def _fit_logistic_mean_one(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    seed: int,
    max_iter: int,
) -> tuple[LogisticRegression, dict[str, float]]:
    x = np.asarray(features, dtype=np.float32)
    domain = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or len(x) != len(domain) or not np.all(np.isfinite(x)):
        raise ValueError("Logistic features/labels must be finite and aligned.")
    sample_weight = balanced_domain_sample_weight_mean_one(domain)
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
        model.fit(x, domain, sample_weight=sample_weight)
    convergence = [row for row in caught if issubclass(row.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"Density/calibration convergence failure: {convergence[0].message}")
    if int(np.max(model.n_iter_)) >= int(max_iter):
        raise RuntimeError("Density/calibration exhausted max_iter.")
    return model, {
        "fit_rows": float(len(domain)),
        "sample_weight_sum": float(np.sum(sample_weight)),
        "sample_weight_mean": float(np.mean(sample_weight)),
        "source_weight_mass": float(np.sum(sample_weight[domain == 0])),
        "target_weight_mass": float(np.sum(sample_weight[domain == 1])),
    }


def pair_validation_fold(excluded_folds: Sequence[int]) -> int:
    """Return the frozen symmetric validation fold for an excluded pair.

    Starting immediately after the larger excluded fold, cycle through 0..4
    and select the first allowed fold.  This yields one reusable fit per
    unordered excluded pair.
    """

    excluded = tuple(sorted(int(value) for value in excluded_folds))
    if len(excluded) != 2 or len(set(excluded)) != 2 or not set(excluded) <= set(FOLD_IDS):
        raise ValueError("Pair validation requires two distinct fixed fold IDs.")
    for offset in range(1, 6):
        candidate = (excluded[-1] + offset) % 5
        if candidate not in excluded:
            return candidate
    raise RuntimeError("No pair validation fold is available.")


def representation_seed(*, experiment_seed: int, excluded_folds: Sequence[int]) -> int:
    """Return the common frozen experiment seed for every nuisance fit."""

    excluded = tuple(sorted(int(value) for value in excluded_folds))
    if len(excluded) not in {1, 2} or len(set(excluded)) != len(excluded):
        raise ValueError("A corrected representation excludes one or two folds.")
    if not set(excluded) <= set(FOLD_IDS):
        raise ValueError("Corrected representation folds must be fixed IDs 0..4.")
    return int(experiment_seed)


def _validate_inputs(
    *,
    source_question: np.ndarray,
    source_answer: np.ndarray,
    target_question: np.ndarray,
    target_answer: np.ndarray,
    source_folds: np.ndarray,
    target_folds: np.ndarray,
    y_source_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sq = np.asarray(source_question, dtype=np.float32)
    sa = np.asarray(source_answer, dtype=np.float32)
    tq = np.asarray(target_question, dtype=np.float32)
    ta = np.asarray(target_answer, dtype=np.float32)
    sf = np.asarray(source_folds, dtype=np.int64).reshape(-1)
    tf = np.asarray(target_folds, dtype=np.int64).reshape(-1)
    y = np.asarray(y_source_unit, dtype=np.float64).reshape(-1)
    if sq.ndim != 2 or sa.shape != sq.shape or tq.ndim != 2 or ta.shape != tq.shape:
        raise ValueError("Question/answer base embeddings must be aligned 2D matrices.")
    if sq.shape[1] != tq.shape[1] or sq.shape[1] != 1024:
        raise ValueError("Corrected M13 requires aligned 1024D BGE-M3 base embeddings.")
    if len(sq) != len(sf) or len(sq) != len(y) or len(tq) != len(tf):
        raise ValueError("Corrected M13 embeddings, folds, and Source outcomes are misaligned.")
    if set(sf.tolist()) != set(FOLD_IDS) or set(tf.tolist()) != set(FOLD_IDS):
        raise ValueError("Source and Target rows must both cover fixed fold IDs 0..4.")
    if any(not np.all(np.isfinite(value)) for value in (sq, sa, tq, ta, y)):
        raise ValueError("Corrected M13 inputs must be finite.")
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("Corrected M13 Source outcomes must be on [0,1].")
    return sq, sa, tq, ta, sf, tf, y


def _fit_reward_projector_and_transform(
    *,
    source_question: np.ndarray,
    source_answer: np.ndarray,
    target_question: np.ndarray,
    target_answer: np.ndarray,
    y_source_unit: np.ndarray,
    representation_train_mask: np.ndarray,
    representation_validation_mask: np.ndarray,
    pca_fit_mask: np.ndarray,
    seed: int,
    config: CorrectedM13Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Source-only learned/PCA components and transform all public rows."""

    import torch
    from torch import nn

    train_mask = np.asarray(representation_train_mask, dtype=bool)
    validation_mask = np.asarray(representation_validation_mask, dtype=bool)
    pca_mask = np.asarray(pca_fit_mask, dtype=bool)
    if not np.any(train_mask) or not np.any(validation_mask) or not np.any(pca_mask):
        raise ValueError("Representation train/validation/PCA masks must be nonempty.")
    if np.any(train_mask & validation_mask):
        raise RuntimeError("Representation train and validation masks overlap.")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)

    x_train_np = source_question[train_mask]
    a_train_np = source_answer[train_mask]
    y_train_np = y_source_unit[train_mask].astype(np.float32)
    x_val_np = source_question[validation_mask]
    a_val_np = source_answer[validation_mask]
    y_val_np = y_source_unit[validation_mask].astype(np.float32)
    y_mean = float(np.mean(y_train_np))
    y_std = float(np.std(y_train_np, ddof=0) + 1e-6)
    train_target = ((y_train_np - y_mean) / y_std).reshape(-1, 1)
    val_target = ((y_val_np - y_mean) / y_std).reshape(-1, 1)

    model = _build_reward_projector(
        context_dim=source_question.shape[1],
        action_dim=source_answer.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
    for epoch in range(config.max_epochs):
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
            if config.patience > 0 and stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Corrected M13 projector did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        learned_sq = _l2_normalize(
            model.encode_context(torch.tensor(source_question)).cpu().numpy()
        )
        learned_sa = _l2_normalize(
            model.encode_action(torch.tensor(source_answer)).cpu().numpy()
        )
        learned_tq = _l2_normalize(
            model.encode_context(torch.tensor(target_question)).cpu().numpy()
        )
        learned_ta = _l2_normalize(
            model.encode_action(torch.tensor(target_answer)).cpu().numpy()
        )

    pca_fit = np.vstack([source_question[pca_mask], source_answer[pca_mask]])
    if config.pca_dim > min(pca_fit.shape):
        raise ValueError("Corrected M13 PCA dimension is infeasible for this fold fit.")
    pca = PCA(n_components=config.pca_dim, random_state=int(seed))
    pca.fit(pca_fit)
    pca_sq = _l2_normalize(pca.transform(source_question).astype(np.float32))
    pca_sa = _l2_normalize(pca.transform(source_answer).astype(np.float32))
    pca_tq = _l2_normalize(pca.transform(target_question).astype(np.float32))
    pca_ta = _l2_normalize(pca.transform(target_answer).astype(np.float32))

    z_sq = np.concatenate([pca_sq, learned_sq], axis=1)
    z_sa = np.concatenate([pca_sa, learned_sa], axis=1)
    z_tq = np.concatenate([pca_tq, learned_tq], axis=1)
    z_ta = np.concatenate([pca_ta, learned_ta], axis=1)
    source_features = np.concatenate([z_sq, z_sa, z_sq * z_sa], axis=1).astype(
        np.float32, copy=False
    )
    target_features = np.concatenate([z_tq, z_ta, z_tq * z_ta], axis=1).astype(
        np.float32, copy=False
    )
    expected_feature_dim = 3 * (config.pca_dim + config.latent_dim)
    if (
        source_features.shape[1] != expected_feature_dim
        or target_features.shape[1] != expected_feature_dim
    ):
        raise RuntimeError("Corrected M13 feature dimension drifted from its frozen config.")
    if config == CorrectedM13Config() and expected_feature_dim != 576:
        raise RuntimeError("The formal corrected M13 default must remain exactly 576D.")

    projector_values = np.concatenate(
        [best_state[name].detach().cpu().numpy().reshape(-1) for name in sorted(best_state)]
    ).astype(np.float32)
    pca_values = np.concatenate(
        [pca.mean_.reshape(-1), pca.components_.reshape(-1)]
    ).astype(np.float32)
    diagnostics = {
        "seed": int(seed),
        "config": asdict(config),
        "feature_dimension": int(expected_feature_dim),
        "representation_train_rows": int(np.sum(train_mask)),
        "representation_validation_rows": int(np.sum(validation_mask)),
        "pca_source_rows": int(np.sum(pca_mask)),
        "pca_fit_rows_after_question_answer_stack": int(len(pca_fit)),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "best_validation_mse_standardized": float(best_metric),
        "projector_parameter_sha256": array_sha256(projector_values),
        "pca_parameter_sha256": array_sha256(pca_values),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    del model, best_state, pca, projector_values, pca_values
    gc.collect()
    return source_features, target_features, diagnostics


def _fit_excluded_fold_density(
    *,
    source_question: np.ndarray,
    source_answer: np.ndarray,
    target_question: np.ndarray,
    target_answer: np.ndarray,
    source_folds: np.ndarray,
    target_folds: np.ndarray,
    y_source_unit: np.ndarray,
    excluded_folds: tuple[int, ...],
    experiment_seed: int,
    config: CorrectedM13Config,
) -> dict[str, Any]:
    excluded = tuple(sorted(int(value) for value in excluded_folds))
    if len(excluded) not in {1, 2} or not set(excluded) <= set(FOLD_IDS):
        raise ValueError("A density fit must exclude one outer fold or one fold pair.")
    allowed = tuple(fold for fold in FOLD_IDS if fold not in excluded)
    validation_fold = (
        (excluded[0] + 1) % 5 if len(excluded) == 1 else pair_validation_fold(excluded)
    )
    if validation_fold not in allowed:
        raise RuntimeError("Frozen representation validation fold is not allowed.")
    fit_seed = representation_seed(
        experiment_seed=int(experiment_seed), excluded_folds=excluded
    )
    representation_train = np.isin(source_folds, allowed) & (
        source_folds != validation_fold
    )
    representation_validation = source_folds == validation_fold
    pca_fit = np.isin(source_folds, allowed)
    if np.any(np.isin(source_folds[representation_train], excluded)) or np.any(
        np.isin(source_folds[representation_validation], excluded)
    ):
        raise RuntimeError("Excluded Source fold entered representation fitting.")

    source_x, target_x, representation_diagnostics = (
        _fit_reward_projector_and_transform(
            source_question=source_question,
            source_answer=source_answer,
            target_question=target_question,
            target_answer=target_answer,
            y_source_unit=y_source_unit,
            representation_train_mask=representation_train,
            representation_validation_mask=representation_validation,
            pca_fit_mask=pca_fit,
            seed=fit_seed,
            config=config,
        )
    )
    features = np.vstack([source_x, target_x]).astype(np.float32, copy=False)
    labels = np.concatenate(
        [np.zeros(len(source_x), dtype=np.int64), np.ones(len(target_x), dtype=np.int64)]
    )
    folds = np.concatenate([source_folds, target_folds])
    base_fit = np.isin(folds, allowed)
    heldout = np.isin(folds, excluded)
    if np.any(base_fit & heldout) or not np.any(heldout):
        raise RuntimeError("Density fit/heldout fold masks are invalid.")
    base, balance_audit = _fit_logistic_mean_one(
        features[base_fit],
        labels[base_fit],
        c_value=config.density_c,
        seed=fit_seed,
        max_iter=config.logistic_max_iter,
    )
    heldout_indices = np.flatnonzero(heldout).astype(np.int64)
    decision = np.asarray(base.decision_function(features[heldout]), dtype=np.float64)
    parameter = np.concatenate([base.coef_.reshape(-1), base.intercept_.reshape(-1)])
    record = {
        "excluded_fold_ids": list(excluded),
        "allowed_fold_ids": list(allowed),
        "representation_validation_fold": int(validation_fold),
        "representation_train_fold_ids": sorted(
            set(source_folds[representation_train].tolist())
        ),
        "pca_fit_fold_ids": sorted(set(source_folds[pca_fit].tolist())),
        "representation_seed": int(fit_seed),
        "base_fit_rows": int(np.sum(base_fit)),
        "heldout_rows": int(np.sum(heldout)),
        "heldout_indices_sha256": array_sha256(heldout_indices),
        "base_parameter_sha256": array_sha256(parameter),
        "base_balance_audit": balance_audit,
        "representation": representation_diagnostics,
        "excluded_source_rewards_used": False,
        "excluded_target_text_used_for_fitting": False,
        "target_rewards_used": False,
    }
    del source_x, target_x, features, base
    gc.collect()
    return {
        "excluded_folds": excluded,
        "heldout_indices": heldout_indices,
        "decision": decision,
        "record": record,
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def crossfit_reward_informed_sn_mips_v2(
    *,
    source_question: np.ndarray,
    source_answer: np.ndarray,
    target_question: np.ndarray,
    target_answer: np.ndarray,
    source_folds: np.ndarray,
    target_folds: np.ndarray,
    y_source_unit: np.ndarray,
    seed: int,
    config: CorrectedM13Config | None = None,
    parallel_fits: int = 1,
) -> dict[str, Any]:
    """Strict 5-fold corrected SN-MIPS with 15 representation/base fits.

    Ten unordered excluded-pair fits supply the nested Platt calibration
    scores.  Five excluded-outer fits supply the final OOF base scores.  Thus a
    held-out Source reward never enters its representation, PCA, base density,
    or calibration nuisance path.
    """

    cfg = config or CorrectedM13Config()
    sq, sa, tq, ta, sf, tf, y = _validate_inputs(
        source_question=source_question,
        source_answer=source_answer,
        target_question=target_question,
        target_answer=target_answer,
        source_folds=source_folds,
        target_folds=target_folds,
        y_source_unit=y_source_unit,
    )
    if parallel_fits < 1 or parallel_fits > 5:
        raise ValueError("parallel_fits must be between 1 and 5.")
    specifications: list[tuple[int, ...]] = [
        (first, second)
        for first in FOLD_IDS
        for second in FOLD_IDS
        if second > first
    ] + [(outer,) for outer in FOLD_IDS]

    kwargs = {
        "source_question": sq,
        "source_answer": sa,
        "target_question": tq,
        "target_answer": ta,
        "source_folds": sf,
        "target_folds": tf,
        "y_source_unit": y,
        "experiment_seed": int(seed),
        "config": cfg,
    }
    if parallel_fits == 1:
        fitted = [
            _fit_excluded_fold_density(excluded_folds=spec, **kwargs)
            for spec in specifications
        ]
    else:
        from joblib import Parallel, delayed, parallel_config

        with parallel_config(
            backend="loky", n_jobs=int(parallel_fits), inner_max_num_threads=1
        ):
            fitted = Parallel()(  # type: ignore[misc]
                delayed(_fit_excluded_fold_density)(excluded_folds=spec, **kwargs)
                for spec in specifications
            )
    by_excluded = {
        tuple(int(value) for value in row["excluded_folds"]): row for row in fitted
    }
    if set(by_excluded) != set(specifications) or len(fitted) != 15:
        raise RuntimeError("Corrected M13 did not complete exactly 10 pair + 5 outer fits.")

    labels = np.concatenate(
        [np.zeros(len(sq), dtype=np.int64), np.ones(len(tq), dtype=np.int64)]
    )
    folds = np.concatenate([sf, tf])
    base_oof_decision = np.full(len(labels), np.nan, dtype=np.float64)
    calibrated_oof_probability = np.full(len(labels), np.nan, dtype=np.float64)
    coverage = np.zeros(len(labels), dtype=np.int64)
    outer_records: list[dict[str, Any]] = []
    for outer in FOLD_IDS:
        calibration_scores: list[np.ndarray] = []
        calibration_labels: list[np.ndarray] = []
        calibration_folds: list[np.ndarray] = []
        for inner in FOLD_IDS:
            if inner == outer:
                continue
            pair = tuple(sorted((outer, inner)))
            pair_fit = by_excluded[pair]
            pair_index = np.asarray(pair_fit["heldout_indices"], dtype=np.int64)
            pair_decision = np.asarray(pair_fit["decision"], dtype=np.float64)
            select = folds[pair_index] == inner
            inner_index = pair_index[select]
            if not np.array_equal(inner_index, np.flatnonzero(folds == inner)):
                raise RuntimeError("Nested calibration item/fold ordering drifted.")
            calibration_scores.append(pair_decision[select])
            calibration_labels.append(labels[inner_index])
            calibration_folds.append(np.full(len(inner_index), inner, dtype=np.int64))
        score_vector = np.concatenate(calibration_scores)
        label_vector = np.concatenate(calibration_labels)
        fold_vector = np.concatenate(calibration_folds)
        if set(fold_vector.tolist()) != set(FOLD_IDS) - {outer}:
            raise RuntimeError("Nested Platt coverage drifted from fixed folds.")
        platt_seed = int(seed)
        platt, platt_balance = _fit_logistic_mean_one(
            score_vector.reshape(-1, 1),
            label_vector,
            c_value=cfg.platt_c,
            seed=platt_seed,
            max_iter=cfg.logistic_max_iter,
        )
        outer_fit = by_excluded[(outer,)]
        outer_index = np.asarray(outer_fit["heldout_indices"], dtype=np.int64)
        expected_index = np.flatnonzero(folds == outer)
        if not np.array_equal(outer_index, expected_index):
            raise RuntimeError("Outer OOF row ordering drifted.")
        outer_decision = np.asarray(outer_fit["decision"], dtype=np.float64)
        calibrated = np.asarray(
            platt.predict_proba(outer_decision.reshape(-1, 1))[:, 1], dtype=np.float64
        )
        base_oof_decision[outer_index] = outer_decision
        calibrated_oof_probability[outer_index] = calibrated
        coverage[outer_index] += 1
        platt_parameter = np.concatenate(
            [platt.coef_.reshape(-1), platt.intercept_.reshape(-1)]
        )
        outer_records.append(
            {
                "outer_fold": int(outer),
                "outer_fit": outer_fit["record"],
                "calibration_rows": int(len(label_vector)),
                "calibration_fold_ids": sorted(set(fold_vector.tolist())),
                "platt_seed": int(platt_seed),
                "platt_slope": float(platt.coef_.reshape(-1)[0]),
                "platt_intercept": float(platt.intercept_.reshape(-1)[0]),
                "platt_parameter_sha256": array_sha256(platt_parameter),
                "platt_balance_audit": platt_balance,
            }
        )

    if not np.all(coverage == 1):
        raise RuntimeError("Corrected M13 OOF coverage must equal one for every row.")
    if any(
        not np.all(np.isfinite(value))
        for value in (base_oof_decision, calibrated_oof_probability)
    ):
        raise RuntimeError("Corrected M13 produced incomplete/non-finite OOF scores.")

    base_probability = _sigmoid(base_oof_decision)
    clipped_probability = np.clip(
        calibrated_oof_probability,
        float(cfg.probability_epsilon),
        1.0 - float(cfg.probability_epsilon),
    )
    raw_ratio = clipped_probability / (1.0 - clipped_probability)
    clipped_ratio = np.clip(raw_ratio, 0.0, float(cfg.weight_clip))
    source_probability = clipped_probability[: len(sq)]
    target_probability = clipped_probability[len(sq) :]
    source_raw_ratio = raw_ratio[: len(sq)]
    source_weight = clipped_ratio[: len(sq)]
    value = self_normalized_value(y, source_weight)
    ess = effective_sample_size(source_weight)
    base_auc = float(roc_auc_score(labels, base_oof_decision))
    calibrated_auc = float(roc_auc_score(labels, calibrated_oof_probability))
    calibrated_accuracy = float(
        np.mean((calibrated_oof_probability >= 0.5).astype(np.int64) == labels)
    )
    eps = float(cfg.probability_epsilon)
    p_for_loss = np.clip(calibrated_oof_probability, eps, 1.0 - eps)
    brier = float(np.mean((calibrated_oof_probability - labels) ** 2))
    log_loss = float(
        -np.mean(labels * np.log(p_for_loss) + (1 - labels) * np.log(1 - p_for_loss))
    )
    base_decision_sd = float(np.std(base_oof_decision, ddof=0))
    calibrated_probability_sd = float(np.std(calibrated_oof_probability, ddof=0))
    source_weight_sd = float(np.std(source_weight, ddof=0))
    # This gate catches a numerical/regularization collapse, while permitting a
    # genuinely indistinguishable Source/Target distribution to reduce to M1.
    collapse = bool(base_auc >= 0.60 and calibrated_probability_sd <= 1e-4)
    if collapse:
        raise RuntimeError(
            "CALIBRATION_COLLAPSE: base OOF AUC >= .60 but calibrated probability SD <= 1e-4."
        )

    weight_quantiles = np.quantile(
        source_weight, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    )
    pair_records = [by_excluded[spec]["record"] for spec in specifications[:10]]
    return {
        "v_hat_unit": value,
        "m1_source_mean_unit": float(np.mean(y)),
        "m13_minus_m1_unit": float(value - np.mean(y)),
        "source_probability_oof": source_probability,
        "target_probability_oof": target_probability,
        "source_raw_ratio_oof": source_raw_ratio,
        "source_weight_oof": source_weight,
        "base_decision_oof": base_oof_decision,
        "base_probability_oof": base_probability,
        "calibrated_probability_oof": calibrated_oof_probability,
        "ess": ess,
        "ess_fraction": ess / len(source_weight),
        "positivity_warning": 0.0 < ess / len(source_weight) < 0.10,
        "base_oof_domain_auc": base_auc,
        "calibrated_oof_domain_auc": calibrated_auc,
        "calibrated_oof_domain_accuracy": calibrated_accuracy,
        "calibrated_oof_brier": brier,
        "calibrated_oof_log_loss": log_loss,
        "base_decision_sd": base_decision_sd,
        "calibrated_probability_sd": calibrated_probability_sd,
        "source_weight_sd": source_weight_sd,
        "source_weight_cv": source_weight_sd / max(float(np.mean(source_weight)), 1e-12),
        "near_uniform_weight_warning": bool(
            source_weight_sd / max(float(np.mean(source_weight)), 1e-12) <= 0.001
        ),
        "source_weight_clip_fraction": float(np.mean(source_raw_ratio >= cfg.weight_clip)),
        "source_weight_quantiles": {
            name: float(value)
            for name, value in zip(
                ("q0", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q100"),
                weight_quantiles,
                strict=True,
            )
        },
        "pair_fit_records": pair_records,
        "outer_records": outer_records,
        "coverage_sha256": array_sha256(coverage),
        "parallel_fit_processes": int(parallel_fits),
        "strict_fit_count": 15,
        "calibration_collapse": False,
        "config": asdict(cfg),
    }


__all__ = [
    "CorrectedM13Config",
    "balanced_domain_sample_weight_mean_one",
    "crossfit_reward_informed_sn_mips_v2",
    "pair_validation_fold",
    "representation_seed",
]

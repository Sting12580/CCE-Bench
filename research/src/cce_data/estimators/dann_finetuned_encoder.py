"""Fine-tuned encoder DM estimator with optional DANN domain adaptation.

This module trains an encoder-based outcome model directly on question-answer
pairs. Behavior/source rows provide reward labels; target rows are used only for
the domain-adversarial loss. Target rewards may be passed for evaluation
diagnostics, but they are never used in training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


@dataclass
class EncoderDANNDMConfig:
    """Configuration for encoder fine-tuned DANN-DM."""

    model_name: str = "google-bert/bert-base-uncased"
    pooling: str = "cls"
    hidden_dim: int = 128
    dropout: float = 0.1
    max_length: int = 512
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    dynamic_padding: bool = False
    max_epochs: int = 20
    learning_rate_encoder: float = 2e-5
    learning_rate_head: float = 1e-3
    weight_decay: float = 0.01
    validation_fraction: float = 0.2
    patience: int = 5
    warmup_epochs: int = 3
    lambda_domain: float = 0.1
    lambda_schedule: str = "logistic"
    unfreeze_last_n_layers: int = 2
    device: str = "auto"
    seed: int = 0
    checkpoint_dir: str | None = None
    compute_domain_diagnostics: bool = True
    domain_probe_oof: bool = False
    record_detailed_diagnostics: bool = False
    record_async_baseline_diagnostics: bool = False


def fit_text_encoder_dann_dm(
    *,
    questions_behavior: list[str],
    answers_behavior: list[str],
    y_behavior: np.ndarray,
    questions_target: list[str],
    answers_target: list[str],
    y_target: np.ndarray | None = None,
    config: EncoderDANNDMConfig | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Fit a text encoder outcome model and return a target-value estimate."""

    cfg = config or EncoderDANNDMConfig()
    _validate_inputs(
        questions_behavior=questions_behavior,
        answers_behavior=answers_behavior,
        y_behavior=y_behavior,
        questions_target=questions_target,
        answers_target=answers_target,
        y_target=y_target,
        cfg=cfg,
    )

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if tokenizer is None or encoder is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
        encoder = encoder or AutoModel.from_pretrained(cfg.model_name)

    _set_seeds(cfg.seed)
    device = _resolve_device(cfg.device)
    trainability_audit = _configure_encoder_trainability(
        encoder,
        cfg.unfreeze_last_n_layers,
    )
    memory_tracker = _MemoryTracker(device)

    tokenized_behavior = tokenizer(
        questions_behavior,
        answers_behavior,
        padding=not cfg.dynamic_padding,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors=None if cfg.dynamic_padding else "pt",
    )
    tokenized_target = tokenizer(
        questions_target,
        answers_target,
        padding=not cfg.dynamic_padding,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors=None if cfg.dynamic_padding else "pt",
    )

    y_np = np.asarray(y_behavior, dtype=np.float32).reshape(-1, 1)
    y_mean = float(y_np.mean())
    y_std = float(y_np.std() + 1e-6)
    y_standardized = (y_np - y_mean) / y_std

    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx = _train_val_split(len(y_np), cfg.validation_fraction, rng)
    train_dataset = _PairDataset(
        tokenized_behavior,
        labels=torch.from_numpy(y_standardized),
        indices=train_idx,
    )
    val_dataset = _PairDataset(
        tokenized_behavior,
        labels=torch.from_numpy(y_standardized),
        indices=val_idx,
    )
    target_dataset = _PairDataset(tokenized_target)
    behavior_full_dataset = _PairDataset(
        tokenized_behavior,
        labels=torch.from_numpy(y_standardized),
    )

    collate_fn = None
    if cfg.dynamic_padding:
        from transformers import DataCollatorWithPadding

        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    behavior_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    use_domain_loss = cfg.lambda_domain > 0.0
    target_loader = (
        DataLoader(
            target_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        if use_domain_loss
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    behavior_full_loader = DataLoader(
        behavior_full_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    target_full_loader = DataLoader(
        target_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    hidden_size = int(getattr(encoder.config, "hidden_size"))
    model = _EncoderDANNRewardModel(
        encoder=encoder,
        hidden_size=hidden_size,
        head_hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    ).to(device)
    if not use_domain_loss:
        for param in model.domain.parameters():
            param.requires_grad = False
    memory_tracker.sample()

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("encoder.") and p.requires_grad
    ]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.learning_rate_encoder})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.learning_rate_head})
    if not param_groups:
        raise ValueError("No trainable parameters are available for encoder fine-tuning.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    reward_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_trained = 0
    epoch_times_seconds: list[float] = []
    epoch_diagnostics: list[dict[str, Any]] = []
    cumulative_adversarial_updates = 0
    if use_domain_loss and target_loader is None:
        raise RuntimeError("target_loader must be available when domain loss is enabled.")
    target_iter = _cycle_loader(target_loader) if target_loader is not None else None
    for epoch in range(cfg.max_epochs):
        epoch_started = time.perf_counter()
        epochs_trained = epoch + 1
        model.train()
        lambda_eff = _lambda_for_epoch(cfg, epoch)
        total_loss = 0.0
        reward_loss_total = 0.0
        steps = 0
        nonzero_adversarial_updates = 0
        reward_encoder_gradient_norms: list[float] = []
        reward_head_gradient_norms: list[float] = []
        async_gradient_snapshot = None
        optimizer.zero_grad()
        loader_steps = len(behavior_loader)
        for step_index, behavior_batch in enumerate(behavior_loader):
            behavior_inputs, behavior_labels = _batch_to_device(behavior_batch, device)

            behavior_z = model.encode(behavior_inputs)
            reward_pred = model.reward_from_z(behavior_z)
            reward_loss = reward_loss_fn(reward_pred, behavior_labels)
            loss = reward_loss
            if use_domain_loss:
                if target_iter is None:
                    raise RuntimeError("target_iter must be initialized when domain loss is enabled.")
                target_batch = next(target_iter)
                target_inputs, _ = _batch_to_device(target_batch, device)
                target_z = model.encode(target_inputs)
                domain_loss = _balanced_domain_loss(
                    model=model,
                    z_behavior=behavior_z,
                    z_target=target_z,
                    lambda_domain=lambda_eff,
                    loss_fn=domain_loss_fn,
                )
                loss = reward_loss + domain_loss
            group_start = (step_index // cfg.gradient_accumulation_steps) * (
                cfg.gradient_accumulation_steps
            )
            group_size = min(
                cfg.gradient_accumulation_steps,
                loader_steps - group_start,
            )
            (loss / float(group_size)).backward()
            should_step = (
                (step_index + 1) % cfg.gradient_accumulation_steps == 0
                or step_index + 1 == loader_steps
            )
            if should_step:
                if cfg.record_detailed_diagnostics:
                    reward_encoder_gradient_norms.append(_parameter_gradient_norm(encoder_params))
                    reward_head_gradient_norms.append(
                        _parameter_gradient_norm(list(model.reward.parameters()))
                    )
                elif cfg.record_async_baseline_diagnostics and async_gradient_snapshot is None:
                    async_gradient_snapshot = (
                        _parameter_gradient_norm_tensor(encoder_params),
                        _parameter_gradient_norm_tensor(list(model.reward.parameters())),
                    )
                if use_domain_loss and lambda_eff > 0.0:
                    nonzero_adversarial_updates += 1
                    cumulative_adversarial_updates += 1
                optimizer.step()
                optimizer.zero_grad()
            total_loss += float(loss.detach().cpu().item())
            reward_loss_total += float(reward_loss.detach().cpu().item())
            steps += 1
            memory_tracker.sample()

        validation_mae_async = None
        if cfg.record_async_baseline_diagnostics:
            metric, validation_mae_async = _validation_mse_with_async_mae(
                model,
                val_loader,
                reward_loss_fn,
                device,
            )
        else:
            metric = _validation_mse(model, val_loader, reward_loss_fn, device)
        validation_mae_epoch = None
        if cfg.record_detailed_diagnostics:
            _, validation_mae_epoch = _validation_reward_metrics(
                model,
                val_loader,
                reward_loss_fn,
                device,
            )
        memory_tracker.sample()
        epoch_times_seconds.append(time.perf_counter() - epoch_started)
        epoch_diagnostics.append(
            {
                "epoch": epoch + 1,
                "reward_loss": reward_loss_total / max(steps, 1),
                "source_domain_bce": None,
                "target_domain_bce": None,
                "balanced_domain_loss": 0.0 if not use_domain_loss else None,
                "total_loss": total_loss / max(steps, 1),
                "source_validation_reward_mse_standardized": float(metric),
                "source_validation_reward_mae_standardized": (
                    validation_mae_async
                    if cfg.record_async_baseline_diagnostics
                    else validation_mae_epoch
                ),
                "lambda_first": float(lambda_eff),
                "lambda_mean": float(lambda_eff),
                "lambda_max": float(lambda_eff),
                "nonzero_adversarial_update_count": nonzero_adversarial_updates,
                "cumulative_adversarial_update_count": cumulative_adversarial_updates,
                "reward_encoder_gradient_norm": (
                    async_gradient_snapshot[0]
                    if async_gradient_snapshot is not None
                    else _mean_or_zero(reward_encoder_gradient_norms)
                ),
                "adversarial_encoder_gradient_norm": 0.0 if not use_domain_loss else None,
                "reward_head_gradient_norm": (
                    async_gradient_snapshot[1]
                    if async_gradient_snapshot is not None
                    else _mean_or_zero(reward_head_gradient_norms)
                ),
                "domain_head_gradient_norm": 0.0 if not use_domain_loss else None,
                "online_domain_head_accuracy": None,
                "online_domain_head_auc": None,
                "best_checkpoint_eligibility": True,
                "wall_time_seconds": epoch_times_seconds[-1],
                "peak_memory_bytes": memory_tracker.diagnostics()["peak_memory_bytes"],
            }
        )
        if np.isnan(metric):
            metric = total_loss / max(steps, 1)
        if metric + 1e-7 < best_metric:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if cfg.patience > 0 and stale_epochs >= cfg.patience:
                break

    if cfg.record_async_baseline_diagnostics:
        for epoch_row in epoch_diagnostics:
            for key in (
                "source_validation_reward_mae_standardized",
                "reward_encoder_gradient_norm",
                "reward_head_gradient_norm",
            ):
                value = epoch_row.get(key)
                if hasattr(value, "detach"):
                    epoch_row[key] = float(value.detach().cpu().item())

    if best_state is not None:
        model.load_state_dict(best_state)

    behavior_pred_std, behavior_z = _predict(model, behavior_full_loader, device)
    target_pred_std, target_z = _predict(model, target_full_loader, device)
    memory_tracker.sample()
    behavior_pred = behavior_pred_std * y_std + y_mean
    target_pred = target_pred_std * y_std + y_mean
    v_hat = float(target_pred.mean())

    y_behavior_raw = y_np.reshape(-1)
    train_mse = _mse(behavior_pred[train_idx], y_behavior_raw[train_idx])
    val_mse = _mse(behavior_pred[val_idx], y_behavior_raw[val_idx]) if len(val_idx) else float("nan")
    val_mae = _mae(behavior_pred[val_idx], y_behavior_raw[val_idx]) if len(val_idx) else float("nan")
    baseline_mse = _mse(np.full_like(y_behavior_raw, y_mean), y_behavior_raw)

    diagnostics: dict[str, Any] = {
        "method": "encoder_dann_dm",
        "config": asdict(cfg),
        "model_name": cfg.model_name,
        "encoder_class": type(model.encoder).__name__,
        "pooling": cfg.pooling,
        "n_behavior": len(questions_behavior),
        "n_target": len(questions_target),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch + 1),
        "best_validation_mse_standardized": float(best_metric),
        "train_mse": train_mse,
        "validation_mse": val_mse,
        "validation_mae": val_mae,
        "baseline_mse": baseline_mse,
        "y_mean": y_mean,
        "y_std": y_std,
        "uses_target_rewards": False,
        "uses_target_rewards_for_training": False,
        "uses_target_rewards_for_early_stopping": False,
        "uses_target_rewards_for_model_selection": False,
        "uses_target_rewards_for_evaluation": y_target is not None,
        "uses_source_rewards_for_reward_head": True,
        "uses_source_rewards_for_domain_head": False,
        "uses_target_domain_labels": bool(use_domain_loss),
        "uses_target_features_for_training": use_domain_loss,
        "uses_target_features_for_domain_adaptation": use_domain_loss,
        "lambda_domain": float(cfg.lambda_domain),
        "lambda_schedule": cfg.lambda_schedule,
        "lambda_effective_last_epoch": float(_lambda_for_epoch(cfg, max(epochs_trained - 1, 0))),
        "warmup_epochs": int(cfg.warmup_epochs),
        "unfreeze_last_n_layers": int(cfg.unfreeze_last_n_layers),
        "encoder_layer_container": trainability_audit["layer_container"],
        "encoder_layer_count": trainability_audit["layer_count"],
        "unfrozen_encoder_layer_names": trainability_audit["unfrozen_layer_names"],
        "epoch_times_seconds": epoch_times_seconds,
        "epoch_diagnostics": epoch_diagnostics,
        "physical_batch_size": int(cfg.batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "effective_batch_size": int(cfg.batch_size * cfg.gradient_accumulation_steps),
        "dynamic_padding": bool(cfg.dynamic_padding),
        "total_encoder_parameters": int(sum(p.numel() for p in model.encoder.parameters())),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_encoder_parameters": int(
            sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        ),
        "trainable_total_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "trainable_parameter_names": [
            name for name, param in model.named_parameters() if param.requires_grad
        ],
        "trainable_non_encoder_parameter_names": [
            name
            for name, param in model.named_parameters()
            if not name.startswith("encoder.") and param.requires_grad
        ],
        "best_checkpoint_sha256": (
            _state_dict_sha256(best_state) if best_state is not None else None
        ),
        "best_checkpoint_adversarial_update_count": int(cumulative_adversarial_updates),
    }
    diagnostics.update(memory_tracker.diagnostics())
    if cfg.compute_domain_diagnostics:
        if cfg.domain_probe_oof:
            diagnostics.update(
                _domain_diagnostics_oof(
                    model,
                    behavior_z,
                    target_z,
                    device,
                    seed=cfg.seed,
                )
            )
        else:
            diagnostics.update(_domain_diagnostics(model, behavior_z, target_z, device))

    if y_target is not None:
        y_target_raw = np.asarray(y_target, dtype=np.float32).reshape(-1)
        diagnostics.update(
            {
                "target_mse_eval_only": _mse(target_pred, y_target_raw),
                "target_mean_bias_eval_only": float(v_hat - float(y_target_raw.mean())),
                "target_mean_mae_eval_only": abs(float(v_hat - float(y_target_raw.mean()))),
            }
        )

    if cfg.checkpoint_dir:
        _save_checkpoint(
            checkpoint_dir=Path(cfg.checkpoint_dir),
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            y_mean=y_mean,
            y_std=y_std,
            diagnostics=diagnostics,
        )

    return {
        "v_hat": v_hat,
        "behavior_pred": behavior_pred.astype(np.float32),
        "target_pred": target_pred.astype(np.float32),
        "diagnostics": diagnostics,
    }


def fit_text_encoder_true_dann_dm(
    *,
    questions_behavior: list[str],
    answers_behavior: list[str],
    y_behavior: np.ndarray,
    questions_target: list[str],
    answers_target: list[str],
    config: EncoderDANNDMConfig,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Fit the strict optimizer-step DANN experiment without a target-reward API.

    Target inputs are text pairs only.  In particular, this function deliberately
    has no ``y_target`` argument, so target rewards cannot participate in loss,
    early stopping, checkpoint selection, or prediction freezing.
    """

    cfg = config
    _validate_inputs(
        questions_behavior=questions_behavior,
        answers_behavior=answers_behavior,
        y_behavior=y_behavior,
        questions_target=questions_target,
        answers_target=answers_target,
        y_target=None,
        cfg=cfg,
    )
    if cfg.warmup_epochs != 0:
        raise ValueError("True DANN requires warmup_epochs=0.")
    if cfg.lambda_schedule != "logistic":
        raise ValueError("True DANN requires the optimizer-step logistic schedule.")
    if cfg.gradient_accumulation_steps != 1:
        raise ValueError("True DANN experiment fixes gradient_accumulation_steps=1.")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if tokenizer is None or encoder is None:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
        encoder = encoder or AutoModel.from_pretrained(cfg.model_name)

    _set_seeds(cfg.seed)
    device = _resolve_device(cfg.device)
    trainability_audit = _configure_encoder_trainability(
        encoder,
        cfg.unfreeze_last_n_layers,
    )
    memory_tracker = _MemoryTracker(device)

    tokenized_behavior = tokenizer(
        questions_behavior,
        answers_behavior,
        padding=not cfg.dynamic_padding,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors=None if cfg.dynamic_padding else "pt",
    )
    tokenized_target = tokenizer(
        questions_target,
        answers_target,
        padding=not cfg.dynamic_padding,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors=None if cfg.dynamic_padding else "pt",
    )

    y_np = np.asarray(y_behavior, dtype=np.float32).reshape(-1, 1)
    y_mean = float(y_np.mean())
    y_std = float(y_np.std() + 1e-6)
    y_standardized = (y_np - y_mean) / y_std
    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx = _train_val_split(len(y_np), cfg.validation_fraction, rng)

    train_dataset = _PairDataset(
        tokenized_behavior,
        labels=torch.from_numpy(y_standardized),
        indices=train_idx,
    )
    val_dataset = _PairDataset(
        tokenized_behavior,
        labels=torch.from_numpy(y_standardized),
        indices=val_idx,
    )
    # These labels are domain labels, not rewards.  The target dataset contains
    # tokenized text and domain=1 only; no score tensor exists in this branch.
    target_dataset = _PairDataset(
        tokenized_target,
        labels=torch.ones((len(questions_target), 1), dtype=torch.float32),
    )
    behavior_full_dataset = _PairDataset(
        tokenized_behavior,
        labels=torch.from_numpy(y_standardized),
    )

    collate_fn = None
    if cfg.dynamic_padding:
        from transformers import DataCollatorWithPadding

        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    behavior_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    behavior_full_loader = DataLoader(
        behavior_full_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    target_full_loader = DataLoader(
        _PairDataset(tokenized_target),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    hidden_size = int(getattr(encoder.config, "hidden_size"))
    model = _EncoderDANNRewardModel(
        encoder=encoder,
        hidden_size=hidden_size,
        head_hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    ).to(device)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    reward_head_params = [p for p in model.reward.parameters() if p.requires_grad]
    domain_head_params = [p for p in model.domain.parameters() if p.requires_grad]
    if not encoder_params or not reward_head_params or not domain_head_params:
        raise ValueError("True DANN requires trainable encoder, reward-head, and domain-head parameters.")
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.learning_rate_encoder},
            {"params": reward_head_params + domain_head_params, "lr": cfg.learning_rate_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    reward_loss_fn = nn.MSELoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    total_training_steps = cfg.max_epochs * len(behavior_loader)
    if total_training_steps <= 0:
        raise RuntimeError("No optimizer steps are available for True DANN training.")
    target_iter = _cycle_loader(target_loader)
    best_state: dict[str, Any] | None = None
    best_metric = float("inf")
    best_epoch = 0
    best_adversarial_updates = 0
    best_complete_nonzero_epochs = 0
    stale_epochs = 0
    global_step = 0
    cumulative_adversarial_updates = 0
    complete_nonzero_epochs = 0
    epoch_diagnostics: list[dict[str, Any]] = []

    for epoch in range(cfg.max_epochs):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        reward_losses: list[float] = []
        source_domain_losses: list[float] = []
        target_domain_losses: list[float] = []
        domain_losses: list[float] = []
        total_losses: list[float] = []
        lambda_values: list[float] = []
        online_domain_probs: list[float] = []
        online_domain_labels: list[int] = []
        nonzero_updates = 0
        gradient_snapshot: dict[str, float] | None = None

        for behavior_batch in behavior_loader:
            behavior_inputs, behavior_rewards = _batch_to_device(behavior_batch, device)
            target_inputs, target_domain_labels = _batch_to_device(next(target_iter), device)
            behavior_z = model.encode(behavior_inputs)
            target_z = model.encode(target_inputs)
            reward_pred = model.reward_from_z(behavior_z)
            reward_loss = _reward_loss_source_only(
                reward_pred,
                behavior_rewards,
                reward_loss_fn,
            )
            lambda_effective = _lambda_for_optimizer_step(
                cfg.lambda_domain,
                global_step,
                total_training_steps,
            )
            (
                balanced_domain_loss,
                source_domain_bce,
                target_domain_bce,
                source_domain_logits,
                target_domain_logits,
            ) = _balanced_domain_loss_components(
                model=model,
                z_behavior=behavior_z,
                z_target=target_z,
                source_domain_labels=torch.zeros(
                    (behavior_z.shape[0], 1),
                    dtype=behavior_z.dtype,
                    device=device,
                ),
                target_domain_labels=target_domain_labels.to(dtype=target_z.dtype),
                lambda_domain=lambda_effective,
                loss_fn=domain_loss_fn,
            )
            total_loss = reward_loss + balanced_domain_loss

            if gradient_snapshot is None:
                reward_encoder_norm, reward_head_norm = _component_gradient_norms(
                    reward_loss,
                    encoder_params,
                    reward_head_params,
                )
                adversarial_encoder_norm, domain_head_norm = _component_gradient_norms(
                    balanced_domain_loss,
                    encoder_params,
                    domain_head_params,
                )
                gradient_snapshot = {
                    "reward_encoder_gradient_norm": reward_encoder_norm,
                    "adversarial_encoder_gradient_norm": adversarial_encoder_norm,
                    "reward_head_gradient_norm": reward_head_norm,
                    "domain_head_gradient_norm": domain_head_norm,
                }

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if lambda_effective <= 0.0:
                raise RuntimeError("DANN optimizer step received zero adversarial lambda.")
            nonzero_updates += 1
            cumulative_adversarial_updates += 1

            reward_losses.append(float(reward_loss.detach().cpu().item()))
            source_domain_losses.append(float(source_domain_bce.detach().cpu().item()))
            target_domain_losses.append(float(target_domain_bce.detach().cpu().item()))
            domain_losses.append(float(balanced_domain_loss.detach().cpu().item()))
            total_losses.append(float(total_loss.detach().cpu().item()))
            lambda_values.append(float(lambda_effective))
            source_probs = torch.sigmoid(source_domain_logits.detach()).cpu().numpy().reshape(-1)
            target_probs = torch.sigmoid(target_domain_logits.detach()).cpu().numpy().reshape(-1)
            online_domain_probs.extend(source_probs.tolist())
            online_domain_probs.extend(target_probs.tolist())
            online_domain_labels.extend([0] * len(source_probs))
            online_domain_labels.extend([1] * len(target_probs))
            memory_tracker.sample()

        if nonzero_updates == len(behavior_loader):
            complete_nonzero_epochs += 1
        val_mse_std, val_mae_std = _validation_reward_metrics(
            model,
            val_loader,
            reward_loss_fn,
            device,
        )
        metric = val_mse_std
        eligible = complete_nonzero_epochs >= 3
        selected = False
        if eligible and metric + 1e-7 < best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            best_adversarial_updates = cumulative_adversarial_updates
            best_complete_nonzero_epochs = complete_nonzero_epochs
            stale_epochs = 0
            selected = True
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif eligible:
            stale_epochs += 1

        memory_tracker.sample()
        online_prob_array = np.asarray(online_domain_probs, dtype=float)
        online_label_array = np.asarray(online_domain_labels, dtype=int)
        online_accuracy = float(
            np.mean((online_prob_array >= 0.5).astype(int) == online_label_array)
        )
        online_auc = _safe_roc_auc(online_label_array, online_prob_array)
        elapsed = time.perf_counter() - epoch_started
        gradient_snapshot = gradient_snapshot or {
            "reward_encoder_gradient_norm": 0.0,
            "adversarial_encoder_gradient_norm": 0.0,
            "reward_head_gradient_norm": 0.0,
            "domain_head_gradient_norm": 0.0,
        }
        epoch_diagnostics.append(
            {
                "epoch": epoch + 1,
                "reward_loss": float(np.mean(reward_losses)),
                "source_domain_bce": float(np.mean(source_domain_losses)),
                "target_domain_bce": float(np.mean(target_domain_losses)),
                "balanced_domain_loss": float(np.mean(domain_losses)),
                "total_loss": float(np.mean(total_losses)),
                "source_validation_reward_mse_standardized": float(val_mse_std),
                "source_validation_reward_mae_standardized": float(val_mae_std),
                "lambda_first": float(lambda_values[0]),
                "lambda_mean": float(np.mean(lambda_values)),
                "lambda_max": float(np.max(lambda_values)),
                "nonzero_adversarial_update_count": int(nonzero_updates),
                "cumulative_adversarial_update_count": int(cumulative_adversarial_updates),
                **gradient_snapshot,
                "online_domain_head_accuracy": online_accuracy,
                "online_domain_head_auc": online_auc,
                "best_checkpoint_eligibility": bool(eligible),
                "best_checkpoint_selected": bool(selected),
                "wall_time_seconds": elapsed,
                "peak_memory_bytes": memory_tracker.diagnostics()["peak_memory_bytes"],
            }
        )
        if epoch + 1 >= 5 and cfg.patience > 0 and stale_epochs >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("No DANN checkpoint satisfied the nonzero-adversarial eligibility gate.")
    if best_adversarial_updates <= 0 or best_complete_nonzero_epochs < 3:
        raise RuntimeError("Best DANN checkpoint did not receive enough nonzero adversarial updates.")
    model.load_state_dict(best_state)
    checkpoint_hash = _state_dict_sha256(best_state)

    behavior_pred_std, behavior_z = _predict(model, behavior_full_loader, device)
    target_pred_std, target_z = _predict(model, target_full_loader, device)
    behavior_pred = behavior_pred_std * y_std + y_mean
    target_pred = target_pred_std * y_std + y_mean
    y_behavior_raw = y_np.reshape(-1)
    diagnostics: dict[str, Any] = {
        "method": "bert512_true_dann_dm",
        "config": asdict(cfg),
        "model_name": cfg.model_name,
        "encoder_class": type(model.encoder).__name__,
        "pooling": cfg.pooling,
        "n_behavior": len(questions_behavior),
        "n_target": len(questions_target),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
        "epochs_trained": len(epoch_diagnostics),
        "best_epoch": int(best_epoch),
        "best_validation_mse_standardized": float(best_metric),
        "best_checkpoint_sha256": checkpoint_hash,
        "best_checkpoint_adversarial_update_count": int(best_adversarial_updates),
        "best_checkpoint_complete_nonzero_lambda_epochs": int(best_complete_nonzero_epochs),
        "train_mse": _mse(behavior_pred[train_idx], y_behavior_raw[train_idx]),
        "validation_mse": _mse(behavior_pred[val_idx], y_behavior_raw[val_idx]),
        "validation_mae": _mae(behavior_pred[val_idx], y_behavior_raw[val_idx]),
        "baseline_mse": _mse(np.full_like(y_behavior_raw, y_mean), y_behavior_raw),
        "y_mean": y_mean,
        "y_std": y_std,
        "uses_source_rewards_for_reward_head": True,
        "uses_source_rewards_for_domain_head": False,
        "uses_target_features_for_domain_adaptation": True,
        "uses_target_domain_labels": True,
        "uses_target_rewards_for_training": False,
        "uses_target_rewards_for_early_stopping": False,
        "uses_target_rewards_for_model_selection": False,
        "uses_target_rewards_for_evaluation": False,
        "lambda_domain": float(cfg.lambda_domain),
        "lambda_schedule": "optimizer_step_logistic",
        "total_training_steps_planned": int(total_training_steps),
        "total_optimizer_steps_completed": int(global_step),
        "cumulative_adversarial_update_count": int(cumulative_adversarial_updates),
        "epoch_times_seconds": [row["wall_time_seconds"] for row in epoch_diagnostics],
        "epoch_diagnostics": epoch_diagnostics,
        "encoder_layer_container": trainability_audit["layer_container"],
        "encoder_layer_count": trainability_audit["layer_count"],
        "unfrozen_encoder_layer_names": trainability_audit["unfrozen_layer_names"],
        "physical_batch_size": int(cfg.batch_size),
        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
        "effective_batch_size": int(cfg.batch_size * cfg.gradient_accumulation_steps),
        "dynamic_padding": bool(cfg.dynamic_padding),
        "total_encoder_parameters": int(sum(p.numel() for p in model.encoder.parameters())),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_encoder_parameters": int(sum(p.numel() for p in encoder_params)),
        "trainable_total_parameters": int(
            sum(p.numel() for p in encoder_params + reward_head_params + domain_head_params)
        ),
        "trainable_parameter_names": [
            name for name, param in model.named_parameters() if param.requires_grad
        ],
        "trainable_non_encoder_parameter_names": [
            name
            for name, param in model.named_parameters()
            if not name.startswith("encoder.") and param.requires_grad
        ],
    }
    diagnostics.update(memory_tracker.diagnostics())
    if cfg.compute_domain_diagnostics:
        diagnostics.update(
            _domain_diagnostics_oof(
                model,
                behavior_z,
                target_z,
                device,
                seed=cfg.seed,
            )
        )
    return {
        "v_hat": float(target_pred.mean()),
        "behavior_pred": behavior_pred.astype(np.float32),
        "target_pred": target_pred.astype(np.float32),
        "diagnostics": diagnostics,
    }


class _PairDataset:
    def __init__(self, encoded: Any, labels: Any | None = None, indices: np.ndarray | None = None):
        import torch

        self.encoded = {}
        for key, value in dict(encoded).items():
            if torch.is_tensor(value):
                self.encoded[key] = value.detach().clone()
            elif isinstance(value, (list, tuple)):
                self.encoded[key] = list(value)
        self.labels = labels.detach().clone() if labels is not None else None
        if not self.encoded:
            raise ValueError("Tokenized pair data did not contain any usable fields.")
        first_value = next(iter(self.encoded.values()))
        encoded_length = int(first_value.shape[0]) if torch.is_tensor(first_value) else len(first_value)
        self.indices = (
            torch.as_tensor(indices, dtype=torch.long)
            if indices is not None
            else torch.arange(encoded_length, dtype=torch.long)
        )

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item].item())
        row = {key: value[idx] for key, value in self.encoded.items()}
        if self.labels is not None:
            row["labels"] = self.labels[idx]
        return row


class _EncoderDANNRewardModel:
    def __new__(cls, *args, **kwargs):
        from torch import nn

        class Model(nn.Module):
            def __init__(self, encoder, hidden_size, head_hidden_dim, dropout, pooling):
                super().__init__()
                self.encoder = encoder
                self.pooling = pooling
                self.reward = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, head_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                self.domain = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, head_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, 1),
                )

            def encode(self, inputs):
                outputs = self.encoder(**inputs)
                hidden = outputs.last_hidden_state
                if self.pooling == "cls":
                    return hidden[:, 0]
                if self.pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            def reward_from_z(self, z):
                return self.reward(z)

            def domain_from_z(self, z, lambda_domain: float):
                return self.domain(_grad_reverse(z, lambda_domain))

        return Model(*args, **kwargs)


def _balanced_domain_loss(model, z_behavior, z_target, lambda_domain: float, loss_fn):
    import torch

    logits_behavior = model.domain_from_z(z_behavior, lambda_domain=lambda_domain)
    logits_target = model.domain_from_z(z_target, lambda_domain=lambda_domain)
    labels_behavior = torch.zeros_like(logits_behavior)
    labels_target = torch.ones_like(logits_target)
    return 0.5 * (
        loss_fn(logits_behavior, labels_behavior)
        + loss_fn(logits_target, labels_target)
    )


def _reward_loss_source_only(reward_predictions, source_rewards, loss_fn):
    """Reward loss boundary: accepts source predictions and source rewards only."""

    if source_rewards is None:
        raise ValueError("Source rewards are required for the reward head.")
    return loss_fn(reward_predictions, source_rewards)


def _balanced_domain_loss_components(
    *,
    model,
    z_behavior,
    z_target,
    source_domain_labels,
    target_domain_labels,
    lambda_domain: float,
    loss_fn,
):
    """Return balanced source/target BCE with lambda applied only by the GRL."""

    logits_behavior = model.domain_from_z(z_behavior, lambda_domain=lambda_domain)
    logits_target = model.domain_from_z(z_target, lambda_domain=lambda_domain)
    source_bce = loss_fn(logits_behavior, source_domain_labels)
    target_bce = loss_fn(logits_target, target_domain_labels)
    balanced = 0.5 * source_bce + 0.5 * target_bce
    return balanced, source_bce, target_bce, logits_behavior, logits_target


def _grad_reverse(x, lambda_domain: float):
    import torch

    class _GradientReverseFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.lambda_domain = float(lambda_domain)
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_domain * grad_output

    return _GradientReverseFunction.apply(x)


def _lambda_for_optimizer_step(lambda_max: float, global_step: int, total_steps: int) -> float:
    if lambda_max < 0.0:
        raise ValueError("lambda_max must be non-negative.")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    p = float(global_step + 1) / float(total_steps)
    return float(lambda_max) * float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def _lambda_for_epoch(cfg: EncoderDANNDMConfig, epoch: int) -> float:
    if epoch < cfg.warmup_epochs:
        return 0.0
    if cfg.lambda_schedule == "constant":
        return float(cfg.lambda_domain)
    if cfg.lambda_schedule == "logistic":
        denom = max(cfg.max_epochs - cfg.warmup_epochs, 1)
        p = float(epoch - cfg.warmup_epochs) / float(denom)
        return float(cfg.lambda_domain) * float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
    raise ValueError(f"Unsupported lambda schedule: {cfg.lambda_schedule}")


def _nested_attribute(root: Any, path: str) -> Any | None:
    value = root
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _discover_encoder_layers(encoder: Any) -> tuple[list[Any], str]:
    """Return the ordered transformer blocks and their architecture-specific path."""

    candidates = (
        "encoder.layer",  # BERT, RoBERTa and related encoders
        "layers",  # ModernBERT, including Ettin encoders
        "encoder.layers",  # common encoder-only implementations
        "transformer.layer",
        "transformer.layers",
        "transformer.h",
        "h",
        "blocks",
    )
    for path in candidates:
        value = _nested_attribute(encoder, path)
        if value is None or isinstance(value, (str, bytes)):
            continue
        try:
            layers = list(value)
        except TypeError:
            continue
        if layers and all(hasattr(layer, "parameters") for layer in layers):
            return layers, path

    model_type = getattr(getattr(encoder, "config", None), "model_type", "unknown")
    raise ValueError(
        "Unable to discover transformer layers for "
        f"{type(encoder).__name__} (model_type={model_type}). Refusing to silently "
        "fall back to full fine-tuning."
    )


def _configure_encoder_trainability(
    encoder: Any,
    unfreeze_last_n_layers: int,
) -> dict[str, Any]:
    for param in encoder.parameters():
        param.requires_grad = False
    if unfreeze_last_n_layers < 0:
        for param in encoder.parameters():
            param.requires_grad = True
        return {
            "layer_container": "all_parameters",
            "layer_count": None,
            "unfrozen_layer_names": ["all_parameters"],
        }
    if unfreeze_last_n_layers == 0:
        return {
            "layer_container": None,
            "layer_count": 0,
            "unfrozen_layer_names": [],
        }

    layers, layer_container = _discover_encoder_layers(encoder)
    if unfreeze_last_n_layers > len(layers):
        raise ValueError(
            f"Requested {unfreeze_last_n_layers} unfrozen layers, but "
            f"{layer_container} contains only {len(layers)}."
        )

    first_unfrozen_index = len(layers) - unfreeze_last_n_layers
    for layer in layers[first_unfrozen_index:]:
        for param in layer.parameters():
            param.requires_grad = True
    return {
        "layer_container": layer_container,
        "layer_count": len(layers),
        "unfrozen_layer_names": [
            f"{layer_container}.{index}"
            for index in range(first_unfrozen_index, len(layers))
        ],
    }


class _MemoryTracker:
    """Best-effort peak memory sampling across process RSS and accelerator memory."""

    def __init__(self, device: str):
        self.device = str(device)
        self.peak_rss_bytes = 0
        self.peak_torch_allocated_bytes = 0
        self.peak_torch_driver_bytes = 0
        self.sample()

    def sample(self) -> None:
        import resource
        import torch

        try:
            import psutil

            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                int(psutil.Process().memory_info().rss),
            )
        except (ImportError, OSError):
            raw_max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            rss_bytes = raw_max_rss if sys.platform == "darwin" else raw_max_rss * 1024
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.peak_torch_allocated_bytes = max(
                self.peak_torch_allocated_bytes,
                int(torch.cuda.max_memory_allocated()),
            )
        elif self.device.startswith("mps"):
            try:
                self.peak_torch_allocated_bytes = max(
                    self.peak_torch_allocated_bytes,
                    int(torch.mps.current_allocated_memory()),
                )
                self.peak_torch_driver_bytes = max(
                    self.peak_torch_driver_bytes,
                    int(torch.mps.driver_allocated_memory()),
                )
            except RuntimeError:
                pass

    def diagnostics(self) -> dict[str, Any]:
        peak_candidates = [
            self.peak_rss_bytes,
            self.peak_torch_allocated_bytes,
            self.peak_torch_driver_bytes,
        ]
        return {
            "peak_memory_bytes": int(max(peak_candidates)),
            "peak_rss_bytes": int(self.peak_rss_bytes),
            "peak_torch_allocated_bytes": int(self.peak_torch_allocated_bytes),
            "peak_torch_driver_bytes": int(self.peak_torch_driver_bytes),
            "peak_memory_measurement": "sampled_high_water_mark",
        }


def _batch_to_device(batch: dict[str, Any], device: str):
    labels = batch.get("labels")
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels"
    }
    if labels is not None:
        # Hugging Face dynamic-padding collators squeeze per-row ``[1]`` labels
        # to ``[B]``. Reward predictions are ``[B, 1]``; restore that shape to
        # prevent MSE broadcasting into an unintended ``[B, B]`` matrix.
        labels = labels.to(device).reshape(-1, 1)
    return inputs, labels


def _cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _validation_mse(model, loader, loss_fn, device: str) -> float:
    import torch

    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = _batch_to_device(batch, device)
            pred = model.reward_from_z(model.encode(inputs))
            losses.append(float(loss_fn(pred, labels).cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def _validation_mse_with_async_mae(model, loader, loss_fn, device: str):
    """Preserve legacy MSE selection while deferring the MAE host sync."""

    import torch

    if len(loader.dataset) == 0:
        return float("nan"), None
    model.eval()
    losses: list[float] = []
    absolute_error_sums = []
    n_values = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = _batch_to_device(batch, device)
            pred = model.reward_from_z(model.encode(inputs))
            absolute_error_sums.append(torch.abs(pred - labels).sum().detach())
            n_values += int(labels.numel())
            losses.append(float(loss_fn(pred, labels).cpu().item()))
    mae_tensor = torch.stack(absolute_error_sums).sum() / max(n_values, 1)
    return float(np.mean(losses)), mae_tensor


def _validation_reward_metrics(model, loader, loss_fn, device: str) -> tuple[float, float]:
    import torch

    if len(loader.dataset) == 0:
        return float("nan"), float("nan")
    model.eval()
    mse_losses: list[float] = []
    absolute_errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = _batch_to_device(batch, device)
            pred = model.reward_from_z(model.encode(inputs))
            mse_losses.append(float(loss_fn(pred, labels).cpu().item()))
            absolute_errors.append(
                np.abs(
                    pred.detach().cpu().numpy().reshape(-1)
                    - labels.detach().cpu().numpy().reshape(-1)
                )
            )
    mae = float(np.concatenate(absolute_errors).mean()) if absolute_errors else float("nan")
    return float(np.mean(mse_losses)), mae


def _predict(model, loader, device: str):
    import torch

    model.eval()
    preds = []
    zs = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = _batch_to_device(batch, device)
            z = model.encode(inputs)
            pred = model.reward_from_z(z)
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            zs.append(z.detach().cpu().numpy())
    return np.concatenate(preds), np.vstack(zs)


def _domain_diagnostics(model, z_behavior: np.ndarray, z_target: np.ndarray, device: str) -> dict[str, Any]:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    x = np.vstack([z_behavior, z_target]).astype(np.float32)
    y = np.concatenate([
        np.zeros(len(z_behavior), dtype=int),
        np.ones(len(z_target), dtype=int),
    ])

    model.eval()
    with torch.no_grad():
        z_t = torch.from_numpy(x).to(device)
        logits = model.domain_from_z(z_t, lambda_domain=0.0).detach().cpu().numpy().reshape(-1)
    domain_prob = 1.0 / (1.0 + np.exp(-logits))
    domain_pred = (domain_prob >= 0.5).astype(int)
    diagnostics = {
        "domain_head_auc": float(roc_auc_score(y, domain_prob)),
        "domain_head_accuracy": float(accuracy_score(y, domain_pred)),
    }

    probe = LogisticRegression(C=0.1, max_iter=2000, random_state=0)
    probe.fit(x, y)
    probe_prob = probe.predict_proba(x)[:, 1]
    probe_pred = (probe_prob >= 0.5).astype(int)
    diagnostics.update(
        {
            "domain_probe_auc": float(roc_auc_score(y, probe_prob)),
            "domain_probe_accuracy": float(accuracy_score(y, probe_pred)),
        }
    )
    return diagnostics


def _domain_diagnostics_oof(
    model,
    z_behavior: np.ndarray,
    z_target: np.ndarray,
    device: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a regularized domain probe using stratified 5-fold OOF predictions."""

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([z_behavior, z_target]).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(len(z_behavior), dtype=int),
            np.ones(len(z_target), dtype=int),
        ]
    )
    model.eval()
    with torch.no_grad():
        z_tensor = torch.from_numpy(x).to(device)
        logits = model.domain_from_z(z_tensor, lambda_domain=0.0).detach().cpu().numpy().reshape(-1)
    logits = np.clip(logits, -40.0, 40.0)
    head_prob = 1.0 / (1.0 + np.exp(-logits))
    head_pred = (head_prob >= 0.5).astype(int)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, random_state=seed),
    )
    probe_prob = cross_val_predict(
        probe,
        x,
        y,
        cv=folds,
        method="predict_proba",
    )[:, 1]
    probe_pred = (probe_prob >= 0.5).astype(int)
    return {
        "domain_head_auc": float(roc_auc_score(y, head_prob)),
        "domain_head_balanced_accuracy": float(balanced_accuracy_score(y, head_pred)),
        "domain_head_log_loss": float(log_loss(y, head_prob, labels=[0, 1])),
        "domain_head_brier_score": float(brier_score_loss(y, head_prob)),
        "domain_probe_protocol": "regularized_logistic_regression_stratified_5_fold_oof",
        "domain_probe_oof_auc": float(roc_auc_score(y, probe_prob)),
        "domain_probe_oof_balanced_accuracy": float(
            balanced_accuracy_score(y, probe_pred)
        ),
        "domain_probe_oof_log_loss": float(log_loss(y, probe_prob, labels=[0, 1])),
        "domain_probe_oof_brier_score": float(brier_score_loss(y, probe_prob)),
        "domain_probe_oof_n": int(len(y)),
    }


def _safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def _component_gradient_norms(loss, encoder_params: list[Any], head_params: list[Any]) -> tuple[float, float]:
    import torch

    parameters = encoder_params + head_params
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    split = len(encoder_params)
    return _gradient_tensor_norm(gradients[:split]), _gradient_tensor_norm(gradients[split:])


def _gradient_tensor_norm(gradients: Any) -> float:
    squares = [float(grad.detach().float().pow(2).sum().cpu().item()) for grad in gradients if grad is not None]
    return float(np.sqrt(np.sum(squares))) if squares else 0.0


def _parameter_gradient_norm(parameters: list[Any]) -> float:
    return _gradient_tensor_norm([parameter.grad for parameter in parameters])


def _parameter_gradient_norm_tensor(parameters: list[Any]):
    import torch

    squares = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        device = parameters[0].device if parameters else "cpu"
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(squares).sum().sqrt().detach()


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _train_val_split(
    n: int,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_fraction <= 0 or n < 5:
        return np.arange(n), np.array([], dtype=int)
    n_val = int(round(n * validation_fraction))
    n_val = min(max(n_val, 1), n - 1)
    idx = rng.permutation(n)
    return idx[n_val:], idx[:n_val]


def _set_seeds(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if len(pred) == 0:
        return float("nan")
    return float(np.mean((pred - truth) ** 2))


def _mae(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if len(pred) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - truth)))


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    model: Any,
    tokenizer: Any,
    cfg: EncoderDANNDMConfig,
    y_mean: float,
    y_std: float,
    diagnostics: dict[str, Any],
) -> None:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = checkpoint_dir / "tokenizer"
    encoder_dir = checkpoint_dir / "encoder"
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(tokenizer_dir)
    if hasattr(model.encoder, "save_pretrained"):
        model.encoder.save_pretrained(encoder_dir)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "diagnostics": diagnostics,
        },
        checkpoint_dir / "model.pt",
    )
    manifest = {
        "model_name": cfg.model_name,
        "pooling": cfg.pooling,
        "checkpoint_file": "model.pt",
        "encoder_dir": "encoder" if encoder_dir.exists() else None,
        "tokenizer_dir": "tokenizer" if tokenizer_dir.exists() else None,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "uses_target_rewards": bool(diagnostics.get("uses_target_rewards", False)),
        "uses_target_features_for_domain_adaptation": bool(
            diagnostics.get("uses_target_features_for_domain_adaptation", False)
        ),
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_inputs(
    *,
    questions_behavior: list[str],
    answers_behavior: list[str],
    y_behavior: np.ndarray,
    questions_target: list[str],
    answers_target: list[str],
    y_target: np.ndarray | None,
    cfg: EncoderDANNDMConfig,
) -> None:
    if len(questions_behavior) != len(answers_behavior) or len(questions_behavior) != len(y_behavior):
        raise ValueError("Behavior questions, answers, and rewards must have the same length.")
    if len(questions_target) != len(answers_target):
        raise ValueError("Target questions and answers must have the same length.")
    if not questions_behavior:
        raise ValueError("At least one behavior row is required.")
    if not questions_target:
        raise ValueError("At least one target row is required.")
    if y_target is not None and len(y_target) != len(questions_target):
        raise ValueError("Target rewards must match target rows when provided.")
    if cfg.pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be 'cls' or 'mean'.")
    if (
        cfg.max_length <= 0
        or cfg.batch_size <= 0
        or cfg.gradient_accumulation_steps <= 0
        or cfg.max_epochs <= 0
    ):
        raise ValueError(
            "max_length, batch_size, gradient_accumulation_steps, and max_epochs "
            "must be positive."
        )
    if cfg.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if not 0 <= cfg.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not 0 <= cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if cfg.lambda_domain < 0:
        raise ValueError("lambda_domain must be non-negative.")
    if cfg.lambda_schedule not in {"constant", "logistic"}:
        raise ValueError("lambda_schedule must be 'constant' or 'logistic'.")

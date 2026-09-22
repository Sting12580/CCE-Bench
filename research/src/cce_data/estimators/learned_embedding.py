"""Reward-informed learned embeddings for real-data OPE.

This implements the "FineTune" style from Learning Action Embeddings for
OPE: keep a pre-defined text embedding fixed, then learn a small supervised
projection from logged clinician rewards. The agent-side reward is never
used here; it is reserved for final evaluation only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LearnedEmbeddingConfig:
    """Configuration for the two-tower reward-informed projection."""

    latent_dim: int = 128
    hidden_dim: int = 128
    merge_strategy: str = "replace"
    pca_dim: int = 0
    max_epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = 0.0
    validation_fraction: float = 0.2
    patience: int = 50
    normalize: bool = True
    seed: int = 0


def learn_reward_informed_embeddings(
    phi_x: np.ndarray,
    phi_a_clinician: np.ndarray,
    phi_a_agent: np.ndarray,
    y_clinician: np.ndarray,
    config: LearnedEmbeddingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Learn low-dimensional context/action embeddings from logged rewards.

    Parameters
    ----------
    phi_x:
        Fixed base embeddings for patient contexts.
    phi_a_clinician:
        Fixed base embeddings for logged clinician actions.
    phi_a_agent:
        Fixed base embeddings for target-agent actions. These are transformed
        by the learned action projector but are not used to train it.
    y_clinician:
        Logged clinician reward. This is the only reward signal used for
        representation learning.
    config:
        Projection/training settings.

    Returns
    -------
    z_x, z_a_clinician, z_a_agent, diagnostics
        Learned embeddings with shape ``(n, d_out)`` plus training diagnostics
        safe to serialize in the Phase 5 JSON report. ``d_out`` is
        ``latent_dim`` for ``merge_strategy='replace'`` and
        ``pca_dim + latent_dim`` for ``merge_strategy='concat-pca'``.
    """
    cfg = config or LearnedEmbeddingConfig()
    _validate_inputs(phi_x, phi_a_clinician, phi_a_agent, y_clinician, cfg)

    import torch
    from torch import nn

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    x_np = np.asarray(phi_x, dtype=np.float32)
    a_np = np.asarray(phi_a_clinician, dtype=np.float32)
    y_np = np.asarray(y_clinician, dtype=np.float32).reshape(-1, 1)

    y_mean = float(y_np.mean())
    y_std = float(y_np.std() + 1e-6)
    y_train_np = (y_np - y_mean) / y_std

    n = len(y_np)
    train_idx, val_idx = _train_val_split(n, cfg.validation_fraction, rng)

    device = "cpu"
    x = torch.from_numpy(x_np).to(device)
    a = torch.from_numpy(a_np).to(device)
    y = torch.from_numpy(y_train_np).to(device)

    model = _build_two_tower_reward_projector(
        context_dim=x_np.shape[1],
        action_dim=a_np.shape[1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    epochs_trained = 0
    stale_epochs = 0

    for epoch in range(cfg.max_epochs):
        epochs_trained = epoch + 1
        model.train()
        optimizer.zero_grad()
        pred = model(x[train_t], a[train_t])
        train_loss = loss_fn(pred, y[train_t])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(val_idx) > 0:
                metric = float(loss_fn(model(x[val_t], a[val_t]), y[val_t]).item())
            else:
                metric = float(train_loss.item())
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

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        learned_x_t = model.encode_context(torch.from_numpy(np.asarray(phi_x, dtype=np.float32)))
        learned_a_cl_t = model.encode_action(
            torch.from_numpy(np.asarray(phi_a_clinician, dtype=np.float32))
        )
        learned_a_ag_t = model.encode_action(
            torch.from_numpy(np.asarray(phi_a_agent, dtype=np.float32))
        )
        if cfg.normalize:
            learned_x_t = torch.nn.functional.normalize(learned_x_t, p=2, dim=1)
            learned_a_cl_t = torch.nn.functional.normalize(learned_a_cl_t, p=2, dim=1)
            learned_a_ag_t = torch.nn.functional.normalize(learned_a_ag_t, p=2, dim=1)

        pred_all = model(x, a).cpu().numpy().reshape(-1) * y_std + y_mean

    learned_x = learned_x_t.cpu().numpy().astype(np.float32)
    learned_a_cl = learned_a_cl_t.cpu().numpy().astype(np.float32)
    learned_a_ag = learned_a_ag_t.cpu().numpy().astype(np.float32)
    if cfg.merge_strategy == "concat-pca":
        z_x, z_a_cl, z_a_ag, pca_diag = _concat_pca_embeddings(
            phi_x=phi_x,
            phi_a_clinician=phi_a_clinician,
            phi_a_agent=phi_a_agent,
            learned_x=learned_x,
            learned_a_clinician=learned_a_cl,
            learned_a_agent=learned_a_ag,
            pca_dim=cfg.pca_dim,
            seed=cfg.seed,
            normalize=cfg.normalize,
        )
    else:
        z_x, z_a_cl, z_a_ag = learned_x, learned_a_cl, learned_a_ag
        pca_diag = None

    y_raw = y_np.reshape(-1)
    train_mse = _mse(pred_all[train_idx], y_raw[train_idx])
    val_mse = _mse(pred_all[val_idx], y_raw[val_idx]) if len(val_idx) > 0 else float("nan")
    baseline_mse = _mse(np.full_like(y_raw, y_mean), y_raw)
    output_dim = int(z_x.shape[1])

    diagnostics = {
        "method": "fine_tune_style_learned_action_embedding",
        "config": asdict(cfg),
        "base_context_dim": int(phi_x.shape[1]),
        "base_action_dim": int(phi_a_clinician.shape[1]),
        "latent_dim": int(cfg.latent_dim),
        "merge_strategy": cfg.merge_strategy,
        "pca_dim": int(cfg.pca_dim),
        "output_embedding_dim": output_dim,
        "feature_dim_after_concat": int(output_dim * 3),
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch + 1),
        "best_validation_mse_standardized": float(best_metric),
        "train_mse": train_mse,
        "validation_mse": val_mse,
        "baseline_mse": baseline_mse,
        "uses_agent_rewards": False,
    }
    if pca_diag is not None:
        diagnostics["pca"] = pca_diag
    return (
        z_x.astype(np.float32),
        z_a_cl.astype(np.float32),
        z_a_ag.astype(np.float32),
        diagnostics,
    )


def learn_unpaired_reward_informed_embeddings(
    phi_x_behavior: np.ndarray,
    phi_a_behavior: np.ndarray,
    phi_x_target: np.ndarray,
    phi_a_target: np.ndarray,
    y_behavior: np.ndarray,
    config: LearnedEmbeddingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Learn reward-informed embeddings for unpaired behavior/target samples.

    This is the same two-tower reward model as ``learn_reward_informed_embeddings``,
    but it supports the train->test setting where logged behavior rows and target
    rows have different lengths. Only behavior rewards are used for fitting.
    """
    cfg = config or LearnedEmbeddingConfig()
    _validate_unpaired_inputs(
        phi_x_behavior,
        phi_a_behavior,
        phi_x_target,
        phi_a_target,
        y_behavior,
        cfg,
    )

    import torch
    from torch import nn

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    x_np = np.asarray(phi_x_behavior, dtype=np.float32)
    a_np = np.asarray(phi_a_behavior, dtype=np.float32)
    y_np = np.asarray(y_behavior, dtype=np.float32).reshape(-1, 1)

    y_mean = float(y_np.mean())
    y_std = float(y_np.std() + 1e-6)
    y_train_np = (y_np - y_mean) / y_std

    n = len(y_np)
    train_idx, val_idx = _train_val_split(n, cfg.validation_fraction, rng)

    device = "cpu"
    x = torch.from_numpy(x_np).to(device)
    a = torch.from_numpy(a_np).to(device)
    y = torch.from_numpy(y_train_np).to(device)

    model = _build_two_tower_reward_projector(
        context_dim=x_np.shape[1],
        action_dim=a_np.shape[1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    epochs_trained = 0
    stale_epochs = 0

    for epoch in range(cfg.max_epochs):
        epochs_trained = epoch + 1
        model.train()
        optimizer.zero_grad()
        pred = model(x[train_t], a[train_t])
        train_loss = loss_fn(pred, y[train_t])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(val_idx) > 0:
                metric = float(loss_fn(model(x[val_t], a[val_t]), y[val_t]).item())
            else:
                metric = float(train_loss.item())
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

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        learned_x_behavior_t = model.encode_context(
            torch.from_numpy(np.asarray(phi_x_behavior, dtype=np.float32))
        )
        learned_a_behavior_t = model.encode_action(
            torch.from_numpy(np.asarray(phi_a_behavior, dtype=np.float32))
        )
        learned_x_target_t = model.encode_context(
            torch.from_numpy(np.asarray(phi_x_target, dtype=np.float32))
        )
        learned_a_target_t = model.encode_action(
            torch.from_numpy(np.asarray(phi_a_target, dtype=np.float32))
        )
        if cfg.normalize:
            learned_x_behavior_t = torch.nn.functional.normalize(learned_x_behavior_t, p=2, dim=1)
            learned_a_behavior_t = torch.nn.functional.normalize(
                learned_a_behavior_t, p=2, dim=1
            )
            learned_x_target_t = torch.nn.functional.normalize(learned_x_target_t, p=2, dim=1)
            learned_a_target_t = torch.nn.functional.normalize(learned_a_target_t, p=2, dim=1)

        pred_all = model(x, a).cpu().numpy().reshape(-1) * y_std + y_mean

    learned_x_behavior = learned_x_behavior_t.cpu().numpy().astype(np.float32)
    learned_a_behavior = learned_a_behavior_t.cpu().numpy().astype(np.float32)
    learned_x_target = learned_x_target_t.cpu().numpy().astype(np.float32)
    learned_a_target = learned_a_target_t.cpu().numpy().astype(np.float32)

    if cfg.merge_strategy == "concat-pca":
        (
            z_x_behavior,
            z_a_behavior,
            z_x_target,
            z_a_target,
            pca_diag,
        ) = _concat_pca_unpaired_embeddings(
            phi_x_behavior=phi_x_behavior,
            phi_a_behavior=phi_a_behavior,
            phi_x_target=phi_x_target,
            phi_a_target=phi_a_target,
            learned_x_behavior=learned_x_behavior,
            learned_a_behavior=learned_a_behavior,
            learned_x_target=learned_x_target,
            learned_a_target=learned_a_target,
            pca_dim=cfg.pca_dim,
            seed=cfg.seed,
            normalize=cfg.normalize,
        )
    else:
        z_x_behavior = learned_x_behavior
        z_a_behavior = learned_a_behavior
        z_x_target = learned_x_target
        z_a_target = learned_a_target
        pca_diag = None

    y_raw = y_np.reshape(-1)
    train_mse = _mse(pred_all[train_idx], y_raw[train_idx])
    val_mse = _mse(pred_all[val_idx], y_raw[val_idx]) if len(val_idx) > 0 else float("nan")
    baseline_mse = _mse(np.full_like(y_raw, y_mean), y_raw)
    output_dim = int(z_x_behavior.shape[1])

    diagnostics = {
        "method": "fine_tune_style_learned_action_embedding_unpaired",
        "config": asdict(cfg),
        "base_context_dim": int(phi_x_behavior.shape[1]),
        "base_action_dim": int(phi_a_behavior.shape[1]),
        "latent_dim": int(cfg.latent_dim),
        "merge_strategy": cfg.merge_strategy,
        "pca_dim": int(cfg.pca_dim),
        "output_embedding_dim": output_dim,
        "feature_dim_after_concat": int(output_dim * 3),
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch + 1),
        "best_validation_mse_standardized": float(best_metric),
        "train_mse": train_mse,
        "validation_mse": val_mse,
        "baseline_mse": baseline_mse,
        "uses_target_rewards": False,
    }
    if pca_diag is not None:
        diagnostics["pca"] = pca_diag
    return (
        z_x_behavior.astype(np.float32),
        z_a_behavior.astype(np.float32),
        z_x_target.astype(np.float32),
        z_a_target.astype(np.float32),
        diagnostics,
    )


def learn_train_only_reward_informed_embeddings(
    phi_x_train: np.ndarray,
    phi_a_train: np.ndarray,
    phi_x_eval: np.ndarray,
    phi_a_eval: np.ndarray,
    y_train: np.ndarray,
    config: LearnedEmbeddingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Learn reward-informed embeddings from train rows and transform eval rows.

    Unlike ``learn_unpaired_reward_informed_embeddings``, this helper fits the
    optional PCA merge only on train split base embeddings. Eval rows are
    transformed by the fitted projector/PCA but their rewards and base feature
    geometry are not used to fit any learned component.
    """
    cfg = config or LearnedEmbeddingConfig()
    _validate_unpaired_inputs(phi_x_train, phi_a_train, phi_x_eval, phi_a_eval, y_train, cfg)

    import torch
    from torch import nn

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    x_np = np.asarray(phi_x_train, dtype=np.float32)
    a_np = np.asarray(phi_a_train, dtype=np.float32)
    y_np = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

    y_mean = float(y_np.mean())
    y_std = float(y_np.std() + 1e-6)
    y_train_np = (y_np - y_mean) / y_std

    n = len(y_np)
    train_idx, val_idx = _train_val_split(n, cfg.validation_fraction, rng)

    device = "cpu"
    x = torch.from_numpy(x_np).to(device)
    a = torch.from_numpy(a_np).to(device)
    y = torch.from_numpy(y_train_np).to(device)

    model = _build_two_tower_reward_projector(
        context_dim=x_np.shape[1],
        action_dim=a_np.shape[1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    epochs_trained = 0
    stale_epochs = 0

    for epoch in range(cfg.max_epochs):
        epochs_trained = epoch + 1
        model.train()
        optimizer.zero_grad()
        pred = model(x[train_t], a[train_t])
        train_loss = loss_fn(pred, y[train_t])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(val_idx) > 0:
                metric = float(loss_fn(model(x[val_t], a[val_t]), y[val_t]).item())
            else:
                metric = float(train_loss.item())
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

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        learned_x_train_t = model.encode_context(
            torch.from_numpy(np.asarray(phi_x_train, dtype=np.float32))
        )
        learned_a_train_t = model.encode_action(
            torch.from_numpy(np.asarray(phi_a_train, dtype=np.float32))
        )
        learned_x_eval_t = model.encode_context(
            torch.from_numpy(np.asarray(phi_x_eval, dtype=np.float32))
        )
        learned_a_eval_t = model.encode_action(
            torch.from_numpy(np.asarray(phi_a_eval, dtype=np.float32))
        )
        if cfg.normalize:
            learned_x_train_t = torch.nn.functional.normalize(learned_x_train_t, p=2, dim=1)
            learned_a_train_t = torch.nn.functional.normalize(learned_a_train_t, p=2, dim=1)
            learned_x_eval_t = torch.nn.functional.normalize(learned_x_eval_t, p=2, dim=1)
            learned_a_eval_t = torch.nn.functional.normalize(learned_a_eval_t, p=2, dim=1)

        pred_all = model(x, a).cpu().numpy().reshape(-1) * y_std + y_mean

    learned_x_train = learned_x_train_t.cpu().numpy().astype(np.float32)
    learned_a_train = learned_a_train_t.cpu().numpy().astype(np.float32)
    learned_x_eval = learned_x_eval_t.cpu().numpy().astype(np.float32)
    learned_a_eval = learned_a_eval_t.cpu().numpy().astype(np.float32)

    if cfg.merge_strategy == "concat-pca":
        (
            z_x_train,
            z_a_train,
            z_x_eval,
            z_a_eval,
            pca_diag,
        ) = _concat_pca_train_eval_embeddings(
            phi_x_train=phi_x_train,
            phi_a_train=phi_a_train,
            phi_x_eval=phi_x_eval,
            phi_a_eval=phi_a_eval,
            learned_x_train=learned_x_train,
            learned_a_train=learned_a_train,
            learned_x_eval=learned_x_eval,
            learned_a_eval=learned_a_eval,
            pca_dim=cfg.pca_dim,
            seed=cfg.seed,
            normalize=cfg.normalize,
        )
    else:
        z_x_train = learned_x_train
        z_a_train = learned_a_train
        z_x_eval = learned_x_eval
        z_a_eval = learned_a_eval
        pca_diag = None

    y_raw = y_np.reshape(-1)
    train_mse = _mse(pred_all[train_idx], y_raw[train_idx])
    val_mse = _mse(pred_all[val_idx], y_raw[val_idx]) if len(val_idx) > 0 else float("nan")
    baseline_mse = _mse(np.full_like(y_raw, y_mean), y_raw)
    output_dim = int(z_x_train.shape[1])

    diagnostics = {
        "method": "fine_tune_style_learned_action_embedding_train_only",
        "config": asdict(cfg),
        "base_context_dim": int(phi_x_train.shape[1]),
        "base_action_dim": int(phi_a_train.shape[1]),
        "latent_dim": int(cfg.latent_dim),
        "merge_strategy": cfg.merge_strategy,
        "pca_dim": int(cfg.pca_dim),
        "output_embedding_dim": output_dim,
        "feature_dim_after_concat": int(output_dim * 3),
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch + 1),
        "best_validation_mse_standardized": float(best_metric),
        "train_mse": train_mse,
        "validation_mse": val_mse,
        "baseline_mse": baseline_mse,
        "uses_eval_rewards": False,
    }
    if pca_diag is not None:
        diagnostics["pca"] = pca_diag
    return (
        z_x_train.astype(np.float32),
        z_a_train.astype(np.float32),
        z_x_eval.astype(np.float32),
        z_a_eval.astype(np.float32),
        diagnostics,
    )


def _build_two_tower_reward_projector(
    context_dim: int,
    action_dim: int,
    latent_dim: int,
    hidden_dim: int,
    dropout: float = 0.0,
):
    """Build the small torch model lazily so importing this file stays light."""
    import torch
    from torch import nn

    class Model(nn.Module):
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

        def encode_context(self, x):
            return self.context(x)

        def encode_action(self, a):
            return self.action(a)

        def forward(self, x, a):
            z_x = self.encode_context(x)
            z_a = self.encode_action(a)
            return self.reward(torch.cat([z_x, z_a, z_x * z_a], dim=1))

    return Model()


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


def _validate_inputs(
    phi_x: np.ndarray,
    phi_a_clinician: np.ndarray,
    phi_a_agent: np.ndarray,
    y_clinician: np.ndarray,
    cfg: LearnedEmbeddingConfig,
) -> None:
    n = len(y_clinician)
    if phi_x.ndim != 2 or phi_a_clinician.ndim != 2 or phi_a_agent.ndim != 2:
        raise ValueError("All embedding inputs must be 2D arrays.")
    if not (len(phi_x) == len(phi_a_clinician) == len(phi_a_agent) == n):
        raise ValueError("Context, clinician action, agent action, and reward lengths differ.")
    if cfg.latent_dim <= 0 or cfg.hidden_dim <= 0:
        raise ValueError("latent_dim and hidden_dim must be positive.")
    if cfg.merge_strategy not in {"replace", "concat-pca"}:
        raise ValueError("merge_strategy must be 'replace' or 'concat-pca'.")
    if cfg.merge_strategy == "concat-pca" and cfg.pca_dim <= 0:
        raise ValueError("pca_dim must be positive when merge_strategy='concat-pca'.")
    if cfg.max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if not 0 <= cfg.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not 0 <= cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")


def _validate_unpaired_inputs(
    phi_x_behavior: np.ndarray,
    phi_a_behavior: np.ndarray,
    phi_x_target: np.ndarray,
    phi_a_target: np.ndarray,
    y_behavior: np.ndarray,
    cfg: LearnedEmbeddingConfig,
) -> None:
    n_behavior = len(y_behavior)
    arrays = {
        "phi_x_behavior": phi_x_behavior,
        "phi_a_behavior": phi_a_behavior,
        "phi_x_target": phi_x_target,
        "phi_a_target": phi_a_target,
    }
    for name, value in arrays.items():
        if value.ndim != 2:
            raise ValueError(f"{name} must be a 2D array.")
    if len(phi_x_behavior) != n_behavior or len(phi_a_behavior) != n_behavior:
        raise ValueError("Behavior context, behavior action, and reward lengths differ.")
    if len(phi_x_target) != len(phi_a_target):
        raise ValueError("Target context and target action lengths differ.")
    if phi_x_behavior.shape[1] != phi_x_target.shape[1]:
        raise ValueError("Behavior and target context embedding dimensions differ.")
    if phi_a_behavior.shape[1] != phi_a_target.shape[1]:
        raise ValueError("Behavior and target action embedding dimensions differ.")
    if cfg.latent_dim <= 0 or cfg.hidden_dim <= 0:
        raise ValueError("latent_dim and hidden_dim must be positive.")
    if cfg.merge_strategy not in {"replace", "concat-pca"}:
        raise ValueError("merge_strategy must be 'replace' or 'concat-pca'.")
    if cfg.merge_strategy == "concat-pca" and cfg.pca_dim <= 0:
        raise ValueError("pca_dim must be positive when merge_strategy='concat-pca'.")
    if cfg.max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if not 0 <= cfg.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not 0 <= cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return float(np.mean((pred - truth) ** 2))


def _concat_pca_embeddings(
    phi_x: np.ndarray,
    phi_a_clinician: np.ndarray,
    phi_a_agent: np.ndarray,
    learned_x: np.ndarray,
    learned_a_clinician: np.ndarray,
    learned_a_agent: np.ndarray,
    pca_dim: int,
    seed: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Preserve base-embedding geometry via PCA and append learned signal."""
    from sklearn.decomposition import PCA

    base = np.vstack([
        np.asarray(phi_x, dtype=np.float32),
        np.asarray(phi_a_clinician, dtype=np.float32),
        np.asarray(phi_a_agent, dtype=np.float32),
    ])
    max_dim = min(base.shape[0], base.shape[1])
    if pca_dim > max_dim:
        raise ValueError(f"pca_dim={pca_dim} exceeds max feasible PCA dim {max_dim}.")

    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(base)
    p_x = pca.transform(phi_x).astype(np.float32)
    p_a_cl = pca.transform(phi_a_clinician).astype(np.float32)
    p_a_ag = pca.transform(phi_a_agent).astype(np.float32)
    if normalize:
        p_x = _l2_normalize(p_x)
        p_a_cl = _l2_normalize(p_a_cl)
        p_a_ag = _l2_normalize(p_a_ag)

    z_x = np.concatenate([p_x, learned_x], axis=1)
    z_a_cl = np.concatenate([p_a_cl, learned_a_clinician], axis=1)
    z_a_ag = np.concatenate([p_a_ag, learned_a_agent], axis=1)
    diag = {
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_components": int(pca_dim),
        "fit_rows": int(base.shape[0]),
    }
    return z_x, z_a_cl, z_a_ag, diag


def _concat_pca_unpaired_embeddings(
    phi_x_behavior: np.ndarray,
    phi_a_behavior: np.ndarray,
    phi_x_target: np.ndarray,
    phi_a_target: np.ndarray,
    learned_x_behavior: np.ndarray,
    learned_a_behavior: np.ndarray,
    learned_x_target: np.ndarray,
    learned_a_target: np.ndarray,
    pca_dim: int,
    seed: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Preserve base geometry for unpaired behavior/target embeddings."""
    from sklearn.decomposition import PCA

    base = np.vstack([
        np.asarray(phi_x_behavior, dtype=np.float32),
        np.asarray(phi_a_behavior, dtype=np.float32),
        np.asarray(phi_x_target, dtype=np.float32),
        np.asarray(phi_a_target, dtype=np.float32),
    ])
    max_dim = min(base.shape[0], base.shape[1])
    if pca_dim > max_dim:
        raise ValueError(f"pca_dim={pca_dim} exceeds max feasible PCA dim {max_dim}.")

    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(base)
    p_x_behavior = pca.transform(phi_x_behavior).astype(np.float32)
    p_a_behavior = pca.transform(phi_a_behavior).astype(np.float32)
    p_x_target = pca.transform(phi_x_target).astype(np.float32)
    p_a_target = pca.transform(phi_a_target).astype(np.float32)
    if normalize:
        p_x_behavior = _l2_normalize(p_x_behavior)
        p_a_behavior = _l2_normalize(p_a_behavior)
        p_x_target = _l2_normalize(p_x_target)
        p_a_target = _l2_normalize(p_a_target)

    z_x_behavior = np.concatenate([p_x_behavior, learned_x_behavior], axis=1)
    z_a_behavior = np.concatenate([p_a_behavior, learned_a_behavior], axis=1)
    z_x_target = np.concatenate([p_x_target, learned_x_target], axis=1)
    z_a_target = np.concatenate([p_a_target, learned_a_target], axis=1)
    diag = {
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_components": int(pca_dim),
        "fit_rows": int(base.shape[0]),
    }
    return z_x_behavior, z_a_behavior, z_x_target, z_a_target, diag


def _concat_pca_train_eval_embeddings(
    phi_x_train: np.ndarray,
    phi_a_train: np.ndarray,
    phi_x_eval: np.ndarray,
    phi_a_eval: np.ndarray,
    learned_x_train: np.ndarray,
    learned_a_train: np.ndarray,
    learned_x_eval: np.ndarray,
    learned_a_eval: np.ndarray,
    pca_dim: int,
    seed: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Preserve base geometry with PCA fitted on train rows only."""
    from sklearn.decomposition import PCA

    base_train = np.vstack([
        np.asarray(phi_x_train, dtype=np.float32),
        np.asarray(phi_a_train, dtype=np.float32),
    ])
    max_dim = min(base_train.shape[0], base_train.shape[1])
    if pca_dim > max_dim:
        raise ValueError(f"pca_dim={pca_dim} exceeds max feasible PCA dim {max_dim}.")

    pca = PCA(n_components=pca_dim, random_state=seed)
    pca.fit(base_train)
    p_x_train = pca.transform(phi_x_train).astype(np.float32)
    p_a_train = pca.transform(phi_a_train).astype(np.float32)
    p_x_eval = pca.transform(phi_x_eval).astype(np.float32)
    p_a_eval = pca.transform(phi_a_eval).astype(np.float32)
    if normalize:
        p_x_train = _l2_normalize(p_x_train)
        p_a_train = _l2_normalize(p_a_train)
        p_x_eval = _l2_normalize(p_x_eval)
        p_a_eval = _l2_normalize(p_a_eval)

    z_x_train = np.concatenate([p_x_train, learned_x_train], axis=1)
    z_a_train = np.concatenate([p_a_train, learned_a_train], axis=1)
    z_x_eval = np.concatenate([p_x_eval, learned_x_eval], axis=1)
    z_a_eval = np.concatenate([p_a_eval, learned_a_eval], axis=1)
    diag = {
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_components": int(pca_dim),
        "fit_rows": int(base_train.shape[0]),
        "fit_scope": "train_only",
    }
    return z_x_train, z_a_train, z_x_eval, z_a_eval, diag


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)
